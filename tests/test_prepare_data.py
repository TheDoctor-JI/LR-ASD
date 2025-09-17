import os
import sys
import json
import math
import argparse
import shutil
import subprocess
import glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

# Ensure repo root is on path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from model.faceDetector.s3fd import S3FD  # face detector


@dataclass
class Track:
    id: str
    start_frame: int
    end_frame: Optional[int]
    bboxes: List[Tuple[int, List[float], float]]  # (frame_idx, [x1,y1,x2,y2], score)
    paths: Dict[str, str] | None = None

    def to_json(self, fps: float) -> Dict:
        end_f = self.end_frame if self.end_frame is not None else (self.start_frame - 1)
        return {
            "id": self.id,
            "start_frame": self.start_frame,
            "end_frame": end_f,
            "start_time": float(self.start_frame / fps),
            "end_time": float((end_f + 1) / fps),
            "bboxes": [
                {"frame": f, "bbox": [float(v) for v in bb], "score": float(sc)}
                for (f, bb, sc) in self.bboxes
            ],
            "paths": (self.paths or {}),
        }


def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    a = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    b = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = a + b - inter + 1e-6
    return float(inter / union) if union > 0 else 0.0


def clamp_bbox(bb: np.ndarray, w: int, h: int) -> np.ndarray:
    x1 = float(max(0, min(w - 1, bb[0])))
    y1 = float(max(0, min(h - 1, bb[1])))
    x2 = float(max(0, min(w - 1, bb[2])))
    y2 = float(max(0, min(h - 1, bb[3])))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_audio(video_path: str, out_wav: str, sr: int = 16000, threads: int = 4) -> None:
    ensure_dir(os.path.dirname(out_wav))
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-qscale:a", "0",
        "-ac", "1",
        "-vn",
        "-threads", str(threads),
        "-ar", str(sr),
        out_wav,
        "-loglevel", "panic",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e}")


def save_frame(frame: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, frame)


def run_detection_and_tracking_from_frames(
    frames_dir: str,
    device: str = 'cuda',
    conf_th: float = 0.9,
    iou_thres: float = 0.5,
    min_track: int = 10,
    num_failed_det: int = 10,
    facedet_scale: float = 0.25,
    forced_fps: float = 25.0,
) -> Tuple[List[Track], Dict]:
    flist = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    if len(flist) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")
    first = cv2.imread(flist[0])
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {flist[0]}")
    height, width = first.shape[:2]

    detector = S3FD(device=device)

    active: Dict[str, Dict] = {}
    finished: List[Track] = []
    next_id = 0

    for frame_idx, fpath in enumerate(flist):
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect faces (match Columbia_test defaults)
        bboxes = detector.detect_faces(img_rgb, conf_th=conf_th, scales=[facedet_scale])
        dets = []
        for b in bboxes:
            x1, y1, x2, y2, sc = b.tolist()
            bb = clamp_bbox(np.array([x1, y1, x2, y2], dtype=np.float32), width, height)
            dets.append((bb, float(sc)))

        # Assign to tracks via greedy IOU
        used = set()
        track_ids = list(active.keys())
        last_boxes = {tid: np.array(active[tid]["bbox"], dtype=np.float32) for tid in track_ids}
        assignments: List[Tuple[str, int]] = []
        for di, (bb, sc) in enumerate(dets):
            best_tid = None
            best_iou = iou_thres
            for tid in track_ids:
                if tid in used:
                    continue
                i = iou(last_boxes[tid], bb)
                if i >= best_iou:
                    best_iou = i
                    best_tid = tid
            if best_tid is not None:
                assignments.append((best_tid, di))
                used.add(best_tid)

        for tid, di in assignments:
            bb, sc = dets[di]
            active[tid]["bbox"] = bb.tolist()
            active[tid]["miss"] = 0
            active[tid]["track"].bboxes.append((frame_idx, bb.tolist(), sc))

        unmatched = [di for di in range(len(dets)) if all(di != adi for _, adi in assignments)]
        for di in unmatched:
            bb, sc = dets[di]
            tid = f"t{next_id}"
            next_id += 1
            tr = Track(id=tid, start_frame=frame_idx, end_frame=None, bboxes=[(frame_idx, bb.tolist(), sc)])
            active[tid] = {"bbox": bb.tolist(), "miss": 0, "track": tr}

        to_remove = []
        for tid, st in active.items():
            if tid in used:
                continue
            st["miss"] += 1
            if st["miss"] > num_failed_det:
                tr: Track = st["track"]
                tr.end_frame = tr.bboxes[-1][0]
                if (tr.end_frame - tr.start_frame + 1) >= min_track:
                    finished.append(tr)
                to_remove.append(tid)
        for tid in to_remove:
            active.pop(tid, None)

    for tid, st in active.items():
        tr: Track = st["track"]
        tr.end_frame = tr.bboxes[-1][0]
        if (tr.end_frame - tr.start_frame + 1) >= min_track:
            finished.append(tr)

    meta = {
        "fps": float(forced_fps),
        "num_frames": int(len(flist)),
        "width": int(width),
        "height": int(height),
    }
    return finished, meta


def write_track_media(
    tracks: List[Track],
    frames_dir: str,
    out_tracks_dir: str,
    fps: float,
    audio_wav: str,
    sample_rate: int = 16000,
    threads: int = 4,
):
    ensure_dir(out_tracks_dir)

    # Build an index for quick bbox lookup
    for tr in tracks:
        tr_dir = os.path.join(out_tracks_dir, tr.id)
        ensure_dir(tr_dir)
        # Persist meta
        meta_path = os.path.join(tr_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "id": tr.id,
                "start_frame": tr.start_frame,
                "end_frame": tr.end_frame,
                "start_time": float(tr.start_frame / fps),
                "end_time": float((tr.end_frame + 1) / fps),
            }, f, indent=2)

        # Crop face video for convenience (224x224)
        # Follow Columbia: write a temporary video, then mux with audio into final face.avi
        face_avi = os.path.join(tr_dir, "face.avi")
        face_avi_tmp = os.path.join(tr_dir, "face_t.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(face_avi_tmp, fourcc, fps, (224, 224))
        bbox_by_frame = {f: (bb, sc) for (f, bb, sc) in tr.bboxes}
        last_bb = None
        for fi in range(tr.start_frame, tr.end_frame + 1):
            frame_path = os.path.join(frames_dir, f"{fi:06d}.jpg")
            img = cv2.imread(frame_path)
            if img is None:
                continue
            if fi in bbox_by_frame:
                last_bb, _ = bbox_by_frame[fi]
            # If missing, reuse last_bb
            if last_bb is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in last_bb]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (224, 224))
            writer.write(roi)
        writer.release()

        # Extract corresponding audio slice
        face_wav = os.path.join(tr_dir, "face.wav")
        ss = tr.start_frame / fps
        to = (tr.end_frame + 1) / fps
        # Match Columbia_test crop audio flags exactly
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_wav,
            "-async", "1",
            "-ac", "1",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-threads", str(threads),
            "-ss", f"{ss:.3f}",
            "-to", f"{to:.3f}",
            face_wav,
            "-loglevel", "panic",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception:
            # Best-effort; continue even if audio clip fails
            pass

        # Mux temp video with audio into final face.avi (match Columbia flags)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", face_avi_tmp,
                "-i", face_wav,
                "-threads", str(threads),
                "-c:v", "copy",
                "-c:a", "copy",
                face_avi,
                "-loglevel", "panic",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # remove temp if mux succeeded
            try:
                os.remove(face_avi_tmp)
            except Exception:
                pass
        except Exception:
            # Fallback: ensure face.avi exists even without audio
            try:
                if os.path.exists(face_avi_tmp):
                    os.replace(face_avi_tmp, face_avi)
            except Exception:
                pass

        # Update track paths
        tr.paths = {
            "face_avi": os.path.relpath(face_avi, start=os.path.dirname(out_tracks_dir)),
            "face_wav": os.path.relpath(face_wav, start=os.path.dirname(out_tracks_dir)),
            "meta": os.path.relpath(meta_path, start=os.path.dirname(out_tracks_dir)),
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare single-scene timeline data with time-stamped tracks")
    parser.add_argument("--video_path", type=str, required=True, help="Input video file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for prepared data")
    parser.add_argument("--device", type=str, default="auto", help="Device for S3FD: auto|cuda|cpu")
    parser.add_argument("--conf_th", type=float, default=0.9, help="Face detection confidence threshold (match Columbia)")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold for tracking association")
    parser.add_argument("--min_track", type=int, default=10, help="Minimum frames to keep a track")
    parser.add_argument("--num_failed_det", type=int, default=10, help="Max missed frames before ending a track")
    parser.add_argument("--threads", type=int, default=10, help="Threads for ffmpeg operations (match Columbia)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--target_fps", type=float, default=25.0, help="Target FPS for CFR AVI conversion (match Columbia)")
    parser.add_argument("--facedet_scale", type=float, default=0.25, help="Face detector scale (match Columbia)")

    args = parser.parse_args()

    out_media = os.path.join(args.out_dir, "media")
    out_frames = os.path.join(out_media, "frames")
    out_tracks = os.path.join(args.out_dir, "tracks")
    ensure_dir(out_frames)
    ensure_dir(out_tracks)

    # Resolve device
    if args.device == "auto":
        try:
            import torch  # lazy import
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"
    else:
        resolved_device = args.device

    # 0) Convert input video to constant-FPS AVI (EXACT flags as Columbia_test)
    video_copy_path = os.path.join(out_media, "video.avi")
    ensure_dir(out_media)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", args.video_path,
            "-qscale:v", "2",
            "-threads", str(args.threads),
            "-async", "1",
            "-r", str(int(args.target_fps)),
            video_copy_path,
            "-loglevel", "panic",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        # Fallback: direct copy if re-encode fails
        try:
            subprocess.run(["ffmpeg", "-y", "-i", args.video_path, "-c:v", "copy", "-c:a", "copy", video_copy_path, "-loglevel", "panic"], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception:
            video_copy_path = os.path.abspath(args.video_path)

    # 1) Extract audio (global timeline)
    audio_wav = os.path.join(out_media, "audio.wav")
    # Extract from the CFR AVI to ensure alignment with frames
    extract_audio(video_copy_path, audio_wav, sr=args.sample_rate, threads=args.threads)

    # 2) Extract frames using ffmpeg (EXACT flags as Columbia_test)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_copy_path,
            "-qscale:v", "2",
            "-threads", str(args.threads),
            "-f", "image2",
            os.path.join(out_frames, "%06d.jpg"),
            "-loglevel", "panic",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed to extract frames: {e}")

    # 3) Detect & track across frames
    tracks, video_meta = run_detection_and_tracking_from_frames(
        out_frames,
        device=resolved_device,
        conf_th=args.conf_th,
        iou_thres=args.iou_thres,
        min_track=args.min_track,
        num_failed_det=args.num_failed_det,
        facedet_scale=float(args.facedet_scale),
        forced_fps=float(args.target_fps),
    )

    # 4) Write per-track convenience media (face crops + audio slices)
    write_track_media(
        tracks,
        frames_dir=out_frames,
        out_tracks_dir=out_tracks,
        fps=video_meta["fps"],
        audio_wav=audio_wav,
        sample_rate=args.sample_rate,
        threads=args.threads,
    )

    # 5) Save tracks.json (time-stamped)
    tracks_json = {
        "video": {
            "path": video_copy_path,
            "fps": video_meta["fps"],
            "num_frames": video_meta["num_frames"],
            "width": video_meta["width"],
            "height": video_meta["height"],
            "audio_path": os.path.abspath(audio_wav),
            "sample_rate": int(args.sample_rate),
        },
        "tracks": [t.to_json(video_meta["fps"]) for t in tracks],
    }
    with open(os.path.join(args.out_dir, "tracks.json"), "w") as f:
        json.dump(tracks_json, f, indent=2)

    print(f"Prepared data written to {args.out_dir}")


if __name__ == "__main__":
    main()
