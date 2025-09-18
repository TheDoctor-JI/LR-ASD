import os
import time
import glob
import json
import numpy as np
import cv2
from scipy.io import wavfile
import sys
import threading
import shutil
import subprocess
from typing import Dict, List
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from realtime_asd import ASDRealtime

# Optional audio playback backends
try:
    import sounddevice as sd  # playback with PortAudio
except Exception:
    sd = None
try:
    import simpleaudio as sa  # lightweight fallback
except Exception:
    sa = None



VISUALIZE_LOCALLY = False

def stream_prepared_timeline(out_dir: str, model_path: str = "weight/pretrain_AVA.model", trigger_interval_s: float = 0.1):
    """
    Emulate realtime streaming using prepared single-scene timeline data.

    Loads tracks.json and media/audio.wav, then streams frames at FPS and pushes
    audio globally and per-track cropped frames based on time-stamped bboxes.
    Tracks are added/removed dynamically according to start/end frames.
    """
    tracks_json_path = os.path.join(out_dir, "tracks.json")
    with open(tracks_json_path, "r") as f:
        data = json.load(f)

    fps = float(data["video"]["fps"]) if "fps" in data["video"] else 25.0
    frame_period = 1.0 / fps
    frames_dir = os.path.join(out_dir, "media", "frames")
    audio_path = data["video"]["audio_path"] if "audio_path" in data["video"] else os.path.join(out_dir, "media", "audio.wav")
    sr, audio = wavfile.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio /= 32768.0

    # Initialize realtime ASD
    asd = ASDRealtime(
        model_path=model_path,
        window_seconds=10.0,
        video_fps=fps,
        score_threshold=0,
    )

    # Build per-frame bbox lists for quick access
    num_frames = int(data["video"].get("num_frames", 0))
    tracks = data["tracks"]
    # Index tracks by start/end
    starts: Dict[int, List[Dict]] = {}
    ends: Dict[int, List[Dict]] = {}
    bbox_by_frame: Dict[str, Dict[int, List[float]]] = {}
    for tr in tracks:
        sf = int(tr["start_frame"]) ; ef = int(tr["end_frame"]) ; tid = tr["id"]
        starts.setdefault(sf, []).append(tr)
        ends.setdefault(ef, []).append(tr)
        # Build bbox map
        bbmap = {}
        for item in tr["bboxes"]:
            bbmap[int(item["frame"]) ] = list(item["bbox"])
        bbox_by_frame[tid] = bbmap

    # Viz setup
    if VISUALIZE_LOCALLY:
        cv2.namedWindow("LR-ASD Realtime", cv2.WINDOW_NORMAL)
    out_dir_vis = os.path.join(out_dir, "realtime_vis")
    os.makedirs(out_dir_vis, exist_ok=True)
    out_path = os.path.join(out_dir_vis, "timeline_realtime_vis.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None
    # Cache audio actually sent to ASD for muxing later
    pushed_audio_chunks: List[np.ndarray] = []

    # Shared results
    results: Dict[str, Dict[str, float | bool | np.ndarray | None]] = {}
    stop_evt = threading.Event()

    def processor_loop():
        last_trigger_time = time.time()
        while not stop_evt.is_set():
            now = time.time()
            if now - last_trigger_time >= trigger_interval_s:
                res = asd.infer(min_required_seconds=0.5)
                t_after_infer = time.time()
                print(f"Infer took {t_after_infer - now:.3f}s")
                for tid, r in res.items():
                    series = r['series']
                    N = max(1, int(round(trigger_interval_s * fps)))
                    avg_score = float(np.mean(series[-N:]))
                    results[tid] = {
                        "score": avg_score,
                        "decision": bool(avg_score >= asd.score_threshold),
                        "series": series,
                    }
                last_trigger_time = now
            time.sleep(0.0025)

    proc_thread = threading.Thread(target=processor_loop)
    proc_thread.start()

    # Streaming loop across global timeline
    audio_per_frame = int(round(sr * frame_period))
    active_tids: set[str] = set()
    last_bb: Dict[str, List[float]] = {}

    for fi in range(num_frames):
        loop_start = time.time()
        frame_path = os.path.join(frames_dir, f"{fi:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            # still maintain cadence
            time.sleep(max(0, frame_period - (time.time() - loop_start)))
            continue

        # Handle track starts
        for tr in starts.get(fi, []):
            tid = tr["id"]
            if tid not in active_tids:
                asd.add_track(tid)
                active_tids.add(tid)
                last_bb.pop(tid, None)

        # Handle track ends (we push frames for this frame before removal if needed)
        end_now = [tr for tr in ends.get(fi, []) if tr["id"] in active_tids]

        # Push audio slice for this frame
        a_start = fi * audio_per_frame
        a_end = min(len(audio), (fi + 1) * audio_per_frame)
        if a_start < a_end:
            chunk = audio[a_start:a_end]
            asd.push_audio_samples(chunk, sr=sr)
            pushed_audio_chunks.append(chunk)

        # Push per-track cropped frame for currently active tracks
        for tid in list(active_tids):
            bbmap = bbox_by_frame.get(tid, {})
            if fi in bbmap:
                last_bb[tid] = bbmap[fi]
            bb = last_bb.get(tid)
            if bb is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bb]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            asd.push_video_frame(tid, roi)

        # Remove tracks ending at this frame
        for tr in end_now:
            tid = tr["id"]
            if tid in active_tids:
                active_tids.remove(tid)
                asd.remove_track(tid)
                last_bb.pop(tid, None)

        # Build visualization
        vis = frame.copy()
        for tid in active_tids:
            bb = last_bb.get(tid)
            if bb is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bb]

            r = results.get(tid, {})
            score = r.get("score", None)
            decision = r.get("decision", False)

            # Visualize the results on the frame
            # Color: green if speaking, red if not
            color = (0, 255, 0) if decision else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 5)
            txt = f"{tid} ({score:.2f}) Speaking={int(decision)}" if score is not None else f"{tid} warming up"
            cv2.putText(vis, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

        if writer is None:
            h_vis, w_vis = vis.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w_vis, h_vis))
            if not writer.isOpened():
                print(f"Warning: cannot open writer for {out_path}")
                writer = None
        if writer is not None:
            writer.write(vis)

        if VISUALIZE_LOCALLY:
            cv2.imshow("LR-ASD Realtime", vis)
            elapsed = time.time() - loop_start
            delay_ms = max(1, int((frame_period - elapsed) * 1000)) if elapsed < frame_period else 1
            key = cv2.waitKey(delay_ms)
            if key & 0xFF == ord('q'):
                break
        else:
            elapsed = time.time() - loop_start
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

    # Cleanup
    stop_evt.set()
    proc_thread.join(timeout=1.0)
    if writer is not None:
        try:
            writer.release()
            print(f"Saved visualization: {out_path}")
        except Exception:
            pass
    # Mux pushed audio into the saved visualization video
    try:
        if pushed_audio_chunks and os.path.exists(out_path):
            audio_stream = np.concatenate(pushed_audio_chunks, axis=0)
            out_base, out_ext = os.path.splitext(out_path)
            out_mux_tmp = f"{out_base}__mux_tmp{out_ext}"
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is not None:
                tmp_wav = f"{out_base}_tmp_audio.wav"
                try:
                    wavfile.write(tmp_wav, sr, (np.clip(audio_stream, -1.0, 1.0) * 32767.0).astype(np.int16))
                    cmd = [
                        ffmpeg_path, "-y",
                        "-i", out_path,
                        "-i", tmp_wav,
                        "-c:v", "copy",
                        "-c:a", "pcm_s16le",
                        "-shortest",
                        out_mux_tmp,
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if proc.returncode == 0 and os.path.exists(out_mux_tmp):
                        os.replace(out_mux_tmp, out_path)
                        print(f"Muxed audio into video: {out_path}")
                    else:
                        print("ffmpeg muxing failed for visualization")
                finally:
                    try:
                        if os.path.exists(tmp_wav):
                            os.remove(tmp_wav)
                    except Exception:
                        pass
    except Exception as e:
        print(f"Audio muxing encountered an error: {e}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example: use prepared data under demo/0002_prepared (run tests/test_prepare_data.py first)
    prep_dir = os.path.join("demo", "0002_prepared")
    if os.path.exists(os.path.join(prep_dir, "tracks.json")):
        stream_prepared_timeline(prep_dir)
    else:
        print(f"Prepared directory not found: {prep_dir}. Run tests/test_prepare_data.py to generate it.")
