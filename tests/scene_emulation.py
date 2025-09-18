import os
import sys
import time
import json
import glob
import pickle
import warnings

import cv2
import numpy as np
from scipy.io import wavfile

warnings.filterwarnings("ignore")

INTEGRATE_WITH_ASD = True
REALTIME_EMULATION = False

class SceneEmulator:
    """
    Streams audio/visual samples at physical rate (25 Hz) using prepared data, if 'REALTIME_EMULATION' is True. Otherwise, we send chunks of 40ms audio and 25hz frames, but the interval between ticks is as fast as possible but might be longer than 40ms due to inference calling

    Expects under base_dir (matches data_preparation.py):
      base_dir/
        media/
          audio.wav
          video.avi
        frames/
          000001.jpg, 000002.jpg, ...
        intermediate/
          tracks.pckl
        metadata.json

    Handlers:
      - create_track_handle(track_id) -> person_id
      - delete_track_handle(person_id) -> None
      - send_audio_sample_handle(chunk, t_frame, t_sec_start, t_sec_end) -> None
          chunk: np.ndarray (640,), int16 (40 ms @ 16 kHz)
      - send_visual_sample_handle(person_id, face_img_224_bgr, t_frame, t_sec) -> None
          face_img_224_bgr: np.ndarray (224, 224, 3), uint8 (BGR)
      - run_inference_handle() -> Any
        Called once per tick after pushing all samples.

    Notes:
      - Track creation occurs at the first frame of the track.
      - Track deletion occurs AFTER sending the last frame for that track.
    """

    def __init__(
        self,
        base_dir,
        send_audio_sample_handle,
        send_visual_sample_handle,
        create_track_handle,
        delete_track_handle,
        run_inference_handle=None,
        crop_scale=0.40,
    ):
        self.base_dir = base_dir
        self.media_dir = os.path.join(base_dir, "media")
        self.frames_dir = os.path.join(base_dir, "frames")
        self.intermediate_dir = os.path.join(base_dir, "intermediate")
        self.meta_path = os.path.join(base_dir, "metadata.json")
        self.tracks_path = os.path.join(self.intermediate_dir, "tracks.pckl")

        if not os.path.isfile(self.meta_path):
            raise FileNotFoundError(f"metadata.json not found at {self.meta_path}")
        if not os.path.isfile(self.tracks_path):
            raise FileNotFoundError(f"tracks.pckl not found at {self.tracks_path}")

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.fps = float(self.meta.get("fps", 25.0))
        self.dt = 1.0 / self.fps  # 0.04s
        self.crop_scale = float(crop_scale)  # must match data_preparation CROP_SCALE for identical crops

        # Frames list (0-based indexing via list index)
        self.flist = sorted(glob.glob(os.path.join(self.frames_dir, "*.jpg")))
        if not self.flist:
            raise FileNotFoundError(f"No frames found in {self.frames_dir}")
        self.num_frames = len(self.flist)

        # Audio (mono 16 kHz)
        audio_path = os.path.join(self.media_dir, "audio.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        self.sr, self.audio = wavfile.read(audio_path)  # dtype=int16
        if self.sr != 16000:
            raise ValueError(f"Expected 16 kHz audio, found {self.sr}")
        # Samples per 25 Hz tick (40 ms)
        self.samples_per_tick = int(round(self.sr / self.fps))  # 640

        # Tracks with smoothed center/size for consistent crops
        with open(self.tracks_path, "rb") as f:
            tracks_pack = pickle.load(f)  # list of {'track': {...}, 'proc_track': {...}}

        # Build internal track list and indices
        self.tracks = []
        self.tracks_by_id = {}
        self.starts_at = {}  # frame -> [track dicts]
        self.ends_at = {}    # frame -> [track dicts]
        for idx, item in enumerate(tracks_pack):
            tr = item.get("track", {})
            pr = item.get("proc_track", None)
            frames = np.asarray(tr.get("frame"))
            bboxes = np.asarray(tr.get("bbox")) if "bbox" in tr else None
            if frames.size == 0:
                continue
            # proc_track holds smoothed size/center, same indexing as frames
            if pr is not None and all(k in pr for k in ("x", "y", "s")):
                px = np.asarray(pr["x"])
                py = np.asarray(pr["y"])
                ps = np.asarray(pr["s"])
            else:
                if bboxes is None:
                    raise ValueError("Track missing both proc_track and bbox.")
                ps = np.maximum(bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]) / 2.0
                py = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
                px = (bboxes[:, 0] + bboxes[:, 2]) / 2.0

            track = {
                "id": idx,  # temporary; will reassign after sort
                "frames": frames.astype(np.int32),   # absolute frame indices (0-based)
                "x": px.astype(np.float32),
                "y": py.astype(np.float32),
                "s": ps.astype(np.float32),
                "start": int(frames[0]),
                "end": int(frames[-1]),
            }
            self.tracks.append(track)

        # Sort tracks by start time to keep id->time order consistent
        self.tracks.sort(key=lambda t: t["start"])
        # Re-assign ids sequentially after sort
        for new_id, tr in enumerate(self.tracks):
            tr["id"] = new_id
            self.tracks_by_id[new_id] = tr
            self.starts_at.setdefault(tr["start"], []).append(tr)
            self.ends_at.setdefault(tr["end"], []).append(tr)

        # Handlers
        self.send_audio_sample_handle = send_audio_sample_handle
        self.send_visual_sample_handle = send_visual_sample_handle
        self.create_track_handle = create_track_handle
        self.delete_track_handle = delete_track_handle
        self.run_inference_handle = run_inference_handle

        # Mapping from track_id -> person_id as returned by create_track_handle
        self.track_to_person = {}
        # Active set of track ids
        self.active_track_ids = set()

    def _crop_face_224(self, img_bgr, cx, cy, s):
        """
        Reproduce crop_video logic:
          - pad with constant 110
          - crop around (cx, cy) with padded window based on s and crop_scale
          - resize to 224x224
        """
        cs = self.crop_scale
        bs = float(s)
        bsi = int(bs * (1 + 2 * cs))  # pad amount

        frame = np.pad(img_bgr, ((bsi, bsi), (bsi, bsi), (0, 0)), mode='constant', constant_values=(110, 110))
        my = float(cy) + bsi
        mx = float(cx) + bsi

        y1 = int(my - bs)
        y2 = int(my + bs * (1 + 2 * cs))
        x1 = int(mx - bs * (1 + cs))
        x2 = int(mx + bs * (1 + cs))

        h, w = frame.shape[:2]
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            fh, fw = img_bgr.shape[:2]
            y_mid, x_mid = fh // 2, fw // 2
            hw = min(fh, fw) // 4
            face = img_bgr[max(0, y_mid - hw):min(fh, y_mid + hw),
                           max(0, x_mid - hw):min(fw, x_mid + hw)]
        face = cv2.resize(face, (224, 224))
        return face

    def run(self, start_frame=0, end_frame=None):
        """
        Run emulation at 25 Hz. For each frame:
          - send 40 ms audio chunk once
          - create new tracks starting at this frame
          - send one visual sample per active track
          - delete tracks that end at this frame (after sending their last sample)
          - call run_inference_handle (if provided)
        If REALTIME_EMULATION = True, sleep to maintain 25 Hz wall-clock rate. if not, run as fast as possible and allow the interval to be longer than 40 ms if needed (e.g. due to inference time)

        Arguments:
          start_frame: start streaming from this frame (0-based)
          end_frame: stop before this frame (defaults to total frames)
        """
        t_start = int(start_frame)
        t_end = int(end_frame) if end_frame is not None else self.num_frames

        for t in range(t_start, min(t_end, self.num_frames)):
            wall_tick_start = time.perf_counter()

            # 1) Audio for this 40 ms interval
            a0 = t * self.samples_per_tick
            a1 = a0 + self.samples_per_tick
            chunk = self.audio[a0:a1]
            if chunk.shape[0] < self.samples_per_tick:
                pad = np.zeros((self.samples_per_tick - chunk.shape[0],), dtype=self.audio.dtype)
                chunk = np.concatenate([chunk, pad], axis=0)

            t_sec_start = t / self.fps
            t_sec_end = (t + 1) / self.fps
            self.send_audio_sample_handle(chunk, t, t_sec_start, t_sec_end)

            # 2) Create new tracks that start now
            for tr in self.starts_at.get(t, []):
                # Let the handler create a corresponding person; store mapping
                pid = self.create_track_handle(tr["id"])
                if pid is None:
                    # If handler does not return an id, default to using track id
                    pid = tr["id"]
                self.track_to_person[tr["id"]] = pid
                self.active_track_ids.add(tr["id"])

            # 3) Visual samples for all currently active tracks
            img_bgr = cv2.imread(self.flist[t])
            if img_bgr is not None:
                for track_id in list(self.active_track_ids):
                    tr = self.tracks_by_id[track_id]
                    if not (tr["start"] <= t <= tr["end"]):
                        continue
                    idx = t - tr["start"]  # index into this track's arrays
                    cx = tr["x"][idx]
                    cy = tr["y"][idx]
                    s = tr["s"][idx]
                    face_224 = self._crop_face_224(img_bgr, cx, cy, s)
                    person_id = self.track_to_person.get(track_id, track_id)
                    self.send_visual_sample_handle(person_id, face_224, t, t_sec_start)

            # 4) Delete tracks that finish at this frame (after sending last sample)
            for tr in self.ends_at.get(t, []):
                tid = tr["id"]
                if tid in self.active_track_ids:
                    person_id = self.track_to_person.get(tid, tid)
                    self.delete_track_handle(person_id)
                    self.active_track_ids.discard(tid)
                    self.track_to_person.pop(tid, None)

            # 5) Run inference once per tick (optional)
            if self.run_inference_handle is not None:
                _ = self.run_inference_handle()

            if REALTIME_EMULATION:
                self._sleep_to_rate(wall_tick_start)

    def _sleep_to_rate(self, tick_start_time):
        elapsed = time.perf_counter() - tick_start_time
        remaining = self.dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _print_audio(chunk, t, t0, t1):
    # Example audio handler
    print(f"[A] t={t:06d} {t0:.3f}-{t1:.3f}s samples={chunk.shape[0]}")


def _print_visual(person_id, face_img, t, t_sec):
    # Example visual handler
    print(f"[V] t={t:06d} {t_sec:.3f}s person={person_id} face={tuple(face_img.shape)}")


def _create_track(track_id):
    # Example: identity mapping
    print(f"[+] create track {track_id}")
    return track_id


def _delete_track(person_id):
    # Example delete
    print(f"[-] delete person {person_id}")


if __name__ == "__main__":
    # Minimal demo usage:
    base = os.path.join("demo", "0003")
    if len(sys.argv) > 1:
        base = sys.argv[1]

    if INTEGRATE_WITH_ASD:
        from realtime_causal_asd import RealtimeCausalASD
        asd_rt = RealtimeCausalASD()
        emulator = SceneEmulator(
            base_dir=base,
            send_audio_sample_handle=asd_rt.push_audio_sample,
            send_visual_sample_handle=asd_rt.push_visual_sample,  # requires person_id
            create_track_handle=asd_rt.create_person,
            delete_track_handle=asd_rt.delete_person,
            run_inference_handle=asd_rt.run_inference,
            crop_scale=0.40,
        )
    else:
        emulator = SceneEmulator(
            base_dir=base,
            send_audio_sample_handle=_print_audio,
            send_visual_sample_handle=_print_visual,
            create_track_handle=_create_track,
            delete_track_handle=_delete_track,
            run_inference_handle=None,  # replace with ASD.run_inference if desired
            crop_scale=0.40,  # must match data_preparation.CROP_SCALE
        )
    emulator.run()