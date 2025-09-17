import time
import numpy as np
import torch
from collections import deque
from typing import Optional, Dict, List

import cv2
from scipy.signal import resample_poly
import python_speech_features

from ASD import ASD


class ASDRealtime:
    """
    Realtime wrapper around LR-ASD with multi-track support.

    Responsibilities:
    - Maintain a shared sliding window of recent raw audio samples.
    - Maintain per-track sliding windows of pre-cropped face frames.
    - Perform batched inference across all tracks when infer() is called.

    Expects:
    - Audio: raw PCM (int16 or float32) at sample_rate (default 16kHz).
    - Video: per-track, pre-cropped face frames. Either 112x112 grayscale, or any BGR image
      which will be converted to 112x112 grayscale with center-crop.

    Not included in this class:
    - Scene detection, face detection, or tracking/cropping. Supply already-cropped face frames
      per track from your own tracking pipeline.

    Notes:
    - Preprocessing matches the demo: MFCC for audio (numcep=13), grayscale 112x112 faces at 25 fps.
    - infer() batches all ready tracks (sufficient context) and returns per-track score series.
    """

    def __init__(
        self,
        model_path: str = "weight/pretrain_AVA.model",
        window_seconds: float = 10.0,
        video_fps: float = 25.0,
        sample_rate: int = 16000,
        score_threshold: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.video_fps = float(video_fps)
        self.window_seconds = float(window_seconds)
        self.score_threshold = float(score_threshold)

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # LR-ASD model
        self._asd = ASD()
        self._asd.loadParameters(model_path)
        self._asd.eval()

        # Sliding window capacities
        self._window_video_frames = int(round(self.window_seconds * self.video_fps))
        self._window_audio_samples = int(round(self.window_seconds * self.sample_rate))

        # Per-track visual buffers
        self._tracks: Dict[str, deque[np.ndarray]] = {}

        # Shared audio buffer (list of chunks)
        self._audio_buffers: deque[np.ndarray] = deque()
        self._audio_len: int = 0

        # Last outputs (per-track)
        self.last_score: Dict[str, Optional[float]] = {}
        self.last_decision: Dict[str, Optional[bool]] = {}

    # ------------------------- Track Management -------------------------

    def add_track(self, track_id: str) -> None:
        if track_id not in self._tracks:
            self._tracks[track_id] = deque(maxlen=self._window_video_frames)
            self.last_score[track_id] = None
            self.last_decision[track_id] = None

    def remove_track(self, track_id: str) -> None:
        if track_id in self._tracks:
            del self._tracks[track_id]
        self.last_score.pop(track_id, None)
        self.last_decision.pop(track_id, None)

    def list_tracks(self) -> List[str]:
        return list(self._tracks.keys())

    # ------------------------- Streaming Inputs -------------------------

    def push_video_frame(self, track_id: str, frame: np.ndarray) -> None:
        """Push a single face-cropped frame for a specific track."""
        if track_id not in self._tracks:
            # Auto-create track if not present
            self.add_track(track_id)
        proc = self._preprocess_frame(frame)
        self._tracks[track_id].append(proc)

    def push_audio_samples(self, samples: np.ndarray, sr: Optional[int] = None) -> None:
        """Push raw PCM audio samples shared by all tracks."""
        if samples is None or len(samples) == 0:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sr is not None and sr != self.sample_rate:
            up = self.sample_rate
            down = int(sr)
            samples = resample_poly(samples, up, down).astype(np.float32)
        # Clip to window size by dropping old chunks
        self._audio_buffers.append(samples)
        self._audio_len += len(samples)
        while self._audio_len > self._window_audio_samples and len(self._audio_buffers) > 0:
            drop = self._audio_len - self._window_audio_samples
            head = self._audio_buffers[0]
            if len(head) <= drop:
                self._audio_buffers.popleft()
                self._audio_len -= len(head)
            else:
                self._audio_buffers[0] = head[drop:]
                self._audio_len -= drop
                break

    # ------------------------- Inference -------------------------

    def infer(self, min_required_seconds: float = 0.5) -> Dict[str, Dict[str, np.ndarray | float | bool]]:
        """
        Run batched inference across all tracks that have at least min_required_seconds of video
        and sufficient shared audio context in the current sliding window.

        Returns a dict keyed by track_id, value is a dict with:
          - 'series': np.ndarray of per-frame scores for the aligned window (shape [T])
          - 'last': float score for the last frame in the window
          - 'decision': bool, last >= score_threshold
        Tracks without enough context are omitted from the result.
        """
        ready_tracks = [
            tid for tid, buf in self._tracks.items()
            if len(buf) >= int(max(2, round(min_required_seconds * self.video_fps)))
        ]
        if len(ready_tracks) == 0:
            return {}

        # Build audio MFCC once
        audio = self._concat_audio().astype(np.float32)
        if audio.size == 0:
            return {}
        mfcc = python_speech_features.mfcc(
            audio,
            self.sample_rate,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        if mfcc.shape[0] < 4:
            return {}

        # Determine common length_sec across tracks and audio (as in demo)
        lengths_sec = []
        for tid in ready_tracks:
            vlen = len(self._tracks[tid])
            lengths_sec.append(vlen / self.video_fps)
        audio_sec = (mfcc.shape[0] - (mfcc.shape[0] % 4)) / 100.0
        length_sec = min(min(lengths_sec), audio_sec, self.window_seconds)
        if length_sec <= 0:
            return {}

        a_len = int(round(length_sec * 100))
        a_len -= (a_len % 4)
        v_len = int(round(length_sec * self.video_fps))
        if a_len <= 0 or v_len <= 0:
            return {}

        # Slice audio MFCC tail and build video batch tail-aligned
        mfcc = mfcc[-a_len:, :]
        video_batch: List[np.ndarray] = []
        for tid in ready_tracks:
            vbuf = np.stack(list(self._tracks[tid]), axis=0)
            video_batch.append(vbuf[-v_len:, :, :])
        video_batch_np = np.stack(video_batch, axis=0)  # (B, T, 112, 112)

        # Tile audio MFCC across batch (shared audio)
        B = video_batch_np.shape[0]
        audio_batch_np = np.tile(mfcc[np.newaxis, :, :], (B, 1, 1))  # (B, a_len, 13)

        # Run model
        with torch.no_grad():
            inputA = torch.from_numpy(audio_batch_np).float().to(self.device)  # (B, Ta, 13)
            inputV = torch.from_numpy(video_batch_np).float().to(self.device)  # (B, Tv, 112, 112)

            embedA = self._asd.model.forward_audio_frontend(inputA)
            embedV = self._asd.model.forward_visual_frontend(inputV)
            out = self._asd.model.forward_audio_visual_backend(embedA, embedV)
            scores_flat = self._asd.lossAV.forward(out, labels=None)  # numpy 1D length B*T

        # Reshape back to (B, T)
        scores_flat = np.asarray(scores_flat)
        if scores_flat.size != B * v_len:
            # Fallback: try to infer T from size
            if B > 0 and scores_flat.size % B == 0:
                v_len = scores_flat.size // B
            else:
                # Unexpected shape; return empty
                return {}
        scores_bt = scores_flat.reshape(B, v_len)

        results: Dict[str, Dict[str, np.ndarray | float | bool]] = {}
        for i, tid in enumerate(ready_tracks):
            series = scores_bt[i]
            last = float(series[-1])
            decision = bool(last >= self.score_threshold)
            self.last_score[tid] = last
            self.last_decision[tid] = decision
            results[tid] = {
                'series': series,
                'last': last,
                'decision': decision,
            }
        return results

    # ------------------------- Utilities -------------------------

    def set_threshold(self, thr: float) -> None:
        self.score_threshold = float(thr)

    def reset(self) -> None:
        self._tracks.clear()
        self._audio_buffers.clear()
        self._audio_len = 0
        self.last_score.clear()
        self.last_decision.clear()

    # ------------------------- Internals -------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Unexpected frame shape: %s" % (frame.shape,))
        h, w = gray.shape[:2]
        if h != 224 or w != 224:
            gray = cv2.resize(gray, (224, 224))
        c = 112
        half = c // 2
        cropped = gray[c - half : c + half, c - half : c + half]
        return cropped.astype(np.uint8)

    def _concat_audio(self) -> np.ndarray:
        if len(self._audio_buffers) == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.concatenate(list(self._audio_buffers), axis=0)
        if len(arr) > self._window_audio_samples:
            arr = arr[-self._window_audio_samples :]
        return arr
