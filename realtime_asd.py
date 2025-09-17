import time
import numpy as np
import torch
from collections import deque
from typing import Optional, Tuple

import cv2
from scipy.signal import resample_poly
import python_speech_features

from ASD import ASD


class ASDRealtime:
    """
    Realtime wrapper around LR-ASD.

    Responsibilities:
    - Maintain sliding windows of recent raw audio samples and pre-cropped face frames.
    - Run inference when step() is called by the external application.

    Expects:
    - Audio: raw PCM (int16 or float32) at sample_rate (default 16kHz).
    - Video: pre-cropped, face-centric frames. Provide either 112x112 grayscale, or any BGR image
      which will be converted to 112x112 grayscale with center-crop.

    NOT included in this class:
    - Scene detection, face detection, or tracking/cropping. Supply already-cropped face frames.

    Notes:
    - Uses the same preprocessing as Columbia_test for features: MFCC for audio (numcep=13),
      grayscale 112x112 faces for video at 25 fps.
    - Produces per-frame scores. Current frame score is the last element from the most recent
      inference window.
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

        # Sliding window storage
        self._window_video_frames = int(round(self.window_seconds * self.video_fps))
        self._window_audio_samples = int(round(self.window_seconds * self.sample_rate))

        self._video_buffer: deque[np.ndarray] = deque(maxlen=self._window_video_frames)
        # Store raw audio samples; compute MFCC just-in-time for each inference window
        self._audio_buffers: deque[np.ndarray] = deque()  # variable chunk sizes
        self._audio_len: int = 0

        # Last outputs
        self.last_score: Optional[float] = None
        self.last_decision: Optional[bool] = None

    # ------------------------- Public API -------------------------

    def push_video_frame(self, frame: np.ndarray) -> None:
        """Push a single face-cropped frame.

        Accepts:
        - 112x112 grayscale (uint8) frame, OR
        - Any BGR image (H,W,3); will be converted to grayscale and center-cropped to 112x112.
        """
        proc = self._preprocess_frame(frame)
        self._video_buffer.append(proc)

    def push_audio_samples(self, samples: np.ndarray, sr: Optional[int] = None) -> None:
        """Push raw PCM audio samples.

        samples: int16 or float32 mono waveform.
        sr: sample rate. If provided and != self.sample_rate, will be resampled.
        """
        if samples is None or len(samples) == 0:
            return
        # Ensure float32 for resampling, then back to int16-like scale
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sr is not None and sr != self.sample_rate:
            # Use resample_poly for higher quality
            # Compute rational approximation of sr -> self.sample_rate
            # We use gcd approach via fractions if needed; here, use direct poly with up/down
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
                # Trim head
                self._audio_buffers[0] = head[drop:]
                self._audio_len -= drop
                break

    def step(self) -> Tuple[Optional[float], Optional[bool]]:
        """Run inference immediately when called by the application.

        Returns (last_score, last_decision). Decision is (score >= threshold).
        """
        # Minimal context check
        if len(self._video_buffer) < 2 or self._audio_len < int(0.5 * self.sample_rate):
            return self.last_score, self.last_decision

        audio_feat, video_feat = self._build_features()
        if audio_feat is None or video_feat is None:
            return self.last_score, self.last_decision

        with torch.no_grad():
            inputA = torch.from_numpy(audio_feat).float().unsqueeze(0).to(self.device)
            inputV = torch.from_numpy(video_feat).float().unsqueeze(0).to(self.device)

            embedA = self._asd.model.forward_audio_frontend(inputA)
            embedV = self._asd.model.forward_visual_frontend(inputV)
            out = self._asd.model.forward_audio_visual_backend(embedA, embedV)
            scores = self._asd.lossAV.forward(out, labels=None)  # numpy vector

        if scores is None or len(scores) == 0:
            return self.last_score, self.last_decision

        self.last_score = float(scores[-1])
        self.last_decision = bool(self.last_score >= self.score_threshold)
        return self.last_score, self.last_decision

    def set_threshold(self, thr: float) -> None:
        self.score_threshold = float(thr)

    def reset(self) -> None:
        self._video_buffer.clear()
        self._audio_buffers.clear()
        self._audio_len = 0
        self.last_score = None
        self.last_decision = None

    # ------------------------- Internals -------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Unexpected frame shape: %s" % (frame.shape,))
        # Resize to at least 224 on shorter side, then center-crop to 112x112 like demo
        h, w = gray.shape[:2]
        if h != 224 or w != 224:
            gray = cv2.resize(gray, (224, 224))
        c = 112
        half = c // 2  # 56
        cropped = gray[c - half : c + half, c - half : c + half]  # 112x112
        return cropped.astype(np.uint8)

    def _concat_audio(self) -> np.ndarray:
        if len(self._audio_buffers) == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.concatenate(list(self._audio_buffers), axis=0)
        # Ensure within window
        if len(arr) > self._window_audio_samples:
            arr = arr[-self._window_audio_samples :]
        return arr

    def _build_features(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # Video features (T, 112, 112)
        vlist = list(self._video_buffer)
        video_feat = np.stack(vlist, axis=0) if len(vlist) > 0 else None

        # Audio MFCC (N, 13), 100 fps
        audio = self._concat_audio().astype(np.float32)
        if audio.size == 0:
            return None, None
        # Convert float32 in [-1,1] or int-like range to int16-like scale if needed (mfcc accepts float)
        # We keep as float32.
        mfcc = python_speech_features.mfcc(
            audio,
            self.sample_rate,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        if mfcc.shape[0] < 4 or video_feat is None:
            return None, None

        # Align lengths like in Columbia_test
        length_sec = min(
            (mfcc.shape[0] - mfcc.shape[0] % 4) / 100.0,  # ensure multiple of 4
            video_feat.shape[0] / self.video_fps,
        )
        if length_sec <= 0:
            return None, None

        a_len = int(round(length_sec * 100))
        v_len = int(round(length_sec * self.video_fps))

        mfcc = mfcc[:a_len, :]
        video_feat = video_feat[:v_len, :, :]
        return mfcc, video_feat
