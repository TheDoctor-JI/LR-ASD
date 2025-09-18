import os
import time
import math
import logging
from collections import deque

import cv2
import numpy as np
import torch
import python_speech_features

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for model import

from ASD import ASD


class RealtimeCausalASD:
    """
    Realtime wrapper for causal ASD inference with explicit person instances.

    - Persons are created/removed explicitly (create_person/delete_person).
    - All persons share a single audio sample ring buffer.
    - Visual samples are pushed per-person (push_visual_sample), validated by id.
    - run_inference runs one forward pass per person using the shared audio buffer
      and that person's visual buffer, following Columbia_test feature prep:
        * Audio: MFCC (numcep=13, winlen=0.025, winstep=0.010) at 16kHz
        * Visual: gray -> resize(224,224) -> center crop(112x112)
        * Multi-duration averaging over durations {1,1,1,2,2,2,3,3,4,5,6}
        * Local temporal smoothing with 5-frame window (±2 frames)
    """

    def __init__(
        self,
        pretrain_model_path: str = "weight/pretrain_AVA.model",
        max_buffer_seconds: int = 10,
        fps: float = 25.0,
        audio_sr: int = 16000,
        device: str = "cuda",
        log_level=logging.INFO,
    ):
        """
        Args:
          pretrain_model_path: path to ASD pretrained weights.
          max_buffer_seconds: max buffered time for audio and visual (seconds).
          fps: visual frame rate (25.0).
          audio_sr: audio sample rate (16000).
          device: 'cuda' (default, as ASD assumes CUDA) or 'cpu' if adapted.
        """
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")
        self.logger = logging.getLogger("RealtimeCausalASD")

        # Model setup (mirrors ASD class usage in Columbia_test)
        self.asd = ASD()
        if pretrain_model_path:
            self.asd.loadParameters(pretrain_model_path)
            self.logger.info(f"Loaded ASD weights from {pretrain_model_path}")
        self.asd.eval()
        torch.set_grad_enabled(False)

        # Buffers/config
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.audio_sr = int(audio_sr)
        self.samples_per_tick = int(round(self.audio_sr / self.fps))  # 640 samples per frame at 16k/25fps
        self.max_buffer_seconds = int(max_buffer_seconds)
        self.max_audio_samples = self.max_buffer_seconds * self.audio_sr
        self.max_video_frames = int(round(self.max_buffer_seconds * self.fps))

        # Shared audio buffer (int16)
        self.audio_buffer = deque(maxlen=self.max_audio_samples)

        # Persons
        self.persons = {}  # id -> {'frames': deque of (112,112) uint8}

        self.logger.info(f"Initialized RealtimeCausalASD: fps={self.fps}, sr={self.audio_sr}, max={self.max_buffer_seconds}s")

    # ------------- Person lifecycle -------------

    def create_person(self, person_id: str) -> int:
        """Create a new person instance and return its id."""
        self.persons[person_id] = {
            "frames": deque(maxlen=self.max_video_frames),  # (112,112) uint8 frames at 25Hz
        }
        self.logger.info(f"Created person id={person_id}")

    def delete_person(self, person_id: str) -> None:
        """Delete an existing person instance."""
        if person_id not in self.persons:
            raise KeyError(f"Person id {person_id} does not exist")
        del self.persons[person_id]
        self.logger.info(f"Deleted person id={person_id}")

    # ------------- Streaming inputs -------------

    def push_audio_sample(self, chunk: np.ndarray, t_frame: int = None, t_sec_start: float = None, t_sec_end: float = None) -> None:
        """
        Append a 40ms mono audio chunk (shape (640,), int16) to the shared audio buffer.
        """
        if not isinstance(chunk, np.ndarray):
            raise TypeError("chunk must be a numpy array")
        if chunk.ndim != 1:
            raise ValueError("chunk must be 1-D")
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16)

        for s in chunk:
            self.audio_buffer.append(int(s))

    def push_visual_sample(self, person_id: str, face_img_bgr: np.ndarray, t_frame: int = None, t_sec: float = None) -> None:
        """
        Append a 25Hz visual face sample for a given person after preprocessing:
          BGR -> gray -> resize(224x224) -> center-crop(112x112)
        """
        if person_id not in self.persons:
            raise KeyError(f"Person id {person_id} does not exist")

        if not isinstance(face_img_bgr, np.ndarray):
            raise TypeError("face_img_bgr must be a numpy array")

        # Preprocess to match Columbia_test.py
        img = face_img_bgr
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            pass
        else:
            raise ValueError("face_img_bgr must be HxWx3 (BGR) or HxW (gray)")

        img224 = cv2.resize(img, (224, 224))
        # center-crop to 112x112
        c = 112
        h1, h2 = int(c - c / 2), int(c + c / 2)  # 56..168
        crop112 = img224[h1:h2, h1:h2].copy()  # (112,112) uint8

        self.persons[person_id]["frames"].append(crop112)

    # ------------- Inference -------------

    def run_inference(self):
        """
        Run ASD once per active person using:
          - Shared audio buffer (latest N seconds aligned to person's visual length).
          - Person's visual buffer (112x112 frames).

        Returns:
          dict: person_id -> {
            'scores': list[float],          # averaged across durations
            'scores_smooth': list[float],   # 5-frame local smoothing (±2)
            'frames_used': int,
            'seconds_used': float,
            'time_sec': float,              # inference wall time for this person
          }

        Logs per-person time and total time.
        """
        total_t0 = time.perf_counter()
        results = {}

        # Snapshot audio buffer as np.int16
        if len(self.audio_buffer) == 0:
            self.logger.warning("Audio buffer is empty; skipping inference for all persons.")
            return results
        audio_all = np.frombuffer(np.asarray(self.audio_buffer, dtype=np.int16).tobytes(), dtype=np.int16)

        # Iterate persons
        for pid, pdata in self.persons.items():
            t0 = time.perf_counter()

            # Visual features
            vid_frames = list(pdata["frames"])
            if len(vid_frames) == 0:
                self.logger.debug(f"Person {pid}: no visual frames; skipping.")
                continue
            videoFeature = np.stack(vid_frames, axis=0)  # (T,112,112), uint8

            # Duration to use from audio (latest matching video duration)
            seconds_needed = len(videoFeature) / self.fps
            samples_needed = int(round(seconds_needed * self.audio_sr))
            if samples_needed <= 0:
                self.logger.debug(f"Person {pid}: insufficient audio duration; skipping.")
                continue

            # Align to the end (causal)
            if len(audio_all) < samples_needed:
                # Not enough audio yet; pad front with zeros
                pad = np.zeros((samples_needed - len(audio_all),), dtype=np.int16)
                audio_slice = np.concatenate([pad, audio_all], axis=0)
            else:
                audio_slice = audio_all[-samples_needed:]

            # Audio MFCC @ 16k (matches Columbia_test)
            audioFeature = python_speech_features.mfcc(audio_slice, samplerate=self.audio_sr, numcep=13, winlen=0.025, winstep=0.010)

            # Make audio/video lengths consistent per Columbia_test
            # length = min((audio_frames - audio_frames%4)/100, video_frames)
            audio_frames = audioFeature.shape[0]
            if audio_frames == 0:
                self.logger.debug(f"Person {pid}: no MFCC frames; skipping.")
                continue

            length_sec = min((audio_frames - audio_frames % 4) / 100.0, float(videoFeature.shape[0]))
            if length_sec <= 0:
                self.logger.debug(f"Person {pid}: computed length <= 0; skipping.")
                continue

            # Trim to integer counts
            audio_len = int(round(length_sec * 100))
            video_len = int(round(length_sec * 25))
            audioFeature = audioFeature[:audio_len, :]
            videoFeature = videoFeature[:video_len, :, :]

            # Multi-duration averaging (as in Columbia_test)
            # This is similar to what people do in object detection with multiple anchors
            durationSet = [1, 2, 3, 4, 5, 6]
            allScore = []
            with torch.no_grad():
                for duration in durationSet:
                    ## Chunk up the input into segments of 'duration' seconds, which are then sent to the model for forward pass
                    batchSize = int(math.ceil(length_sec / duration))
                    scores = []
                    for i in range(batchSize):
                        a0 = int(i * duration * 100)
                        a1 = int((i + 1) * duration * 100)
                        v0 = int(i * duration * 25)
                        v1 = int((i + 1) * duration * 25)
                        a_chunk = audioFeature[a0:a1, :]
                        v_chunk = videoFeature[v0:v1, :, :]

                        if a_chunk.shape[0] == 0 or v_chunk.shape[0] == 0:
                            continue

                        # To tensors with batch dim
                        inputA = torch.FloatTensor(a_chunk).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(v_chunk).unsqueeze(0).cuda()

                        # Frontends and backend, same as Columbia_test
                        embedA = self.asd.model.forward_audio_frontend(inputA)
                        embedV = self.asd.model.forward_visual_frontend(inputV)
                        out = self.asd.model.forward_audio_visual_backend(embedA, embedV)

                        # lossAV.forward(out, labels=None) may return tensor, ndarray, list, or scalar
                        val = self.asd.lossAV.forward(out, labels=None)

                        # Normalize to 1D list of floats
                        if isinstance(val, torch.Tensor):
                            vals = val.detach().cpu().numpy().ravel().tolist()
                        elif isinstance(val, np.ndarray):
                            vals = val.ravel().tolist()
                        elif isinstance(val, (list, tuple)):
                            vals = list(map(float, val))
                        else:
                            vals = [float(val)]
                        scores.extend(vals)

                    if len(scores) > 0:
                        allScore.append(np.array(scores, dtype=np.float32))

            if len(allScore) == 0:
                self.logger.debug(f"Person {pid}: got no scores; skipping.")
                continue

            # Average across durations and round to 1 decimal, for each frame
            avg_scores = np.mean(np.stack(allScore, axis=0), axis=0)
            avg_scores = np.round(avg_scores, 1).astype(float)

            # 5-frame smoothing (±2) like visualization() in Columbia_test
            smoothed = []
            for fidx in range(len(avg_scores)):
                l = max(fidx - 2, 0)
                r = min(fidx + 3, len(avg_scores))
                smoothed.append(float(np.mean(avg_scores[l:r])))

            t1 = time.perf_counter()
            results[pid] = {
                "scores": avg_scores.tolist(),
                "scores_smooth": smoothed,
                "frames_used": len(avg_scores),
                "seconds_used": len(avg_scores) / self.fps,
                "time_sec": t1 - t0,
            }
            self.logger.info(f"Inference person={pid} frames={len(avg_scores)} time={t1 - t0:.3f}s")

        total_t1 = time.perf_counter()
        self.logger.info(f"Total inference time: {total_t1 - total_t0:.3f}s for {len(results)} persons")
        return results