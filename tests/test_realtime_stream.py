import os
import time
import glob
import numpy as np
import cv2
from scipy.io import wavfile
import os, sys
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


def stream_demo_sequence(video_folder: str, model_path: str = "weight/pretrain_AVA.model"):
    """
    Emulate realtime streaming using demo preprocessed clips in demo/0001.

    This test script performs the preprocessing steps (face tracking/cropping) OUTSIDE
    of the ASD class by reusing the already-cropped face clip produced by Columbia_test.

    Expected input files (produced by running the demo):
      demo/0001/pycrop/00000.avi
      demo/0001/pycrop/00000.wav
    Or multiple face tracks like 00000.avi, 00001.avi, etc.
    """
    pycrop = os.path.join(video_folder, "pycrop")
    avi_files = sorted(glob.glob(os.path.join(pycrop, "*.avi")))
    assert len(avi_files) > 0, f"No face-crop avi clips found in {pycrop}. Run the demo first."

    # Initialize realtime ASD
    asd = ASDRealtime(
        model_path=model_path,
        window_seconds=5.0,
        video_fps=25.0,
        score_threshold=-1.5,
    )

    cv2.namedWindow("LR-ASD Realtime", cv2.WINDOW_NORMAL)

    for avi_path in avi_files:
        wav_path = avi_path.replace(".avi", ".wav")
        if not os.path.exists(wav_path):
            print(f"Warning: missing audio for {avi_path}, skipping")
            continue
        print(f"Streaming {os.path.basename(avi_path)} ...")

        # Load audio fully; we will feed in small chunks to emulate streaming
        sr, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Normalize to float32 [-1,1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio /= 32768.0

        # Start audio playback for situational awareness
        play_obj = None
        try:
            if sd is not None:
                sd.stop()
                sd.play(audio, sr)
            elif sa is not None:
                audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                play_obj = sa.play_buffer(audio_i16, 1, 2, sr)
            else:
                print("Audio playback disabled (install 'sounddevice' or 'simpleaudio').")
        except Exception as e:
            print(f"Audio playback initialization failed: {e}")

        # Open video for frame-by-frame feed
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            print(f"Failed to open {avi_path}")
            # Stop audio if started
            if sd is not None:
                sd.stop()
            elif play_obj is not None:
                play_obj.stop()
            continue

        # Emulate realtime: feed per frame and a matching audio chunk to ASD
        frame_period = 1.0 / 25.0
        audio_chunk = int(round(frame_period * sr))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Push one frame
            asd.push_video_frame(frame)

            # Push matching audio slice into ASD
            start = frame_idx * audio_chunk
            end = min(len(audio), (frame_idx + 1) * audio_chunk)
            if start < end:
                asd.push_audio_samples(audio[start:end], sr=sr)

            # External app drives inference each frame
            score, decision = asd.step()

            # Visualization overlay
            vis = frame.copy()
            txt = f"score={score:.2f} active={int(decision)}" if score is not None else "warming up..."
            color = (0, 255, 0) if (score is not None and decision) else (0, 0, 255)
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            # Draw border to indicate decision
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (2, 2), (w - 3, h - 3), color, 3)

            cv2.imshow("LR-ASD Realtime", vis)
            # Keep visualization responsive; audio playback paces overall flow
            key = cv2.waitKey(max(1, int(frame_period * 1000 / 2)))
            if key & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()
        # Stop audio playback for this clip
        try:
            if sd is not None:
                sd.stop()
            elif play_obj is not None:
                play_obj.stop()
        except Exception:
            pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Default to demo path created by the repository's demo
    stream_demo_sequence(os.path.join("demo", "0001"))
