import os
import time
import glob
import numpy as np
import cv2
from scipy.io import wavfile
import sys
import threading
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


def stream_demo_sequence(video_folder: str, model_path: str = "weight/pretrain_AVA.model", trigger_interval_s: float = 0.2):
    """
    Emulate realtime streaming using demo preprocessed clips in demo/0001.

    The ASD class is externally driven: stream data at realtime cadence in the main thread and
    trigger infer() on a separate processing thread every trigger_interval_s.
    Visualization shows the average score during last trigger window.
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

    track_id = "t0"
    asd.add_track(track_id)

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

        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            print(f"Failed to open {avi_path}")
            if sd is not None:
                sd.stop()
            elif play_obj is not None:
                play_obj.stop()
            continue

        # Realtime streaming settings
        video_fps = 25.0
        frame_period = 1.0 / video_fps
        audio_chunk = int(round(frame_period * sr))

        # Shared result between threads
        result = {"score": None, "decision": False}
        result_lock = threading.Lock()
        stop_evt = threading.Event()

        def processor_loop():
            last_trigger_time = time.time()
            while not stop_evt.is_set():
                now = time.time()
                if now - last_trigger_time >= trigger_interval_s:
                    res = asd.infer(min_required_seconds=0.5)
                    if track_id in res:
                        series = res[track_id]['series']
                        N = max(1, int(round(trigger_interval_s * video_fps)))
                        avg_score = float(np.mean(series[-N:]))
                        with result_lock:
                            result["score"] = avg_score
                            result["decision"] = bool(avg_score >= asd.score_threshold)
                    last_trigger_time = now
                time.sleep(0.005)  # avoid busy-wait

        proc_thread = threading.Thread(target=processor_loop, daemon=True)
        proc_thread.start()

        frame_idx = 0
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Push one frame at realtime cadence
            asd.push_video_frame(track_id, frame)

            # Push matching audio slice into ASD
            a_start = frame_idx * audio_chunk
            a_end = min(len(audio), (frame_idx + 1) * audio_chunk)
            if a_start < a_end:
                asd.push_audio_samples(audio[a_start:a_end], sr=sr)

            # Visualization overlay (latest result from processor thread)
            with result_lock:
                last_score = result["score"]
                decision = result["decision"]

            vis = frame.copy()
            txt = (
                f"avg@{trigger_interval_s:.1f}s={last_score:.2f} active={int(decision)}"
                if last_score is not None else "warming up..."
            )
            color = (0, 255, 0) if (last_score is not None and decision) else (0, 0, 255)
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (2, 2), (w - 3, h - 3), color, 3)

            # Display and sleep to maintain realtime cadence
            cv2.imshow("LR-ASD Realtime", vis)
            elapsed = time.time() - loop_start
            delay_ms = max(1, int((frame_period - elapsed) * 1000)) if elapsed < frame_period else 1
            key = cv2.waitKey(delay_ms)
            if key & 0xFF == ord('q'):
                break

            frame_idx += 1

        # Cleanup this clip
        stop_evt.set()
        proc_thread.join(timeout=1.0)
        cap.release()
        try:
            if sd is not None:
                sd.stop()
            elif play_obj is not None:
                play_obj.stop()
        except Exception:
            pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_demo_sequence(os.path.join("demo", "0001"))
