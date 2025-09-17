import os
import time
import glob
import numpy as np
import cv2
from scipy.io import wavfile
import sys
import threading
import shutil
import subprocess
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

def stream_demo_sequence(video_folder: str, model_path: str = "weight/pretrain_AVA.model", trigger_interval_s: float = 0.05):
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
        window_seconds=10.0,
        video_fps=25.0,
        score_threshold=0,
    )

    track_id = "t0"
    asd.add_track(track_id)

    if VISUALIZE_LOCALLY:
        cv2.namedWindow("LR-ASD Realtime", cv2.WINDOW_NORMAL)

    # Prepare output directory for visualization videos
    out_dir = os.path.join(video_folder, "pywork", "realtime_vis")
    os.makedirs(out_dir, exist_ok=True)

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
        if VISUALIZE_LOCALLY:
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
        stop_evt = threading.Event()

        def processor_loop():
            last_trigger_time = time.time()
            while not stop_evt.is_set():
                now = time.time()
                if now - last_trigger_time >= trigger_interval_s:
                    res = asd.infer(min_required_seconds=0.5)
                    t_after_infer = time.time()
                    print(f"  [t={now:.1f}] infer() took {t_after_infer - now:.3f}s")
                    if track_id in res:
                        series = res[track_id]['series']
                        N = max(1, int(round(trigger_interval_s * video_fps)))
                        avg_score = float(np.mean(series[-N:]))
                        result["score"] = avg_score
                        result["decision"] = bool(avg_score >= asd.score_threshold)
                        print(f"  [t={now:.1f}] avg_score={avg_score:.3f} active={result['decision']}")
                    else:
                        print(f"  [t={now:.1f}] res: {res} No result for track {track_id}")

                    last_trigger_time = now
                time.sleep(0.005)  # avoid busy-wait

        proc_thread = threading.Thread(target=processor_loop)
        proc_thread.start()

        frame_idx = 0
        # Video writer will be lazily initialized once frame size is known
        writer = None
        out_path = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(avi_path))[0]}_realtime_vis.avi",
        )
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # AVI codec

        # Collect audio chunks we push to ASD so we can mux them into the saved video
        pushed_audio_chunks = []

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
                audio_chunk_data = audio[a_start:a_end]
                asd.push_audio_samples(audio_chunk_data, sr=sr)
                pushed_audio_chunks.append(audio_chunk_data)

            # Visualization overlay (latest result from processor thread, no thread safety for now)
            print(result)
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

            # Initialize writer when we know the size
            if writer is None:
                h_vis, w_vis = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, video_fps, (w_vis, h_vis))
                if not writer.isOpened():
                    print(f"Warning: cannot open writer for {out_path}")
                    writer = None

            # Write visualization frame
            if writer is not None:
                writer.write(vis)

            # Display and sleep to maintain realtime cadence
            if VISUALIZE_LOCALLY:
                cv2.imshow("LR-ASD Realtime", vis)
                elapsed = time.time() - loop_start
                delay_ms = max(1, int((frame_period - elapsed) * 1000)) if elapsed < frame_period else 1
                key = cv2.waitKey(delay_ms)
                if key & 0xFF == ord('q'):
                    break
            else:##If not visualizing, simply sleep to maintain cadence
                elapsed = time.time() - loop_start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)

            frame_idx += 1

        # Cleanup this clip
        stop_evt.set()
        proc_thread.join(timeout=1.0)
        cap.release()
        # Finalize writer for this clip
        if writer is not None:
            try:
                writer.release()
                print(f"Saved visualization: {out_path}")
            except Exception:
                pass
        try:
            if sd is not None:
                sd.stop()
            elif play_obj is not None:
                play_obj.stop()
        except Exception:
            pass

        # After saving the silent visualization video, mux in the exact audio we streamed to ASD
        try:
            if pushed_audio_chunks and os.path.exists(out_path):
                audio_stream = np.concatenate(pushed_audio_chunks, axis=0)
                # Ensure audio in int16 for WAV (ffmpeg path) while keeping MoviePy path in float32
                out_base, out_ext = os.path.splitext(out_path)
                out_mux_tmp = f"{out_base}__mux_tmp{out_ext}"

                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path is not None:
                    # Write a temporary WAV next to the video
                    tmp_wav = f"{out_base}_tmp_audio.wav"
                    try:
                        wavfile.write(tmp_wav, sr, (np.clip(audio_stream, -1.0, 1.0) * 32767.0).astype(np.int16))
                        cmd = [
                            ffmpeg_path, "-y",
                            "-i", out_path,
                            "-i", tmp_wav,
                            "-c:v", "copy",
                            # Use PCM for AVI container compatibility
                            "-c:a", "pcm_s16le",
                            "-shortest",
                            out_mux_tmp,
                        ]
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        if proc.returncode == 0 and os.path.exists(out_mux_tmp):
                            # Replace original file atomically
                            os.replace(out_mux_tmp, out_path)
                            print(f"Muxed audio into video: {out_path}")
                        else:
                            print("ffmpeg muxing failed, attempting MoviePy fallback...")
                            raise RuntimeError("ffmpeg mux failed")
                    finally:
                        try:
                            if os.path.exists(tmp_wav):
                                os.remove(tmp_wav)
                        except Exception:
                            pass
                else:
                    # Fallback: try MoviePy (requires moviepy and imageio-ffmpeg)
                    try:
                        from moviepy.editor import VideoFileClip
                        from moviepy.audio.AudioClip import AudioArrayClip

                        clip = VideoFileClip(out_path)
                        aud = AudioArrayClip(audio_stream.reshape(-1, 1).astype(np.float32), fps=sr)
                        # Ensure audio matches video duration
                        aud = aud.set_duration(clip.duration)
                        clip = clip.set_audio(aud)
                        # Write to temporary file matching container, then replace original
                        clip.write_videofile(out_mux_tmp, codec="mpeg4", audio_codec="pcm_s16le", fps=video_fps, verbose=False, logger=None)
                        clip.close()
                        if os.path.exists(out_mux_tmp):
                            os.replace(out_mux_tmp, out_path)
                            print(f"Muxed audio into video: {out_path}")
                    except Exception as e:
                        print(f"MoviePy muxing failed: {e}. Audio will not be embedded.")
        except Exception as e:
            print(f"Audio muxing encountered an error: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_demo_sequence(os.path.join("demo", "0002"))
