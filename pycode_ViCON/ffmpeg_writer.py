"""Shared H.264 frame writer used by the ViCON recorders.
Drop-in for the cv2.VideoWriter calls in the recorders: same
write(bgr_frame) / release() / isOpened() surface.
"""
import shutil
import subprocess


class FfmpegWriter:
    """Encode raw BGR frames to H.264/mp4 via an ffmpeg subprocess pipe."""

    def __init__(self, path, width, height, fps, crf=23):
        self.path = path
        self.proc = None
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found on PATH — cannot record video.")
            return
        fps = fps if fps and fps > 1 else 30.0
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{int(width)}x{int(height)}", "-r", f"{fps:g}",
               "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", path]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except OSError as e:
            print(f"failed to start ffmpeg ({e}) — no video will be recorded.")
            self.proc = None

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def write(self, frame):
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            self.proc = None       # ffmpeg died (bad args / disk full); stop feeding it

    def release(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.proc.kill()
        finally:
            self.proc = None
