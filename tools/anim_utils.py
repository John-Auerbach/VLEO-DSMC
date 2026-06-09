"""Helpers for parallel per-frame precomputation in animation scripts.

The animation scripts spend almost all their time loading and reducing each
dump frame (a single grid frame can be ~500 MB on disk and tens of GB in a
DataFrame). Rendering the frame is cheap by comparison. These helpers let a
script precompute every frame's payload up front, optionally across several
worker processes, while printing live progress. The animation is then assembled
from the cached payloads, so each file is read exactly once.
"""

import multiprocessing
import shutil
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


_CODEC_CACHE = None


def pick_video_codec(candidates=("libx264", "h264", "mpeg4")):
    """Return the first ffmpeg video codec that actually encodes on this machine.

    Some ffmpeg builds (e.g. Roar's module) advertise an ``h264`` encoder that is
    really the hardware ``h264_v4l2m2m`` wrapper and fails at runtime on a compute
    node with no video device, while only the software ``mpeg4`` encoder works.
    Probing with a tiny synthetic clip up front lets us pick a working codec once,
    instead of discovering the failure after re-rendering every animation frame.
    Result is cached for the process.
    """
    global _CODEC_CACHE
    if _CODEC_CACHE is not None:
        return _CODEC_CACHE

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Let matplotlib fall back to its own writer discovery.
        _CODEC_CACHE = candidates[-1]
        return _CODEC_CACHE

    for codec in candidates:
        out = os.path.join(tempfile.gettempdir(), f"_codectest_{os.getpid()}.mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", "color=c=black:s=64x64:d=1", "-vcodec", codec,
               "-pix_fmt", "yuv420p", "-y", out]
        try:
            rc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode
        except Exception:
            rc = 1
        finally:
            try:
                os.remove(out)
            except OSError:
                pass
        if rc == 0:
            _CODEC_CACHE = codec
            return codec

    _CODEC_CACHE = candidates[-1]
    return _CODEC_CACHE


def save_animation(ani, outpath, fps, dpi, bitrate=None):
    """Save a matplotlib animation to mp4 using a codec known to work here.

    Picks the codec via :func:`pick_video_codec` (cached), so this works on a
    laptop with full ffmpeg (libx264) and on Roar's module (mpeg4 only) without
    per-script changes. Also points matplotlib at the absolute ffmpeg binary so
    it does not rely on the writer's bare-name PATH lookup.
    """
    import matplotlib
    from matplotlib.animation import FFMpegWriter

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg

    codec = pick_video_codec()
    writer = FFMpegWriter(fps=fps, codec=codec, bitrate=bitrate)
    ani.save(outpath, writer=writer, dpi=dpi)
    print(f"Saved {outpath} (codec={codec})")


def compute_payloads_parallel(indices, fn, jobs=1, label="frame"):
    """Compute per-frame payloads, optionally across worker processes.

    Parameters
    ----------
    indices : iterable of int
        Frame indices to process, e.g. ``range(len(timesteps))``.
    fn : callable
        A *top-level* (picklable) function ``fn(i) -> (step, payload)``. It may
        read module globals; with the ``fork`` start method (Linux default)
        workers inherit them, so no extra arguments are needed.
    jobs : int
        Number of worker processes. ``jobs <= 1`` runs serially, which keeps the
        default behaviour identical on a laptop. Raise it only on a high-memory
        node: peak memory is roughly ``jobs`` frames held at once.
    label : str
        Word used in the progress line.

    Returns
    -------
    dict
        ``{i: (step, payload)}`` for every index. Progress is printed as each
        frame finishes, including the timestep it corresponds to.
    """
    indices = list(indices)
    total = len(indices)
    results = {}

    def _serial():
        for n, i in enumerate(indices, 1):
            step, payload = fn(i)
            results[i] = (step, payload)
            print(f"  {label} {n}/{total} (step {step}) done", flush=True)
        return results

    if jobs <= 1 or total <= 1:
        return _serial()

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        # 'fork' unavailable (non-Linux spawn-only platforms): run serially so
        # functions relying on inherited module globals still work.
        return _serial()

    done = 0
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futures = {ex.submit(fn, i): i for i in indices}
        for fut in as_completed(futures):
            i = futures[fut]
            step, payload = fut.result()
            results[i] = (step, payload)
            done += 1
            print(f"  {label} {done}/{total} (step {step}) done", flush=True)
    return results
