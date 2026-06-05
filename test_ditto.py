#!/usr/bin/env python3
"""Minimal standalone Ditto test harness.

Zero Aidols stack required. Calls Ditto's StreamSDK directly.
Use this to validate emo / drive_eye / blink / smo_k_d changes before deployment.

USAGE — compare blink vs no-blink, or smo_k_d tuning:

    python test_ditto_isolated.py \
        --photo /tmp/test/ref.jpg \
        --audio /tmp/test/test_16k.wav \
        --out   /tmp/test/out_baseline.mp4

    python test_ditto_isolated.py \
        --photo /tmp/test/ref.jpg \
        --audio /tmp/test/test_16k.wav \
        --out   /tmp/test/out_blink_smo.mp4 \
        --drive-eye on --blink --smo-k-d 1

    ffmpeg -i out_baseline.mp4.tmp.mp4 -i out_blink_smo.mp4.tmp.mp4 \
        -filter_complex "[0:v][1:v]hstack=inputs=2" \
        -c:v libx264 -crf 18 comparison.mp4

DITTO EMO CLASSES (condition_handler.py):
    0=Angry  1=Disgust  2=Fear  3=Happy  4=Neutral  5=Sad  6=Surprise  7=Contempt
"""

import argparse
import math
import sys
import time

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Chunking — mirrors ditto_engine.chunk_audio_for_ditto (no Aidols import)
# ---------------------------------------------------------------------------

DEFAULT_CHUNKSIZE = (3, 5, 2)
DITTO_AUDIO_SR = 16000
DITTO_FPS = 25


def _resample_if_needed(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == DITTO_AUDIO_SR:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(DITTO_AUDIO_SR, src_sr)
    up, down = DITTO_AUDIO_SR // g, src_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def _chunk_audio(audio_16k: np.ndarray, chunksize=DEFAULT_CHUNKSIZE):
    audio_padded = np.concatenate(
        [np.zeros((chunksize[0] * 640,), dtype=np.float32), audio_16k], 0
    )
    split_len = int(sum(chunksize) * 0.04 * DITTO_AUDIO_SR) + 80
    step = chunksize[1] * 640
    for i in range(0, len(audio_padded), step):
        chunk = audio_padded[i : i + split_len]
        if len(chunk) < split_len:
            chunk = np.pad(chunk, (0, split_len - len(chunk)), mode="constant")
        yield chunk


# ---------------------------------------------------------------------------
# Blink generator
# ---------------------------------------------------------------------------

def _make_blink_arr(n_frames: int, fps: int = DITTO_FPS, seed: int = 42) -> np.ndarray:
    """Procedural blink pattern: ~12-20 blinks/min, randomised interval.

    Returns a float32 array of length n_frames where 0.0 = natural open,
    -1.0 = fully closed. Ditto interprets this as delta_eye_open offsets.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros(n_frames, dtype=np.float32)
    i = fps * 2  # first blink after 2 s to avoid hitting the fade-in
    while i < n_frames - 8:
        # Close phase: 3 frames ramp to -1.0
        for j in range(3):
            if i + j < n_frames:
                arr[i + j] = -(j + 1) / 3.0
        # Hold closed: 1 frame
        if i + 3 < n_frames:
            arr[i + 3] = -1.0
        # Open phase: 4 frames ramp back to 0.0
        for j in range(4):
            if i + 4 + j < n_frames:
                arr[i + 4 + j] = -(1.0 - (j + 1) / 4.0)
        # Next blink: random 2.5 – 5.0 s interval
        interval = int(rng.uniform(2.5, 5.0) * fps)
        i += interval
    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Isolated Ditto inference test — no Aidols stack"
    )
    parser.add_argument("--photo",   required=True,  help="Reference photo path (jpg/png)")
    parser.add_argument("--audio",   required=True,  help="Speech WAV (any SR — auto-resampled to 16 kHz)")
    parser.add_argument("--out",     required=True,  help="Output MP4 path (Ditto appends .tmp.mp4)")
    parser.add_argument("--emo",     type=int, default=4,
                        help="Ditto emotion class 0-7 (default 4=Neutral)")
    parser.add_argument("--fade-in", type=int, default=8,
                        help="Frames to fade in (default 8 → 0.32 s @ 25 fps)")
    parser.add_argument("--fade-out",type=int, default=8,
                        help="Frames to fade out (default 8)")
    parser.add_argument("--eye-f0-mode", action="store_true",
                        help="Lock eye tracking to first F0 frame")
    parser.add_argument("--drive-eye",
                        choices=["auto", "on", "off"], default="on",
                        help="drive_eye: on=True (default, best result), off=False, auto=Ditto default")
    parser.add_argument("--blink", action="store_true",
                        help="Enable procedural blink pattern via delta_eye_arr")
    parser.add_argument("--smo-k-d", type=int, default=None,
                        help="Lip sync smoothing kernel (1=crispest). Omit to use Ditto default.")
    parser.add_argument("--ditto-root",
                        default="/runpod-volume/aidols/models/ditto",
                        help="Path to cloned ditto-talkinghead repo")
    parser.add_argument("--ditto-cfg", default=None,
                        help="Override ditto_cfg pkl path")
    parser.add_argument("--ditto-data", default=None,
                        help="Override ditto_data checkpoint dir")
    args = parser.parse_args()

    ditto_root  = args.ditto_root
    ditto_cfg   = args.ditto_cfg  or f"{ditto_root}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
    ditto_data  = args.ditto_data or f"{ditto_root}/checkpoints/ditto_pytorch"
    drive_eye   = {"auto": None, "on": True, "off": False}[args.drive_eye]

    print("=" * 60)
    print(f"  photo     : {args.photo}")
    print(f"  audio     : {args.audio}")
    print(f"  output    : {args.out}  (actual file: {args.out}.tmp.mp4)")
    print(f"  emo       : {args.emo}  (0=Angry 1=Disgust 2=Fear 3=Happy 4=Neutral 5=Sad 6=Surprise 7=Contempt)")
    print(f"  fade_in   : {args.fade_in} frames")
    print(f"  fade_out  : {args.fade_out} frames")
    print(f"  eye_f0    : {args.eye_f0_mode}")
    print(f"  drive_eye : {drive_eye!r}")
    print(f"  blink     : {args.blink}")
    print(f"  smo_k_d   : {args.smo_k_d!r}  (None = Ditto default)")
    print(f"  ditto_root: {ditto_root}")
    print("=" * 60)

    # 1. Load + normalise audio
    audio_raw, src_sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio_raw.ndim > 1:
        audio_raw = audio_raw.mean(axis=1)
    audio_16k = _resample_if_needed(audio_raw, src_sr)
    audio_seconds = len(audio_16k) / DITTO_AUDIO_SR
    print(f"[audio]  {audio_seconds:.2f}s  |  src_sr={src_sr}  samples_16k={len(audio_16k)}")

    # 2. Load Ditto SDK
    if ditto_root not in sys.path:
        sys.path.insert(0, ditto_root)

    try:
        import torch
        from stream_pipeline_online import StreamSDK
    except ImportError as exc:
        print(f"\nERROR: Cannot import Ditto — {exc}")
        print("Make sure --ditto-root points to a cloned ditto-talkinghead repo")
        sys.exit(1)

    cuda_ok = torch.cuda.is_available()
    print(f"[torch]  CUDA={cuda_ok}  version={torch.__version__}")
    if not cuda_ok:
        print("WARNING: No CUDA detected — inference will be extremely slow on CPU")

    t_load = time.time()
    sdk = StreamSDK(ditto_cfg, ditto_data)
    print(f"[ditto]  Loaded in {time.time()-t_load:.1f}s")

    # 3. Setup session
    n_frames = math.ceil(audio_seconds * DITTO_FPS)
    setup_kwargs: dict = {"eye_f0_mode": args.eye_f0_mode}
    if drive_eye is not None:
        setup_kwargs["drive_eye"] = drive_eye
    if args.emo != 4:
        setup_kwargs["emo"] = args.emo
    if args.smo_k_d is not None:
        setup_kwargs["smo_k_d"] = args.smo_k_d

    sdk.setup(args.photo, args.out, **setup_kwargs)

    nd_kwargs: dict = {"fade_in": args.fade_in, "fade_out": args.fade_out}
    if args.blink:
        blink_arr = _make_blink_arr(n_frames)
        nd_kwargs["delta_eye_arr"] = blink_arr
        print(f"[blink]  {int((blink_arr < 0).sum())} blink frames over {n_frames} total")

    sdk.setup_Nd(N_d=n_frames, **nd_kwargs)
    print(f"[setup]  n_frames={n_frames}  setup_kwargs={setup_kwargs}  nd_kwargs_keys={list(nd_kwargs)}")

    # 4. Run chunks
    chunks = list(_chunk_audio(audio_16k))
    print(f"[run]    {len(chunks)} chunks ...")
    t_run = time.time()
    for i, chunk in enumerate(chunks):
        sdk.run_chunk(chunk, DEFAULT_CHUNKSIZE)
        if i % 10 == 0:
            elapsed = time.time() - t_run
            pct = 100 * (i + 1) / len(chunks)
            print(f"         {pct:5.1f}%  {elapsed:.1f}s", end="\r", flush=True)

    sdk.close()
    ditto_elapsed = time.time() - t_run
    rtf = ditto_elapsed / audio_seconds

    print(f"\n[done]   {ditto_elapsed:.1f}s  |  RTF={rtf:.2f}  (target <1.0)")
    print(f"[out]    {args.out}.tmp.mp4")


if __name__ == "__main__":
    main()
