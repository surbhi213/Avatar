import os
import sys
import uuid
import asyncio
import threading
import queue
import io
import struct
import time
from pathlib import Path
import shutil
import wave
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

# Fixed input sizes -> let cuDNN pick the fastest conv kernels (VAE decode).
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from PIL import Image


# =========================================================
# PATHS
# =========================================================

# Portable project layout for local development and RunPod.
# The old DigitalOcean absolute path is intentionally not used.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVEAVATAR_ROOT = Path(
    os.environ.get("LIVEAVATAR_ROOT", str(_PROJECT_ROOT / "LiveAvatar"))
).expanduser().resolve()

if not LIVEAVATAR_ROOT.exists():
    raise RuntimeError(
        f"LIVEAVATAR_ROOT does not exist: {LIVEAVATAR_ROOT}. "
        "Set LIVEAVATAR_ROOT to the deployed LiveAvatar directory."
    )

sys.path.insert(0, str(LIVEAVATAR_ROOT))


# =========================================================
# LIVEAVATAR IMPORTS
# =========================================================

from liveavatar.models.wan.causal_s2v_pipeline import WanS2V

from liveavatar.models.wan.wan_2_2.configs import (
    WAN_CONFIGS,
    MAX_AREA_CONFIGS,
)

from liveavatar.utils.args_config import (
    parse_args_for_training_config,
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="LiveAvatar Realtime Service"
)

MEDIA_AUDIO = 1
MEDIA_FRAME = 2
MEDIA_START = 3
MEDIA_END = 4
# Raw 16 kHz mono int16 audio appended while a reply is being generated.
MEDIA_AUDIO_PCM = 5

def make_media_packet(message_type: int, payload: bytes = b""):
    timestamp_us = int(time.perf_counter() * 1_000_000)
    return struct.pack("!BQ", message_type, timestamp_us) + payload



# =========================================================
# CONFIG
# =========================================================

BASE_DIR = LIVEAVATAR_ROOT

CKPT_DIR = Path(
    os.environ.get(
        "LIVEAVATAR_CKPT_DIR",
        "/runpod-volume/models/Wan2.2-S2V-14B",
    )
).expanduser().resolve()

TRAINING_CONFIG = Path(
    os.environ.get(
        "LIVEAVATAR_TRAINING_CONFIG",
        str(BASE_DIR / "liveavatar/configs/s2v_causal_sft.yaml"),
    )
).expanduser().resolve()

if not CKPT_DIR.exists():
    raise RuntimeError(f"LiveAvatar checkpoint directory not found: {CKPT_DIR}")
if not TRAINING_CONFIG.exists():
    raise RuntimeError(f"LiveAvatar training config not found: {TRAINING_CONFIG}")


# =========================================================
# HORSES
# =========================================================

HORSES = {
    "5": {
        "image": BASE_DIR / "poc_inputs/minett.jpg",
    },

    "2": {
        "image": BASE_DIR / "poc_inputs/good_deal.jpeg",
    },
    "3": {
        "image": BASE_DIR / "poc_inputs/ABEL.JPG",
         },
    "6": {
        "image": BASE_DIR / "poc_inputs/THIRD PILLAR.jpeg",
        },
    "7": {
        "image": BASE_DIR / "poc_inputs/sweet flowers.jpeg",
                },
    "8": {
        "image": BASE_DIR / "poc_inputs/MY SPECIAL EVENT.jpeg",
         },
}


# =========================================================
# GLOBAL MODEL
# =========================================================

    # pipeline = None
    # training_settings = None

    # generation_lock = threading.Lock()
pipeline = None
training_settings = None

# Realtime mode:
# Keep the generation models on the H100 instead of repeatedly
# moving them between CPU and GPU.
REALTIME_MODE = True

# Prevent two WanS2V generations from using the GPU at the same time.
generation_lock = threading.Lock()

# Indicates that the model has been fully prepared for realtime use.
pipeline_ready = False

# Output resolution budget (pixels per frame); sizes snap to multiples of 64.
# DiT + VAE cost scales with it, so this is the biggest single speed knob.
#   288000 -> 448x640 (original quality)
#   256000 -> 384x576
#   184320 -> 320x512 (default: ~57% of the pixels, roughly 2x faster DiT)
#   147456 -> 320x448
# Resolution budget in pixels per frame (184320 ~ 512x320 / 448x384; above real time with FlashAttention 3).
AVATAR_MAX_AREA = int(os.environ.get("AVATAR_MAX_AREA", 184320))

# Denoising steps per block. The pipeline caches the schedule from its first
# call (the warm-up), so this is fixed for the life of the process and every
# request uses it regardless of what the caller asks for.
SAMPLE_STEPS = int(os.environ.get("AVATAR_SAMPLE_STEPS", 1))

# Skip the motion re-encode after the first 48-frame clip (a ~2.5 s stall).
# Later clips then keep the photo-based motion context.
FAST_CLIP_TRANSITION = os.environ.get("AVATAR_FAST_CLIP_TRANSITION", "1") == "1"
# LiveAvatar output frame rate; used only for the realtime-ratio log.
AVATAR_FPS = 25

JPEG_QUALITY = int(os.environ.get("AVATAR_JPEG_QUALITY", "80"))

# Shared JPEG encoders. PIL releases the GIL while encoding, so frames of a
# chunk are encoded in parallel instead of one by one.
JPEG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")

SPEECH_PROMPT = "A realistic horse speaking naturally to the camera"

# Idle: closed mouth, only blinking and small body movement.
IDLE_PROMPT = (
    "A realistic horse standing calmly with its mouth closed, "
    "blinking its eyes gently and shifting its body slightly, not talking"
)

IDLE_CLIP_SECONDS = 4
IDLE_INFER_FRAMES = 48
IDLE_SAMPLE_STEPS = SAMPLE_STEPS

# Audio quieter than this RMS (int16 scale) is treated as an idle request.
SILENCE_RMS_THRESHOLD = 150.0

# Optional runtime overrides (avatar_tune.json next to this file):
#   {"max_area": 122880, "sample_steps": 2}
# Re-read whenever it changes, so speed/quality can be tuned without a restart.
TUNE_FILE = Path(__file__).with_name("avatar_tune.json")
_tune_mtime = None


def apply_tuning():
    global AVATAR_MAX_AREA, SAMPLE_STEPS, FAST_CLIP_TRANSITION, _tune_mtime

    try:
        mtime = TUNE_FILE.stat().st_mtime
    except FileNotFoundError:
        return

    if mtime == _tune_mtime:
        return

    _tune_mtime = mtime

    try:
        import json

        tune = json.loads(TUNE_FILE.read_text())
        area = int(tune.get("max_area", AVATAR_MAX_AREA))
        steps = int(tune.get("sample_steps", SAMPLE_STEPS))
        fast_clip = bool(tune.get("fast_clip_transition", FAST_CLIP_TRANSITION))
    except Exception as exc:
        print(f"[TUNE] ignored invalid {TUNE_FILE.name}: {exc!r}", flush=True)
        return

    if not (16384 <= area <= 288000 and 1 <= steps <= 4):
        print(f"[TUNE] out of range: area={area} steps={steps}", flush=True)
        return

    if steps != SAMPLE_STEPS and pipeline is not None:
        # The pipeline caches its sampling schedule on first use.
        pipeline._sampler_timesteps = None

    AVATAR_MAX_AREA, SAMPLE_STEPS, FAST_CLIP_TRANSITION = area, steps, fast_clip
    print(
        f"[TUNE] max_area={area} sample_steps={steps} fast_clip_transition={fast_clip}",
        flush=True,
    )


# horse_id -> list of JPEG bytes, generated once at startup.
IDLE_CLIPS = {}
IDLE_CLIP_ERRORS = {}
# =========================================================
# SESSIONS
# =========================================================

# class AvatarSession:

#     def __init__(
#         self,
#         session_id: str,
#         horse_id: str,
#         image_path: Path,
#     ):
#         self.session_id = session_id
#         self.horse_id = horse_id
#         self.image_path = image_path

#         self.frame_queue = queue.Queue(
#             maxsize=30
#         )

#         self.running = True

#         self.websocket = None

#         self.generation_thread = None

#         self.last_error = None

class AvatarSession:

    def __init__(
        self,
        session_id: str,
        horse_id: str,
        image_path: Path,
    ):
        self.session_id = session_id
        self.horse_id = horse_id
        self.image_path = image_path

        # Unbounded on purpose: the browser schedules frame N at N/fps on the
        # audio clock, so a dropped frame shifts every later frame out of
        # lip sync. A 30 s answer is ~750 JPEGs (~25 MB) at most.
        self.media_queue = queue.Queue()

        # One worker per session: encodes chunks off the GPU thread while
        # keeping chunk order (and END after the last frame).
        self.encoder = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"enc-{session_id[:8]}",
        )

        self.running = True

        self.websocket = None

        self.generation_thread = None

        self.last_error = None

        self.request_time = time.perf_counter()
        self.last_activity = time.time()

        # Set when the user interrupts: generation stops at the next chunk.
        self.cancel = threading.Event()

        # Audio being streamed in for the reply that is currently generating.
        self.stream = None

        # Turns the user interrupted; late audio for them is ignored.
        self.cancelled_turns = set()

sessions = {}
MAX_SESSIONS = int(os.environ.get("AVATAR_MAX_SESSIONS", "8"))
SESSION_TTL_SECONDS = int(os.environ.get("AVATAR_SESSION_TTL", "1800"))


# =========================================================
# REQUESTS
# =========================================================

class StartSessionRequest(BaseModel):

    horse_id: str


class AudioGenerationRequest(BaseModel):

    session_id: str
    audio_path: str = ""
    infer_frames: int = 48
    sample_steps: int = 3
    # infer_frames: int = 48
    # sample_steps: int = 4


# =========================================================
# LOAD MODEL
# =========================================================

def initialize_distributed():

    if not torch.distributed.is_available():
        return

    if torch.distributed.is_initialized():
        return

    os.environ.setdefault(
        "MASTER_ADDR",
        "127.0.0.1",
    )

    os.environ.setdefault(
        "MASTER_PORT",
        "29501",
    )

    torch.distributed.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
    )


def load_model():

    global pipeline
    global training_settings

    print("=" * 70)
    print("Loading LiveAvatar")
    print("=" * 70)

    initialize_distributed()

    cfg = WAN_CONFIGS["s2v-14B"]

    pipeline = WanS2V(

        config=cfg,

        checkpoint_dir=str(
            CKPT_DIR
        ),

        device_id=0,

        rank=0,

        t5_fsdp=False,

        dit_fsdp=False,

        use_sp=False,

        sp_size=1,

        t5_cpu=False,

        convert_model_dtype=True,

        single_gpu=True,

        offload_kv_cache=False,
    )

    # -----------------------------------------------------
    # LoRA
    # -----------------------------------------------------

    training_settings = (
        parse_args_for_training_config(
            str(TRAINING_CONFIG)
        )
    )

    pipeline.noise_model = (
        pipeline.add_lora_to_model(
            pipeline.noise_model,

            lora_rank=training_settings[
                "lora_rank"
            ],

            lora_alpha=training_settings[
                "lora_alpha"
            ],

            lora_target_modules=training_settings[
                "lora_target_modules"
            ],

            init_lora_weights=training_settings[
                "init_lora_weights"
            ],

            # pretrained_lora_path=(
            #     "Quark-Vision/Live-Avatar"
            # ),
            pretrained_lora_path=os.environ.get(
                "LIVEAVATAR_LORA_DIR",
                "/runpod-volume/models/Live-Avatar",
            ),
            load_lora_weight_only=False,
        )
    )

    # -----------------------------------------------------
    # FP8
    # -----------------------------------------------------

    try:

        from liveavatar.utils.fp8_linear import (
            replace_linear_with_scaled_fp8
        )

        replace_linear_with_scaled_fp8(
            pipeline.noise_model,
            ignore_keys=[
                "text_embedding",
                "time_embedding",
                "time_projection",
                "head.head",
                "casual_audio_encoder.encoder.final_linear",
            ],
        )

        print("FP8 enabled for noise model linears.")
    except Exception as exc:

        print(
            "FP8 initialization failed:",
            repr(exc),
        )

    print("=" * 70)
    print("LiveAvatar READY")
    print("=" * 70)

# =========================================================
# PER-REQUEST SETUP CACHES
# =========================================================

def install_prep_caches():
    """
    Every reply re-encodes the same prompt with T5 and the same horse photo
    with the VAE (about 3.5 s before the first frame). Both results depend
    only on their inputs, so keep them.
    """

    from collections import OrderedDict

    original_encode_prompt = pipeline.encode_prompt
    prompt_cache = {}

    def cached_encode_prompt(input_prompt, n_prompt=None, offload_model=True):
        key = (input_prompt, n_prompt)
        if key not in prompt_cache:
            prompt_cache[key] = original_encode_prompt(
                input_prompt, n_prompt, offload_model
            )
        return prompt_cache[key]

    pipeline.encode_prompt = cached_encode_prompt

    original_vae_encode = pipeline.vae.encode
    vae_cache = OrderedDict()

    def cached_vae_encode(videos):
        if not torch.is_tensor(videos):
            return original_vae_encode(videos)

        data = videos.detach().double()
        key = (
            tuple(videos.shape),
            str(videos.dtype),
            float(data.sum()),
            float(data.abs().sum()),
        )

        if key in vae_cache:
            vae_cache.move_to_end(key)
            return vae_cache[key]

        result = original_vae_encode(videos)
        vae_cache[key] = result
        if len(vae_cache) > 16:
            vae_cache.popitem(last=False)
        return result

    pipeline.vae.encode = cached_vae_encode


# =========================================================
# REALTIME PIPELINE PREPARATION
# =========================================================

def prepare_realtime_pipeline():
    """
    Keep the models required for realtime generation on GPU.

    This is intentionally done once during FastAPI startup so that
    the first user request does not have to perform CPU -> GPU
    model transfers.
    """

    global pipeline_ready

    if pipeline is None:
        raise RuntimeError(
            "Cannot prepare realtime pipeline before model is loaded"
        )

    print("=" * 70)
    print("Preparing realtime avatar pipeline...")
    print("=" * 70)

    device = pipeline.device
    dtype = pipeline.param_dtype

    # -----------------------------------------------------
    # NOISE / DiT MODEL
    # -----------------------------------------------------

    print("[WARMUP] Moving noise model to GPU...")

    pipeline.noise_model.to(
        device=device
      )

    pipeline.noise_model.eval()
    pipeline.noise_model.requires_grad_(False)

    # -----------------------------------------------------
    # VAE
    # -----------------------------------------------------

    print("[WARMUP] Moving VAE to GPU...")

    pipeline.vae.model.to(
        device=device,
        dtype=dtype,
    )

    pipeline.vae.model.eval()
    pipeline.vae.model.requires_grad_(False)

    # -----------------------------------------------------
    # AUDIO ENCODER
    # -----------------------------------------------------

    print("[WARMUP] Moving audio encoder to GPU...")

    pipeline.audio_encoder.model.to(
        device=device,
        dtype=dtype,
    )

    pipeline.audio_encoder.model.eval()
    pipeline.audio_encoder.model.requires_grad_(False)

    # -----------------------------------------------------
    # TEXT ENCODER
    #
    # This is useful because realtime generation should not
    # have to perform another CPU -> GPU transfer.
    # -----------------------------------------------------

    print("[WARMUP] Moving text encoder to GPU...")

    pipeline.text_encoder.model.to(
        device=device,
    )

    pipeline.text_encoder.model.eval()
    pipeline.text_encoder.model.requires_grad_(False)

    # -----------------------------------------------------
    # CUDA INITIALIZATION
    # -----------------------------------------------------

    if torch.cuda.is_available():

        print("[WARMUP] Initializing CUDA...")

        torch.cuda.synchronize(device)

        # Small CUDA operation to initialize the CUDA context.
        warmup_tensor = torch.zeros(
            (1,),
            device=device,
            dtype=dtype,
        )

        warmup_tensor = warmup_tensor + 1

        torch.cuda.synchronize(device)

        del warmup_tensor

    # Everything stays on the GPU. Without this the pipeline moves the text and
    # audio encoders to the CPU (and empties the CUDA cache) at the start of
    # every reply, which costs time and breaks background audio preparation.
    pipeline.init_on_cpu = False

    pipeline_ready = True

    print("=" * 70)
    print("Realtime avatar pipeline READY")
    print(f"Device : {device}")
    print(f"Dtype  : {dtype}")
    print("Models remain loaded on GPU.")
    print("=" * 70)
# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup():

    load_model()
    install_prep_caches()
    if REALTIME_MODE:
        prepare_realtime_pipeline()
        if os.environ.get("AVATAR_WARMUP", "1") == "1":
            warmup_generation()
        # The UI draws its own idle animation; enable only if a client needs
        # GET /horse/{id}/idle.
        if os.environ.get("AVATAR_PRECOMPUTE_IDLE") == "1":
            precompute_idle_clips()


def write_silent_wav(path: Path, seconds: int):
    sample_rate = 16_000
    samples = (np.random.randn(sample_rate * seconds) * 30).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def precompute_idle_clips():
    """
    Generate each horse's idle loop once (closed mouth, blinking, body
    movement) so a session never waits for the GPU to get an idle animation.
    Served by GET /horse/{horse_id}/idle.
    """

    audio_dir = Path("/tmp/liveavatar_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    for horse_id, horse in HORSES.items():
        if not horse["image"].exists():
            IDLE_CLIP_ERRORS[horse_id] = "horse image not found"
            continue

        idle_audio = audio_dir / f"idle_{horse_id}.wav"
        write_silent_wav(idle_audio, IDLE_CLIP_SECONDS)
        frames = []

        def collect(video_tensor, frames=frames):
            for frame in to_uint8_frames(video_tensor):
                frames.append(encode_jpeg(frame))

        started = time.perf_counter()
        print(f"[IDLE] generating idle clip for horse {horse_id}...", flush=True)

        try:
            with generation_lock:
                run_pipeline(
                    horse["image"],
                    idle_audio,
                    IDLE_INFER_FRAMES,
                    IDLE_SAMPLE_STEPS,
                    collect,
                    IDLE_PROMPT,
                )
            IDLE_CLIPS[horse_id] = frames
            print(
                f"[IDLE] horse {horse_id}: {len(frames)} frames in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:
            IDLE_CLIP_ERRORS[horse_id] = repr(exc)
            print(f"[IDLE] horse {horse_id} failed: {exc!r}", flush=True)
        finally:
            idle_audio.unlink(missing_ok=True)


@app.get("/horse/{horse_id}/idle")
async def horse_idle(horse_id: str):
    """Idle loop as repeated [uint32 big-endian length][JPEG bytes]."""

    frames = IDLE_CLIPS.get(horse_id)
    if not frames:
        raise HTTPException(
            503,
            IDLE_CLIP_ERRORS.get(horse_id, "Idle clip is not ready yet"),
        )

    body = b"".join(struct.pack("!I", len(jpeg)) + jpeg for jpeg in frames)
    return Response(content=body, media_type="application/octet-stream")


WARMUP_SECONDS = tuple(
    int(x.strip()) for x in os.environ.get("AVATAR_WARMUP_SECONDS", "3").split(",") if x.strip()
) or (3,)


def write_speech_like_wav(path: Path, seconds: int):
    """Loud, modulated tone: not silence, so it takes the normal reply path."""

    sample_rate = 16_000
    t = np.arange(sample_rate * seconds) / sample_rate
    samples = (
        np.sin(2 * np.pi * 140 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 6000
        + np.random.randn(t.size) * 200
    ).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def warmup_generation():
    """
    Generate throwaway replies before serving requests.

    The first generation after startup (or at a new frame size / audio window
    length) spends seconds on kernel selection and allocator growth. This runs
    the same code path as a real streamed reply, twice per horse, with
    speech-like audio, so users never pay for it.
    """

    print("[WARMUP] running warm-up generations...", flush=True)
    started = time.perf_counter()

    for horse_id, horse in HORSES.items():
        if not horse["image"].exists():
            print(f"[WARMUP] skipped horse {horse_id}: image not found", flush=True)
            continue

        for seconds in WARMUP_SECONDS:
            session = AvatarSession(
                session_id=f"warmup-{horse_id}",
                horse_id=horse_id,
                image_path=horse["image"],
            )
            stream = AudioStream()
            stream.turn = "warmup"
            session.stream = stream
            horse_started = time.perf_counter()

            try:
                wav_path = Path("/tmp/liveavatar_audio") / f"warmup_{horse_id}.wav"
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                write_speech_like_wav(wav_path, seconds)
                stream.append(decode_audio_bytes(wav_path.read_bytes()))
                stream.finish()
                wav_path.unlink(missing_ok=True)

                worker = threading.Thread(
                    target=generate_avatar_stream,
                    args=(session, stream, 48),
                    daemon=True,
                )
                worker.start()
                worker.join()

                if session.last_error:
                    print(f"[WARMUP] horse {horse_id} error: {session.last_error}", flush=True)
                print(
                    f"[WARMUP] horse {horse_id} ({seconds}s audio) done in "
                    f"{time.perf_counter() - horse_started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                # Non-fatal: the service still works, only the first reply is slower.
                print(f"[WARMUP] horse {horse_id} failed: {exc!r}", flush=True)
            finally:
                session.running = False
                session.encoder.shutdown(wait=False, cancel_futures=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"[WARMUP] warm-up generation done in {time.perf_counter() - started:.2f}s",
        flush=True,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/ping")
async def ping():
    # Required by a custom FastAPI load-balanced RunPod worker.
    # Do not use this as the model-readiness check; /health does that.
    return {"status": "ok"}

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "pipeline_ready": pipeline_ready,
        "realtime_mode": REALTIME_MODE,
        "sessions": len(sessions),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


# =========================================================
# SESSION CLEANUP
# =========================================================

def cleanup_stale_sessions():
    now = time.time()
    stale = []
    for sid, session in list(sessions.items()):
        if now - session.last_activity > SESSION_TTL_SECONDS:
            stale.append(sid)

    for sid in stale:
        session = sessions.pop(sid, None)
        if session is None:
            continue
        session.running = False
        session.cancel.set()
        if session.stream is not None:
            session.stream.cancel()
        try:
            session.encoder.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        print(f"[SESSION] cleaned stale session={sid}", flush=True)


def clear_media_queue(session: AvatarSession):
    while True:
        try:
            session.media_queue.get_nowait()
        except queue.Empty:
            return


# =========================================================
# START SESSION
# =========================================================

@app.post("/session/start")
async def start_session(
    request: StartSessionRequest,
):

    cleanup_stale_sessions()

    if pipeline is None:

        raise HTTPException(
            503,
            "LiveAvatar model is not loaded",
        )

    horse = HORSES.get(
        request.horse_id
    )

    if horse is None:

        raise HTTPException(
            404,
            f"Unknown horse: {request.horse_id}",
        )

    image_path = horse["image"]

    if not image_path.exists():

        raise HTTPException(
            404,
            f"Horse image not found: {image_path}",
        )

    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(503, "Avatar worker is at session capacity")

    session_id = uuid.uuid4().hex

    session = AvatarSession(
        session_id=session_id,
        horse_id=request.horse_id,
        image_path=image_path,
    )

    sessions[session_id] = session

    return {
        "session_id": session_id,
        "horse_id": request.horse_id,
        "status": "ready",
    }


# =========================================================
# STOP SESSION
# =========================================================

@app.post("/session/stop/{session_id}")
async def stop_session(
    session_id: str,
):

    session = sessions.pop(
        session_id,
        None,
    )

    if session is None:

        return {
            "status": "already_stopped"
        }

    session.running = False
    session.last_activity = time.time()
    session.cancel.set()
    if session.stream is not None:
        session.stream.cancel()
    clear_media_queue(session)
    session.encoder.shutdown(wait=False, cancel_futures=True)

    return {
        "status": "stopped",
        "session_id": session_id,
    }


# =========================================================
# CANCEL (user interrupted the horse)
# =========================================================

@app.post("/session/cancel/{session_id}")
async def cancel_generation(session_id: str, turn: str = ""):

    session = sessions.get(session_id)

    if session is None:
        return {"status": "unknown_session"}

    if turn:
        session.cancelled_turns.add(turn)
        if len(session.cancelled_turns) > 50:
            session.cancelled_turns.pop()

    session.cancel.set()
    if session.stream is not None:
        session.stream.cancel()

    return {"status": "cancelling"}


# =========================================================
# JPEG ENCODING
# =========================================================

def to_uint8_frames(video_tensor):
    """
    [1, 3, T, H, W] or [3, T, H, W] float in [-1, 1]
    -> numpy uint8 [T, H, W, 3].

    One vectorized op on the tensor's own device and a single
    device->host copy, instead of per-frame CPU math.
    """

    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]

    return (
        video_tensor
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def encode_jpeg(frame: np.ndarray) -> bytes:

    buffer = io.BytesIO()

    Image.fromarray(frame).save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
    )

    return buffer.getvalue()


# =========================================================
# FRAME CALLBACK
# =========================================================

class EnoughFrames(BaseException):
    """Raised from the stream callback to stop generating past the audio's end.

    BaseException on purpose: the pipeline swallows Exception from callbacks.
    """


def audio_frame_budget(audio_path):
    """Frames needed to cover the audio (+ a small margin), or None if unknown.

    The pipeline generates whole 48-frame clips, so a 4 s answer would
    otherwise render ~140 frames instead of ~100 and hold the GPU for
    the extra time.
    """

    try:
        import soundfile

        seconds = soundfile.info(str(audio_path)).duration
    except Exception:
        return None

    if not seconds or seconds <= 0:
        return None

    return int(seconds * AVATAR_FPS) + 4


def make_frame_callback(session: AvatarSession, max_frames=None):
    """
    Called by the pipeline on the GPU thread for every decoded block
    (12 new frames; the tensor is still on the GPU).

    It only copies the chunk to host memory and hands encoding to the
    session's encoder thread, so the GPU starts the next chunk
    immediately instead of waiting for JPEG encoding + queueing.
    """

    callback_count = 0
    produced = [0]
    generation_start = time.perf_counter()
    last_callback = generation_start

    def encode_and_queue(frames, chunk_index):

        if not session.running:
            return

        started = time.perf_counter()

        try:
            for jpeg in JPEG_POOL.map(encode_jpeg, frames):
                session.media_queue.put(
                    make_media_packet(MEDIA_FRAME, jpeg)
                )
        except Exception as exc:
            session.last_error = f"Frame encoding failed: {exc}"
            print(f"[STREAM] encode error: {exc}", flush=True)
            return

        print(
            f"[STREAM] chunk #{chunk_index} encoded {len(frames)} frames "
            f"in {time.perf_counter() - started:.2f}s",
            flush=True,
        )

    def callback(video_tensor):
        nonlocal callback_count, last_callback

        if not session.running:
            return

        if session.cancel.is_set():
            print("[STREAM] cancelled by client; stopping generation", flush=True)
            raise EnoughFrames()

        now = time.perf_counter()
        interval = now - last_callback
        last_callback = now
        callback_count += 1

        try:
            frames = to_uint8_frames(video_tensor)
        except Exception as exc:
            session.last_error = f"Frame conversion failed: {exc}"
            print(f"[STREAM] callback error: {exc}", flush=True)
            return

        # >= 1.0 means this chunk was generated at least as fast as it plays.
        video_seconds = len(frames) / AVATAR_FPS
        realtime_ratio = video_seconds / interval if interval > 0 else 0.0

        print(
            f"[STREAM] CALLBACK #{callback_count} "
            f"t={now - generation_start:.2f}s "
            f"chunk_time={interval:.2f}s "
            f"frames={len(frames)} ({video_seconds:.2f}s video) "
            f"realtime_x={realtime_ratio:.2f}",
            flush=True,
        )

        try:
            session.encoder.submit(encode_and_queue, frames, callback_count)
        except RuntimeError:
            # Encoder was shut down because the session stopped.
            pass

        produced[0] += len(frames)

        if produced[0] == len(frames):
            print(
                f"[TIMING] first chunk {now - session.request_time:.2f}s "
                f"after the generate request",
                flush=True,
            )

        limit = max_frames() if callable(max_frames) else max_frames

        if limit is not None and produced[0] >= limit:
            print(
                f"[STREAM] enough frames ({produced[0]}/{limit}); "
                f"stopping generation early",
                flush=True,
            )
            raise EnoughFrames()

    return callback


# =========================================================
# PIPELINE CALL
# =========================================================

def is_silent_audio(audio_path) -> bool:
    """True for the near-silent WAV the UI sends to request an idle clip.

    Only WAV is inspected; anything else (the MP3 TTS replies) counts as speech.
    """

    try:
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getsampwidth() != 2:
                return False
            samples = np.frombuffer(
                wav.readframes(wav.getnframes()), dtype=np.int16
            )
    except Exception:
        return False

    if samples.size == 0:
        return False

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return rms < SILENCE_RMS_THRESHOLD


def run_pipeline(
    image_path,
    audio_path,
    infer_frames,
    sample_steps,
    callback,
    prompt=SPEECH_PROMPT,
    audio_provider=None,
):

    with torch.no_grad():
        pipeline.generate(
            input_prompt=prompt,
            ref_image_path=str(image_path),
            audio_path=str(audio_path) if audio_path else None,
            enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None,
            # num_repeat = number of 48-frame clips; None covers the whole
            # audio (1 would cap every answer at ~1.9 s of video).
            tts_text=None, num_repeat=None, pose_video=None, max_area=AVATAR_MAX_AREA,
            infer_frames=infer_frames, shift=5.0, sample_solver="euler",
            sampling_steps=sample_steps, guide_scale=0, seed=-1,
            offload_model=False, init_first_frame=False, use_dataset=False,
            dataset_sample_idx=0, drop_motion_noisy=False, num_gpus_dit=1,
            enable_vae_parallel=False, input_video_for_sam2=None,
            enable_online_decode=True, stream_callback=callback,
            fast_clip_transition=FAST_CLIP_TRANSITION,
            audio_provider=audio_provider,
        )


# =========================================================
# AUDIO GENERATION
# =========================================================
def generate_avatar_video(session: AvatarSession, audio_path: str, infer_frames: int, sample_steps: int):
    try:
        print(f"[AVATAR DEBUG] session={session.session_id}", flush=True)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            raise ValueError("Avatar generation audio is empty")

        if session.cancel.is_set():
            session.media_queue.put(make_media_packet(MEDIA_START))
            session.encoder.submit(
                session.media_queue.put, make_media_packet(MEDIA_END)
            )
            return

        idle = is_silent_audio(audio_path)
        if idle:
            print("[AVATAR] silent audio -> idle prompt", flush=True)

        with generation_lock:
            apply_tuning()
            session.media_queue.put(make_media_packet(MEDIA_START))
            session.media_queue.put(make_media_packet(MEDIA_AUDIO, audio_bytes))
            print(f"[STREAM] queued audio bytes={len(audio_bytes)}", flush=True)

            started = time.perf_counter()
            print(
                f"[TIMING] lock acquired {started - session.request_time:.2f}s "
                f"after the generate request",
                flush=True,
            )
            try:
                run_pipeline(
                    session.image_path,
                    audio_path,
                    infer_frames,
                    SAMPLE_STEPS,
                    make_frame_callback(
                        session,
                        None if idle else audio_frame_budget(audio_path),
                    ),
                    IDLE_PROMPT if idle else SPEECH_PROMPT,
                )
            except EnoughFrames:
                pass
            print(
                f"[AVATAR DEBUG] pipeline.generate() finished "
                f"in {time.perf_counter() - started:.2f}s",
                flush=True,
            )

        # Queue END through the encoder so it lands after the last frame.
        # Done outside the GPU lock: the next generation can start while
        # the final chunk is still being encoded.
        if session.running:
            session.encoder.submit(
                session.media_queue.put,
                make_media_packet(MEDIA_END),
            )
    except Exception as exc:
        import traceback
        session.last_error = f"{type(exc).__name__}: {exc}"
        print(f"Avatar generation error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as cleanup_error:
            print(f"[AVATAR DEBUG] Audio cleanup failed: {cleanup_error}", flush=True)

# =========================================================
# GENERATE
# =========================================================
@app.post("/session/generate")
async def generate_session(
    request: AudioGenerationRequest,
):

    session = sessions.get(
        request.session_id
    )

    if session is None:

        raise HTTPException(
            404,
            "Avatar session not found",
        )

    # -----------------------------------------------------
    # ORIGINAL AUDIO FROM DJANGO
    # -----------------------------------------------------

    if not request.audio_path:
        raise HTTPException(
            400,
            "audio_path is no longer accepted for remote Django audio. "
            "Use POST /session/{session_id}/audio with the audio bytes.",
        )

    source_audio = Path(request.audio_path).expanduser().resolve()

    if not source_audio.exists() or not source_audio.is_file():
        raise HTTPException(404, "Audio file not found on the avatar worker")

    session.last_activity = time.time()

    # -----------------------------------------------------
    # DON'T ALLOW TWO GENERATIONS FOR SAME SESSION
    # -----------------------------------------------------

    if session.generation_thread:

        if session.generation_thread.is_alive():

            raise HTTPException(
                409,
                "Avatar is already generating",
            )

    # -----------------------------------------------------
    # CREATE SAFE AUDIO DIRECTORY
    # -----------------------------------------------------

    avatar_audio_dir = Path(
        "/tmp/liveavatar_audio"
    )

    avatar_audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # COPY AUDIO
    #
    # Django owns source_audio.
    # FastAPI owns generation_audio.
    # -----------------------------------------------------

    generation_audio = (
        avatar_audio_dir
        / f"{request.session_id}_{uuid.uuid4().hex}.mp3"
    )

    try:

        shutil.copy2(
            source_audio,
            generation_audio,
        )

    except Exception as exc:

        raise HTTPException(
            500,
            f"Failed to copy avatar audio: {exc}",
        )

    # -----------------------------------------------------
    # VERIFY COPY
    # -----------------------------------------------------

    if not generation_audio.exists():

        raise HTTPException(
            500,
            "Avatar audio copy was not created",
        )

    print(
        f"[AVATAR GENERATE] "
        f"session={request.session_id}",
        flush=True,
    )

    print(
        f"[AVATAR GENERATE] "
        f"source_audio={source_audio}",
        flush=True,
    )

    print(
        f"[AVATAR GENERATE] "
        f"generation_audio={generation_audio}",
        flush=True,
    )

    print(
        f"[AVATAR GENERATE] "
        f"generation_audio_exists="
        f"{generation_audio.exists()}",
        flush=True,
    )

    print(
        f"[AVATAR GENERATE] "
        f"generation_audio_size="
        f"{generation_audio.stat().st_size}",
        flush=True,
    )

    session.last_error = None
    session.cancel.clear()
    session.request_time = time.perf_counter()

    # -----------------------------------------------------
    # START BACKGROUND GPU GENERATION
    # -----------------------------------------------------

    session.generation_thread = (
        threading.Thread(
            target=generate_avatar_video,
            args=(
                session,
                str(generation_audio),
                request.infer_frames,
                request.sample_steps,
            ),
            daemon=True,
        )
    )

    session.generation_thread.start()

    # -----------------------------------------------------
    # RETURN IMMEDIATELY
    # -----------------------------------------------------

    return {
        "status": "started",
        "session_id": request.session_id,
    }
# =========================================================
# STREAMING AUDIO IN (reply audio arrives while video is generated)
# =========================================================

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE // AVATAR_FPS  # 640
# Audio features need a little future context beyond a clip's own frames.
STREAM_LOOKAHEAD_FRAMES = 5
# Frames of earlier audio kept in front of each clip's window.
STREAM_CONTEXT_FRAMES = 25
# How long the generator waits for more audio before giving up.
STREAM_AUDIO_TIMEOUT = 60.0


class AudioStream:
    """Growing 16 kHz mono audio of one reply."""

    def __init__(self):
        self.cond = threading.Condition()
        self.chunks = []
        self.total = 0
        self.finished = False
        self.cancelled = False
        self.turn = ""

    def append(self, samples: np.ndarray):
        with self.cond:
            self.chunks.append(samples)
            self.total += len(samples)
            self.cond.notify_all()

    def finish(self):
        with self.cond:
            self.finished = True
            self.cond.notify_all()

    def cancel(self):
        with self.cond:
            self.cancelled = True
            self.finished = True
            self.cond.notify_all()

    def frames(self) -> int:
        return self.total // SAMPLES_PER_FRAME

    def wait_frames(self, needed: int) -> bool:
        """Blocks until `needed` frames of audio exist. False if it ended first."""

        deadline = time.perf_counter() + STREAM_AUDIO_TIMEOUT
        with self.cond:
            while self.total // SAMPLES_PER_FRAME < needed:
                if self.finished:
                    return False
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self.finished = True
                    return False
                self.cond.wait(timeout=min(remaining, 1.0))
            return True

    def window(self, start_frame: int, end_frame: int) -> np.ndarray:
        with self.cond:
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        return audio[start_frame * SAMPLES_PER_FRAME : end_frame * SAMPLES_PER_FRAME]


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """Decode WAV/FLAC/OGG/MP3/etc. to float32 mono 16 kHz.

    soundfile is preferred for native formats. FFmpeg is the production
    fallback because ElevenLabs commonly returns MP3 audio.
    """
    import soundfile

    try:
        samples, rate = soundfile.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
        mono = samples.mean(axis=1)
        if rate != SAMPLE_RATE:
            import librosa
            mono = librosa.resample(
                mono, orig_sr=rate, target_sr=SAMPLE_RATE
            )
        return np.ascontiguousarray(mono, dtype=np.float32)
    except Exception as soundfile_error:
        with tempfile.TemporaryDirectory(prefix="liveavatar-audio-") as tmp:
            src = Path(tmp) / "input.bin"
            src.write_bytes(data)
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(src),
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "f32le", "pipe:1",
            ]
            try:
                result = subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=30,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Audio decode failed with soundfile ({soundfile_error}) "
                    "and ffmpeg is not installed"
                ) from exc
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.decode("utf-8", "ignore")[-500:]
                raise RuntimeError(
                    f"Audio decode failed: {detail}"
                ) from exc
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            if audio.size == 0:
                raise RuntimeError("Decoded audio is empty")
            return np.ascontiguousarray(audio, dtype=np.float32)


def encode_audio_window(samples: np.ndarray, infer_frames: int):
    """Audio features for a window of samples, shaped like pipeline.encode_audio."""

    z = pipeline.audio_encoder.extract_audio_feat_from_array(
        samples, SAMPLE_RATE, return_all_layers=True
    )
    bucket, _ = pipeline.audio_encoder.get_audio_embed_bucket_fps(
        z,
        fps=pipeline.fps,
        batch_frames=infer_frames,
        m=pipeline.audio_sample_m,
    )
    bucket = bucket.to(pipeline.device, pipeline.param_dtype).unsqueeze(0)
    if len(bucket.shape) == 3:
        bucket = bucket.permute(0, 2, 1)
    elif len(bucket.shape) == 4:
        bucket = bucket.permute(0, 2, 3, 1)
    return bucket


def make_audio_provider(session: AvatarSession, stream: AudioStream, infer_frames: int):
    """
    Called by the pipeline once per clip; returns that clip's audio features.

    The next clip's features are prepared in the background while the current
    clip renders, so clip boundaries don't stall the video.
    """

    def compute(clip_index: int):
        first = clip_index * infer_frames
        last = first + infer_frames

        # Wait for the clip's audio (plus a little context after it). If the
        # reply ends earlier, a shorter clip is padded with silence.
        stream.wait_frames(last + STREAM_LOOKAHEAD_FRAMES)

        if session.cancel.is_set() or stream.cancelled:
            return None
        if first >= stream.frames() and stream.finished:
            return None

        start = max(0, first - STREAM_CONTEXT_FRAMES)
        end = min(stream.frames(), last + STREAM_LOOKAHEAD_FRAMES + STREAM_CONTEXT_FRAMES)
        samples = stream.window(start, end)
        if samples.size == 0:
            return None

        # Always the same length: the GPU tunes its kernels per input length, and a
        # new length in the middle of a reply costs seconds.
        target = (
            STREAM_CONTEXT_FRAMES + infer_frames + STREAM_LOOKAHEAD_FRAMES + STREAM_CONTEXT_FRAMES
        ) * SAMPLES_PER_FRAME
        if samples.size < target:
            samples = np.pad(samples, (0, target - samples.size), mode="constant")
        else:
            samples = samples[:target]

        with torch.no_grad():
            features = encode_audio_window(samples, infer_frames)

        offset = first - start
        return features[..., offset : offset + infer_frames]

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-feat")
    pending = {}

    def provider(clip_index: int):
        future = pending.pop(clip_index, None) or executor.submit(compute, clip_index)
        features = future.result()
        if features is not None:
            pending[clip_index + 1] = executor.submit(compute, clip_index + 1)
        return features

    provider.close = lambda: executor.shutdown(wait=False, cancel_futures=True)
    return provider


def generate_avatar_stream(session: AvatarSession, stream: AudioStream, infer_frames: int):
    try:
        with generation_lock:
            apply_tuning()
            started = time.perf_counter()
            print(
                f"[TIMING] stream lock acquired {started - session.request_time:.2f}s "
                f"after the first audio",
                flush=True,
            )

            def frame_limit():
                # Only known once the reply's audio has ended.
                if not stream.finished:
                    return None
                return stream.frames() + 4

            provider = make_audio_provider(session, stream, infer_frames)
            try:
                run_pipeline(
                    session.image_path,
                    None,
                    infer_frames,
                    SAMPLE_STEPS,
                    make_frame_callback(session, frame_limit),
                    SPEECH_PROMPT,
                    audio_provider=provider,
                )
            except EnoughFrames:
                pass
            finally:
                provider.close()

            print(
                f"[AVATAR DEBUG] streamed generation finished in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )

        if session.running:
            session.encoder.submit(
                session.media_queue.put,
                make_media_packet(MEDIA_END, stream.turn.encode()),
            )
    except Exception as exc:
        import traceback

        session.last_error = f"{type(exc).__name__}: {exc}"
        print(f"Avatar stream error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        if session.running:
            session.media_queue.put(make_media_packet(MEDIA_END, stream.turn.encode()))
    finally:
        stream.finish()
        if session.stream is stream:
            session.stream = None


@app.post("/session/{session_id}/audio")
async def append_session_audio(
    session_id: str,
    request: Request,
    last: bool = False,
    infer_frames: int = 48,
    turn: str = "",
):
    """
    Adds audio (any format, e.g. an MP3 sentence) to the reply being spoken.
    The first call starts generating; later calls extend it. `last=true`
    marks the end of the reply.
    """

    session = sessions.get(session_id)

    if session is None:
        raise HTTPException(404, "Avatar session not found")

    session.last_activity = time.time()

    if turn and turn in session.cancelled_turns:
        return {"status": "cancelled"}

    data = await request.body()

    samples = np.zeros(0, dtype=np.float32)
    if data:
        try:
            samples = await asyncio.to_thread(decode_audio_bytes, data)
        except Exception as exc:
            raise HTTPException(400, f"Could not decode audio: {exc}")

    stream = session.stream
    started = False

    if stream is None or stream.finished:
        if session.generation_thread and session.generation_thread.is_alive():
            raise HTTPException(409, "Avatar is already generating")
        if samples.size == 0:
            return {"status": "empty"}

        stream = AudioStream()
        stream.turn = turn
        session.stream = stream
        session.last_error = None
        session.cancel.clear()
        session.request_time = time.perf_counter()
        # The turn id rides on START and END so the browser can drop a reply
        # the user has already interrupted.
        session.media_queue.put(make_media_packet(MEDIA_START, turn.encode()))
        started = True

    if samples.size:
        stream.append(samples)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        session.media_queue.put(make_media_packet(MEDIA_AUDIO_PCM, pcm))

    if last:
        stream.finish()

    if started:
        session.generation_thread = threading.Thread(
            target=generate_avatar_stream,
            args=(session, stream, infer_frames),
            daemon=True,
        )
        session.generation_thread.start()

    return {
        "status": "started" if started else "appended",
        "seconds": round(stream.total / SAMPLE_RATE, 2),
    }


# =========================================================
# FRAME WEBSOCKET
# =========================================================

@app.websocket("/session/{session_id}/frames")
async def frame_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.send_json({"type":"error","message":"Use /session/{session_id}/stream for synchronized audio + video."})
    await websocket.close(code=1000)


@app.websocket("/session/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: str):
    session = sessions.get(session_id)
    if session is None:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    session.websocket = websocket
    print(f"[STREAM WS] ACCEPTED session={session_id}", flush=True)
    try:
        while session.running:
            try:
                packet = await asyncio.to_thread(session.media_queue.get, True, 1)
            except queue.Empty:
                if session.last_error:
                    err=session.last_error; session.last_error=None
                    try: await websocket.send_json({"type":"error","message":err})
                    except Exception: break
                continue

            # Drain whatever else is already queued (a whole encoded chunk
            # usually is) without a thread hop per frame.
            batch = [packet]
            while len(batch) < 64:
                try:
                    batch.append(session.media_queue.get_nowait())
                except queue.Empty:
                    break

            try:
                for item in batch:
                    await websocket.send_bytes(item)
            except Exception as exc:
                print(f"[STREAM WS] send failed: {type(exc).__name__}: {exc}", flush=True)
                break
    except WebSocketDisconnect as exc:
        print(f"[STREAM WS] DISCONNECTED session={session_id} code={exc.code}", flush=True)
    except Exception as exc:
        import traceback; traceback.print_exc()
    finally:
        if session.websocket is websocket:
            session.websocket = None
        # A disconnected client cannot consume queued frames. Cancel the
        # generation to avoid wasting GPU and RAM on an orphaned session.
        if session.running:
            session.cancel.set()
            if session.stream is not None:
                session.stream.cancel()
        clear_media_queue(session)
        print(f"[STREAM WS] CLOSED session={session_id}", flush=True)
