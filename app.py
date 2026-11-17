import os
import sys
import uuid
import json
import asyncio
import shutil
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Prepend virtual environment directory so ffmpeg and other tools are in PATH
_bin_dir = os.path.dirname(sys.executable)
if _bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import clipper_core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clipper_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "media", "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "media", "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="AI Video Clipper Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/media/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

executor = ThreadPoolExecutor(max_workers=2)
jobs: Dict[str, Dict[str, Any]] = {}

def update_job(job_id: str, status: str, message: str, progress: int, **kwargs):
    if job_id in jobs:
        jobs[job_id]["status"] = status
        jobs[job_id]["message"] = message
        jobs[job_id]["progress"] = progress
        for k, v in kwargs.items():
            jobs[job_id][k] = v

def generate_upload_guide_text(clips: list, user_rules: Optional[str] = None) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("           REELS & SHORTS UPLOAD PLAYBOOK & STRATEGY GUIDE")
    lines.append("=" * 80)
    if user_rules and user_rules.strip():
        lines.append("DIRECTIVES & RULES APPLIED:")
        lines.append(f"\"{user_rules.strip()}\"")
        lines.append("-" * 80)
    lines.append(f"TOTAL REELS GENERATED: {len(clips)}")
    lines.append("FORMAT: 9:16 Vertical (Instagram Reels / TikTok / YouTube Shorts)")
    lines.append("=" * 80)
    lines.append("")

    for c in clips:
        cid = c.get("id", 1)
        title = c.get("title", f"Clip #{cid}")
        dur = c.get("duration", 0)
        st = c.get("start_time", 0)
        et = c.get("end_time", 0)
        st_str = f"{int(st//60):02d}:{int(st%60):02d}"
        et_str = f"{int(et//60):02d}:{int(et%60):02d}"

        lines.append("-" * 80)
        lines.append(f"REEL #{cid}: {title.upper()}")
        lines.append(f"File: clip_{cid}.mp4 | Duration: {dur}s | Timestamps: {st_str} - {et_str}")
        lines.append("-" * 80)
        lines.append("📌 RECOMMENDED TITLE:")
        lines.append(f"   {title}")
        lines.append("")
        lines.append("📝 CAPTION & HASHTAGS (Copy & Paste):")
        for cl in c.get("caption", "").split("\n"):
            lines.append(f"   {cl}")
        lines.append("")
        lines.append("💬 CALL TO ACTION (CTA):")
        lines.append(f"   {c.get('call_to_action', '')}")
        lines.append("")
        lines.append("📍 PINNED COMMENT (Post immediately after uploading):")
        lines.append(f"   {c.get('pinned_comment', '')}")
        lines.append("")
        lines.append("⏱️ OPTIMAL POSTING TIME:")
        lines.append(f"   {c.get('best_post_time', '12:00 PM - 2:00 PM or 7:00 PM - 9:00 PM')}")
        lines.append("")
        lines.append("🎵 RECOMMENDED AUDIO / SOUND VIBE:")
        lines.append(f"   {c.get('song_suggestion', 'Trending Beat')}")
        lines.append("")
        lines.append("🚀 HOW TO UPLOAD THIS REEL (Step-by-Step Instructions):")
        for inst in c.get("upload_instructions", "").split("\n"):
            lines.append(f"   • {inst.strip()}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("Follow these guidelines for maximum algorithmic engagement and reach!")
    lines.append("=" * 80)
    return "\n".join(lines)

def run_job_pipeline(job_id: str, video_path: str, options: Dict[str, Any]):
    try:
        def progress_cb(msg: str, pct: int):
            update_job(job_id, status="processing", message=msg, progress=pct)

        user_rules = str(options.get("user_rules") or "").strip()
        whisper_model = options.get("whisper_model", "base")
        ollama_model = options.get("ollama_model", "deepseek-r1:8b")
        api_key = options.get("api_key", None)
        add_captions = options.get("add_captions", True)

        # Step 1: Transcribe video audio locally using Whisper
        progress_cb(f"Extracting audio & transcribing locally ({whisper_model})...", 15)
        whisper_result = clipper_core.transcribe_video_locally(
            video_path=video_path,
            model_name=whisper_model,
            progress_callback=progress_cb
        )
        transcript = whisper_result.get("text", "")
        segments = whisper_result.get("segments", [])
        if not transcript.strip():
            raise ValueError("No speech or audio could be transcribed from the video.")

        # Probe video duration
        video_duration = None
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as v_clip:
                video_duration = float(v_clip.duration)
        except Exception:
            if segments:
                video_duration = max([s.get("end", 0.0) for s in segments], default=120.0)

        update_job(
            job_id,
            status="processing",
            message="Audio extracted. Comparing speech to rules & selecting top 10 moments...",
            progress=38,
            transcript=transcript
        )

        # Step 2: Extract top distinct clips comparing audio to rules, then enrich with AI
        progress_cb(f"Evaluating moments strictly against rules with {ollama_model}...", 45)
        clips_plan = clipper_core.analyze_transcript_with_ollama(
            transcript=transcript,
            model_name=ollama_model,
            clip_count=10,
            user_rules=user_rules,
            segments=segments,
            video_duration=video_duration,
            progress_callback=progress_cb,
            api_key=api_key
        )

        if not clips_plan:
            raise ValueError("No valid clips could be found matching your rules.")

        update_job(
            job_id,
            status="processing",
            message=f"Identified {len(clips_plan)} clips based on rules. Rendering 9:16 vertical reels...",
            progress=55,
            clips_plan=clips_plan
        )

        # Step 3: Render 9:16 Clips with MoviePy (Multi-threaded parallel workers)
        job_out_dir = os.path.join(OUTPUTS_DIR, job_id)
        os.makedirs(job_out_dir, exist_ok=True)

        def render_single_clip(item):
            idx, clip_info = item
            clip_num = idx + 1
            out_filename = f"clip_{clip_num}.mp4"
            out_filepath = os.path.join(job_out_dir, out_filename)

            dur = float(clip_info.get("duration", 20.0))

            clipper_core.create_video_clip(
                video_path=video_path,
                clip_data=clip_info,
                output_filename=out_filepath,
                target_duration=int(dur),
                add_caption=add_captions,
                caption_style="center"
            )

            actual_dur = round(float(clip_info.get("duration") or dur), 1)

            return (idx, {
                "id": clip_num,
                "title": clip_info.get("title", f"Reel #{clip_num}"),
                "caption": clip_info.get("caption", ""),
                "text_overlay": clip_info.get("text_overlay", f"Reel #{clip_num}"),
                "song_suggestion": clip_info.get("song_suggestion", "Trending Beat"),
                "call_to_action": clip_info.get("call_to_action", "Follow for more!"),
                "pinned_comment": clip_info.get("pinned_comment", "Drop your thoughts below!"),
                "best_post_time": clip_info.get("best_post_time", "12:00 PM - 2:00 PM"),
                "upload_instructions": clip_info.get("upload_instructions", "Post with original sound at 100%."),
                "start_time": clip_info.get("start_time"),
                "end_time": clip_info.get("end_time"),
                "duration": actual_dur,
                "video_url": f"/media/outputs/{job_id}/{out_filename}",
                "download_name": f"reel_{clip_num}_{int(actual_dur)}s.mp4"
            })

        results_map = {}
        completed_count = 0
        with ThreadPoolExecutor(max_workers=2) as render_pool:
            futures = [render_pool.submit(render_single_clip, (i, c)) for i, c in enumerate(clips_plan)]
            for fut in futures:
                idx, clip_dict = fut.result()
                results_map[idx] = clip_dict
                completed_count += 1
                progress_pct = 55 + int((completed_count / len(clips_plan)) * 40)
                update_job(
                    job_id,
                    status="processing",
                    message=f"Rendered {completed_count} of {len(clips_plan)} full-screen vertical reels...",
                    progress=progress_pct
                )

        rendered_clips = [results_map[i] for i in range(len(clips_plan))]

        guide_content = generate_upload_guide_text(rendered_clips, user_rules=user_rules)
        guide_path = os.path.join(job_out_dir, "upload_guide.txt")
        try:
            with open(guide_path, "w", encoding="utf-8") as gf:
                gf.write(guide_content)
        except Exception as ge:
            logger.warning(f"Failed to write upload guide: {ge}")

        update_job(
            job_id,
            status="completed",
            message=f"Successfully generated {len(rendered_clips)} vertical 9:16 reels from your rules!",
            progress=100,
            clips=rendered_clips,
            transcript=transcript,
            upload_guide_url=f"/media/outputs/{job_id}/upload_guide.txt",
            upload_guide_text=guide_content
        )

    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {e}")
        update_job(
            job_id,
            status="failed",
            message=f"Error: {str(e)}",
            progress=100,
            error=str(e)
        )

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AI Video Clipper Studio loading...</h1>"

@app.get("/api/system/status")
async def get_system_status():
    local_models = clipper_core.get_installed_ollama_models()
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    models_list = [
        {
            "id": "gemini-2.5-flash",
            "name": "Google Gemini 2.5 Flash",
            "badge": "Top Recommended • Multimodal",
            "provider": "gemini",
            "requires_key": not has_gemini_key,
            "has_env_key": has_gemini_key,
            "key_name": "Google Gemini API Key",
            "key_url": "https://aistudio.google.com/app/apikey"
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Google Gemini 1.5 Pro",
            "badge": "Deep Reasoning • 2M Context",
            "provider": "gemini",
            "requires_key": not has_gemini_key,
            "has_env_key": has_gemini_key,
            "key_name": "Google Gemini API Key",
            "key_url": "https://aistudio.google.com/app/apikey"
        },
        {
            "id": "gpt-4o",
            "name": "OpenAI GPT-4o",
            "badge": "Frontier Intelligence",
            "provider": "openai",
            "requires_key": not has_openai_key,
            "has_env_key": has_openai_key,
            "key_name": "OpenAI API Key",
            "key_url": "https://platform.openai.com/api-keys"
        },
        {
            "id": "deepseek-r1:8b",
            "name": "DeepSeek-R1 8B (Reasoning)",
            "badge": "Local • Chain-of-Thought Reasoning",
            "provider": "ollama",
            "requires_key": False,
            "has_env_key": False
        },
        {
            "id": "qwen2.5:7b",
            "name": "Qwen 2.5 7B",
            "badge": "Local • Fast Structured",
            "provider": "ollama",
            "requires_key": False,
            "has_env_key": False
        },
        {
            "id": "llama3.1",
            "name": "Llama 3.1 8B",
            "badge": "Local • Meta AI",
            "provider": "ollama",
            "requires_key": False,
            "has_env_key": False
        }
    ]

    return {
        "ollama_connected": len(local_models) > 0,
        "models": models_list,
        "recommended_model": "gemini-2.5-flash" if has_gemini_key else "deepseek-r1:8b",
        "has_gemini_key": has_gemini_key,
        "has_openai_key": has_openai_key,
        "whisper_model": "base"
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    saved_filename = f"{job_id}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {
        "job_id": job_id,
        "original_filename": file.filename,
        "video_path": saved_path,
        "video_url": f"/media/uploads/{saved_filename}",
        "status": "ready",
        "message": "Video uploaded. Ready to start clipping.",
        "progress": 0,
        "clips": [],
        "transcript": "",
        "segments": []
    }

    return {"job_id": job_id, "filename": file.filename}

@app.post("/api/start/{job_id}")
async def start_processing(
    job_id: str,
    user_rules: Optional[str] = Form(""),
    whisper_model: Optional[str] = Form("base"),
    ollama_model: Optional[str] = Form("deepseek-r1:8b"),
    api_key: Optional[str] = Form(None)
):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        return {"status": "already_running", "job_id": job_id}

    options = {
        "user_rules": (user_rules or "").strip(),
        "whisper_model": (whisper_model or "base").strip(),
        "ollama_model": (ollama_model or "deepseek-r1:8b").strip(),
        "api_key": (api_key or "").strip() or None,
        "add_captions": True,
        "caption_style": "center"
    }

    update_job(job_id, status="processing", message="Initializing AI video clipping from your rules...", progress=10)
    executor.submit(run_job_pipeline, job_id, job["video_path"], options)

    return {"status": "started", "job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    async def event_generator():
        last_progress = -1
        last_message = ""
        while True:
            if job_id in jobs:
                job = jobs[job_id]
                cur_progress = job.get("progress", 0)
                cur_msg = job.get("message", "")
                cur_status = job.get("status", "")

                if cur_progress != last_progress or cur_msg != last_message:
                    last_progress = cur_progress
                    last_message = cur_msg
                    yield {
                        "event": "update",
                        "data": {
                            "status": cur_status,
                            "progress": cur_progress,
                            "message": cur_msg,
                            "clips": job.get("clips", []),
                            "transcript": job.get("transcript", "")
                        }
                    }

                if cur_status in ["completed", "failed"]:
                    yield {
                        "event": "done",
                        "data": {
                            "status": cur_status,
                            "progress": cur_progress,
                            "message": cur_msg,
                            "clips": job.get("clips", []),
                            "error": job.get("error", "")
                        }
                    }
                    break

            await asyncio.sleep(0.8)

    return EventSourceResponse(event_generator())
