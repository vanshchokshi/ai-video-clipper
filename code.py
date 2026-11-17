import os
import sys

# Ensure virtual environment binaries (such as ffmpeg) are available in PATH
_bin_dir = os.path.dirname(sys.executable)
if _bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

import clipper_core

def process_video_with_rules(video_path, user_rules="", model_name="llama3"):
    print(f"Transcribing video locally using Whisper...")
    transcript_data = clipper_core.transcribe_video_locally(video_path)
    transcript = transcript_data.get("text", "")

    print(f"Analyzing transcript & strictly applying rules with Ollama ({model_name})...")
    clips = clipper_core.analyze_transcript_with_ollama(
        transcript=transcript,
        model_name=model_name,
        user_rules=user_rules
    )

    print(f"Found {len(clips)} clips strictly following rules. Rendering 9:16 vertical reels...")
    for i, clip_info in enumerate(clips):
        output_file = f"clip_{i+1}.mp4"
        dur = int(clip_info.get("duration", 20))
        print(f"\nExporting {output_file} [{clip_info.get('start_time')}s -> {clip_info.get('end_time')}s] (Duration: {dur}s)...")
        print(f"Title: {clip_info.get('title')}")
        print(f"Hook Text: {clip_info.get('text_overlay')}")

        clipper_core.create_video_clip(
            video_path=video_path,
            clip_data=clip_info,
            output_filename=output_file,
            target_duration=dur,
            add_caption=True
        )
        print(f"Exported: {output_file}")

def main(video_path, user_rules=""):
    process_video_with_rules(video_path, user_rules=user_rules)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video = sys.argv[1]
        rules = sys.argv[2] if len(sys.argv) > 2 else ""
        main(video, user_rules=rules)
    else:
        print("Usage: python code.py <video_path> [rules_and_instructions]")
