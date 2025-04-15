from fastapi import FastAPI, File, UploadFile
import openai
import shutil
import subprocess

app = FastAPI()
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

@app.post("/api/start-ai-guide")
async def start_ai_guide():
    """ AI gives real-time instructions while recording """
    prompt = "The user is recording a marketing video. Give real-time guidance on angles, lighting, and speaking style."
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"instructions": response["choices"][0]["message"]["content"]}

@app.post("/api/auto-edit")
async def auto_edit_video(video: UploadFile = File(...)):
    """ AI automatically edits the video """
    video_path = f"temp/{video.filename}"
    output_path = f"temp/edited_{video.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Apply AI-based video editing
    subprocess.run(["ffmpeg", "-i", video_path, "-vf", "eq=brightness=0.06:saturation=1.5", output_path])
    return {"editedVideoUrl": output_path}

@app.post("/api/generate-captions")
async def generate_captions(video_text: str):
    """ AI generates captions for the video """
    prompt = f"Generate engaging captions for this video: {video_text}"
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"captions": response["choices"][0]["message"]["content"]}

