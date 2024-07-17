import streamlit as st
import os
import tempfile
import zipfile
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from riffusion_text import RiffusionPipeline

def generate_audio(prompt, duration):
    pipeline = RiffusionPipeline()
    output = pipeline.generate_audio_from_prompt(prompt, duration)
    audio_clip_path = output.audios[0]
    return audio_clip_path

def split_video_into_clips(video_path, num_clips):
    video = VideoFileClip(video_path)
    clip_duration = video.duration / num_clips
    clips = [video.subclip(i * clip_duration, (i + 1) * clip_duration) for i in range(num_clips)]

    clip_paths = []
    for i, clip in enumerate(clips):
        clip_path = f"{tempfile.gettempdir()}/clip_{i}.mp4"
        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
        clip_paths.append(clip_path)

    return clip_paths

def process_video(uploaded_file, num_clips, prompt, target_clip_index, num_columns):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    clips = split_video_into_clips(video_path, num_clips)

    duration = VideoFileClip(clips[target_clip_index]).duration
    audio_clip_path = generate_audio(prompt, duration)

    target_clip = VideoFileClip(clips[target_clip_index]).set_audio(AudioFileClip(audio_clip_path))
    clips[target_clip_index] = target_clip

    clip_paths = []
    for i, clip in enumerate(clips):
        clip_path = f"{tempfile.gettempdir()}/clip_{i}.mp4"
        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
        clip_paths.append(clip_path)

    return clip_paths

def create_zip_file(clip_paths):
    zip_path = f"{tempfile.gettempdir()}/clips.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for clip_path in clip_paths:
            zipf.write(clip_path, os.path.basename(clip_path))
    return zip_path


st.title("🎬✂️AppVideoChange✂️🎬")
video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
num_clips = st.number_input("Number of clips", min_value=1, step=1)
prompt = st.text_input("Prompt for audio")
clip_index = st.number_input("Clip index for audio", min_value=1, step=1)
num_columns = st.number_input("Number of columns to display clips", min_value=1, step=1)

if st.button("Process Video"):
    if video_file is not None:
        clip_paths = process_video(video_file, num_clips, prompt, clip_index - 1, num_columns)

        zip_path = create_zip_file(clip_paths)

        st.success("Video processing completed!")
        st.download_button("Download Clips", data=open(zip_path, "rb"), file_name="clips.zip")

        cols = st.columns(num_columns)
        for i, clip_path in enumerate(clip_paths):
            cols[i % num_columns].video(clip_path)
