import typing as T

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
import os
import tempfile
import zipfile
from moviepy.editor import VideoFileClip, AudioFileClip


def generate_audio(prompt, duration, device, params, seed):
    image = streamlit_util.run_txt2img(
        prompt=prompt,
        num_inference_steps=30,
        guidance=7.0,
        negative_prompt="",
        seed=seed,
        width=512,
        height=512,
        checkpoint="riffusion/riffusion-model-v1",
        device=device,
        scheduler="DPMSolverMultistepScheduler",
    )

    segment = streamlit_util.audio_segment_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
    )

    audio_path = os.path.join(tempfile.gettempdir(),
                              f"{prompt.replace(' ', '_')}_{seed}.mp3")
    segment.export(audio_path, format="mp3")
    return audio_path


def split_video_into_clips(video_path, num_clips):
    video = VideoFileClip(video_path)
    clip_duration = video.duration / num_clips
    clips = [video.subclip(i * clip_duration, (i + 1) * clip_duration)
             for i in range(num_clips)]
    return clips


def process_video(
        uploaded_file,
        num_clips,
        prompt,
        target_clip_index,
        num_columns,
        device,
        params,
        seed
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    clips = split_video_into_clips(video_path, num_clips)

    duration = clips[target_clip_index].duration
    audio_clip_path = generate_audio(prompt, duration, device, params, seed)

    audio = AudioFileClip(audio_clip_path)
    clips[target_clip_index] = clips[target_clip_index].set_audio(audio)

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

device = streamlit_util.select_device(st.sidebar)
extension = streamlit_util.select_audio_extension(st.sidebar)
checkpoint = streamlit_util.select_checkpoint(st.sidebar)

if not prompt:
    st.info("Enter a prompt")

use_20k = st.sidebar.checkbox("Use 20kHz", value=False)

if use_20k:
    params = SpectrogramParams(
        min_frequency=10,
        max_frequency=20000,
        sample_rate=44100,
        stereo=True,
    )
else:
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
        stereo=False,
    )

starting_seed = T.cast(
    int,
    st.sidebar.number_input(
        "Seed",
        value=42,
        help="Change this to generate different variations",
    ),
)

if st.button("Process Video"):
    if video_file is not None:
        clip_paths = process_video(
            video_file, num_clips,
            prompt,
            clip_index - 1,
            num_columns,
            device,
            params,
            starting_seed
        )

        zip_path = create_zip_file(clip_paths)
        st.success("Video processing completed!")
        st.download_button("Download Clips",
                           data=open(zip_path, "rb"),
                           file_name="clips.zip")

        cols = st.columns(num_columns)
        for i, clip_path in enumerate(clip_paths):
            cols[i % num_columns].video(clip_path)
