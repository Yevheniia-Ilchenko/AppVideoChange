import streamlit as st
import os


# def main():
st.title("🎬✂️AppVideoChange✂️🎬")
st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
st.number_input("Number of clips", min_value=1, step=1)
st.text_input("Prompt for audio")
st.number_input("Clip index for audio", min_value=1, step=1)
st.number_input("Number of columns to display clips", min_value=1, step=1)
