import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Tidak bisa membuka video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    frame_time = 1.0 / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    default_points = [
        [frame_width // 4, frame_height // 4],
        [3 * frame_width // 4, frame_height // 4],
        [frame_width - 50, frame_height // 2],
        [3 * frame_width // 4, 3 * frame_height // 4], 
        [frame_width // 4, 3 * frame_height // 4],
        [50, frame_height // 2] 
    ]

    frame_placeholder = st.empty()

    st.sidebar.subheader("🎯 Atur Bounding Box")

    previous_points = default_points.copy()
    points = []
    
    for i in range(6):
        x = st.sidebar.slider(f"Poin {i+1} - Posisi X", 0, frame_width, default_points[i][0], key=f"x{i}")
        y = st.sidebar.slider(f"Poin {i+1} - Posisi Y", 0, frame_height, default_points[i][1], key=f"y{i}")
        points.append([x, y])
    print("koordinat",points)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        out.write(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")

        time.sleep(frame_time)

    cap.release()
    out.release()

st.title("AI Video Processing")

video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if video_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())
    
    output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")

    process_video(temp_video.name, output_video_path)

    st.sidebar.subheader("📥 Unduh Video")
    with open(output_video_path, "rb") as file:
        st.download_button(
            label="Download Hasil Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
