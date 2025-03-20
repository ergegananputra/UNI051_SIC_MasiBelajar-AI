import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from app.models import SafezoneModel

# Initialize the SafezoneModel
safezone_model = SafezoneModel()

# Set Streamlit page configuration to wide layout
st.set_page_config(
    page_title="AI Video Processing",
    layout="wide",  # Use wide layout
    initial_sidebar_state="collapsed"  # Collapse the sidebar by default
)

st.title("🎥 AI Video Processing with Safezone Detection")

# Add a custom header and description for the file uploader
st.markdown(
    """
    ## 📂 Upload Your Video
    Drag and drop your video file here or click to browse. Supported formats: **MP4, AVI, MOV**.
    """
)

# File uploader with drag-and-drop functionality
video_file = st.file_uploader(
    label="Upload video",
    type=["mp4", "avi", "mov"],
    label_visibility="collapsed"  # Hide the default label
)

if video_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())

    # Get video dimensions for default safezone points
    cap = cv2.VideoCapture(temp_video.name)
    if not cap.isOpened():
        st.error("❌ Error: Unable to open the video file.")
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Define default safezone points (rectangle)
        default_points = [
            [frame_width // 4, frame_height // 4],  # Top-left
            [3 * frame_width // 4, frame_height // 4],  # Top-right
            [3 * frame_width // 4, 3 * frame_height // 4],  # Bottom-right
            [frame_width // 4, 3 * frame_height // 4]  # Bottom-left
        ]

        st.subheader("🎯 Adjust Safezone")

        # Allow users to dynamically adjust the number of points
        num_points = st.number_input(
            "Number of Points (Minimum 3)", 
            min_value=3, 
            max_value=10, 
            value=len(default_points), 
            step=1, 
            key="num_points"
        )

        # Create side-by-side layout for video and safezone inputs
        col1, col2 = st.columns([3, 2])  # Adjust column width ratio

        with col1:
            frame_placeholder = st.empty()  # Placeholder for video frames

        points = []
        with col2:
            for i in range(num_points):
                st.markdown(f"### Point {i+1}")
                col_x, col_y = st.columns(2)
                with col_x:
                    x = st.number_input(
                        f"Posisi X (Point {i+1})", 
                        min_value=0, 
                        max_value=frame_width, 
                        value=default_points[i % len(default_points)][0], 
                        step=1, 
                        key=f"x{i}"
                    )
                with col_y:
                    y = st.number_input(
                        f"Posisi Y (Point {i+1})", 
                        min_value=0, 
                        max_value=frame_height, 
                        value=default_points[i % len(default_points)][1], 
                        step=1, 
                        key=f"y{i}"
                    )
                points.append([x, y])

        # Process the video using SafezoneModel
        st.write("⏳ Processing video...")

        cap = cv2.VideoCapture(temp_video.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Pass the frame to the SafezoneModel for object detection
            processed_frame = safezone_model.process_frame(frame, points)

            # Draw the safezone polygon on the processed frame
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(processed_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Overlay the coordinates of each point
            for i, (x, y) in enumerate(points):
                # Draw a small circle at each point
                cv2.circle(processed_frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                # Add the coordinate text near the point
                cv2.putText(
                    processed_frame,
                    f"({int(x)}, {int(y)})",
                    (int(x) + 10, int(y) - 10),  # Offset the text slightly from the point
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1
                )

            # Convert the frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("✅ Video processing complete!")