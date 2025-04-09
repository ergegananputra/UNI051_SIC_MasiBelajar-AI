import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from app.models import  MasiBelajarModel

# Initialize the SafezoneModel
masibelajar_model = MasiBelajarModel(
    od_weight='app/models/object_detection/config/best.pt',
    pose_weight='app/models/key_points/config/yolo11m-pose.pt',
    tracker='app/models/tracker/tracker.yaml',
)

# Set Streamlit page configuration to wide layout
st.set_page_config(
    page_title="AI Video Processing",
    layout="wide",  # Use wide layout
    initial_sidebar_state="collapsed"  # Collapse the sidebar by default
)

st.title("🎥 Lokari")

# Add a custom header and description
st.markdown(
    """
    ## 📂 Upload Your Video or Enter a Stream URL
    - Drag and drop your video file here or click to browse. Supported formats: **MP4, AVI, MOV**.
    - Alternatively, enter a **stream URL** (e.g., RTSP or HTTP) to process a live video stream.
    """
)

# Add a tab layout for "File Upload" and "Stream URL"
tab1, tab2 = st.tabs(["📂 File Upload", "🌐 Stream URL"])

# File Upload Tab
with tab1:
    video_file = st.file_uploader(
        label="Upload video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

# Stream URL Tab
with tab2:
    stream_url = st.text_input(
        label="Enter Stream URL",
        placeholder="e.g., rtsp://username:password@ip_address:port/stream",
    )

# Process the video or stream
if video_file or stream_url:
    if video_file:
        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        video_source = temp_video.name
    else:
        # Use the stream URL as the video source
        video_source = stream_url

    # Get video dimensions for default safezone points
    cap = cv2.VideoCapture(video_source)
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

        # Initialize session state for points and num_points
        if "points" not in st.session_state:
            st.session_state.points = default_points
        if "num_points" not in st.session_state:
            st.session_state.num_points = len(default_points)

        st.subheader("🎯 Adjust Safezone")


        # Create side-by-side layout for video and safezone inputs
        col1, col2 = st.columns([3, 2])  # Adjust column width ratio

        with col1:
            available_classes = ['toddler', 'non-toddler']  # Add more classes if needed
            selected_classes = st.multiselect(
                "Select the target classes to monitor:",
                options=available_classes,
                default=['toddler', 'non-toddler']  # Default selected classes
            )
            
            frame_placeholder = st.empty()  # Placeholder for video frames

            message_placeholder = st.empty()  # Placeholder for messages
    
        with col2:
            # Allow users to input points manually via a text area
            st.markdown("### 📋 Copy-Paste Points")
            points_input = st.text_area(
                "Enter points as a list of [x, y] pairs (e.g., [[100, 200], [300, 400], [500, 600]])",
                value=str(st.session_state.points),
                height=100,
            )

            # Parse the input points
            try:
                parsed_points = eval(points_input)  # Convert string to list
                if not all(isinstance(point, list) and len(point) == 2 for point in parsed_points):
                    raise ValueError("Invalid format")
                
                num_points = len(parsed_points)  # Update num_points based on the input
                if num_points < 3:
                    raise ValueError("At least 3 points are required.")
                
                st.session_state.points = parsed_points  # Update session state with parsed points
                num_points_textfield = num_points  # Update the number of points in the text field
            except Exception as e:
                st.error(f"❌ Invalid points format. Please enter a list of [x, y] pairs. {e}")
                num_points_textfield = len(st.session_state.points)  # Reset to the default number of points

            def update_points():
                # Update the session state with the new number of points
                num_points = st.session_state.num_points
                if num_points > len(st.session_state.points):
                    # Add new points if the number has increased
                    st.session_state.points.extend([[0, 0]] * (num_points - len(st.session_state.points)))
                elif num_points < len(st.session_state.points):
                    # Remove excess points if the number has decreased
                    st.session_state.points = st.session_state.points[:num_points]
                # Update the points in the text area
                st.session_state.points = st.session_state.points[:num_points]
                st.session_state.num_points = num_points  # Update the session state with the new number of points

            # Allow users to dynamically adjust the number of points
            num_points_textfield = st.number_input(
                "Number of Points (Minimum 3)", 
                min_value=3, 
                max_value=10, 
                value=len(st.session_state.points),  # Use the session state value
                step=1, 
                key="num_points",
                on_change=lambda: update_points()  # Update points when the number changes
            )

            # Display number inputs for each point
            st.markdown("### Adjust Points")
            for i in range(num_points):
                st.markdown(f"### Point {i+1}")
                col_x, col_y = st.columns(2)
                with col_x:
                    x = st.number_input(
                        f"Posisi X (Point {i+1})", 
                        min_value=0, 
                        max_value=frame_width, 
                        value=st.session_state.points[i][0],  # Use the parsed points
                        step=1, 
                        key=f"x{i}"
                    )
                with col_y:
                    y = st.number_input(
                        f"Posisi Y (Point {i+1})", 
                        min_value=0, 
                        max_value=frame_height, 
                        value=st.session_state.points[i][1],  # Use the parsed points
                        step=1, 
                        key=f"y{i}"
                    )
                st.session_state.points[i] = [x, y]  # Update the points list dynamically

        
        # Process the video using SafezoneModel
        st.write("⏳ Processing video...")

        for result, frame in masibelajar_model.analyze_frame(
            inference_path=video_source,
            safezone_points=st.session_state.points,
            target_class=selected_classes,
            preview=True,
            stream=True,
            verbose=True,
            track=True,
        ):
            pts = np.array(st.session_state.points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Overlay the coordinates of each point
            for i, (x, y) in enumerate(st.session_state.points):
                # Draw a small circle at each point
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                # Add the coordinate text near the point
                cv2.putText(
                    frame,
                    f"({int(x)}, {int(y)})",
                    (int(x) + 10, int(y) - 10),  # Offset the text slightly from the point
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1
                )

            message_placeholder.json(result)

            frame_placeholder.image(
                frame, 
                channels="BGR",
                use_container_width=True
            )

        st.success("✅ Video processing complete!")