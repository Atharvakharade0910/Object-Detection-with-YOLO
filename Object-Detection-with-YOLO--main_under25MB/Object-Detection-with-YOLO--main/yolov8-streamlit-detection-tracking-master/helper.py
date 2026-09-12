from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import numpy as np
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib in streamlit
from PIL import Image


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def get_yolo_classes(model):
    """
    Get the class names from the YOLO model and add more common objects for detection.
    
    Parameters:
        model: A YOLO object detection model.
        
    Returns:
        A dictionary of class indices and names.
    """
    # Get original class names
    original_classes = model.names
    
    # We can't modify the model's detection capabilities, but we can help users
    # see all available classes more clearly by providing extra common objects
    # that might be in the pretrained model but with different naming
    common_names = {
        # People and living beings
        'person': 'person',
        'pedestrian': 'person',
        'child': 'person',
        'adult': 'person',
        'dog': 'dog',
        'cat': 'cat',
        'bird': 'bird',
        'horse': 'horse',
        'sheep': 'sheep',
        'cow': 'cow',
        
        # Vehicles
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'bicycle': 'bicycle',
        'motorcycle': 'motorcycle',
        'airplane': 'airplane',
        'train': 'train',
        'boat': 'boat',
        
        # Common objects
        'backpack': 'backpack',
        'umbrella': 'umbrella',
        'handbag': 'handbag',
        'tie': 'tie',
        'suitcase': 'suitcase',
        'bottle': 'bottle',
        'cup': 'cup',
        'bowl': 'bowl',
        'chair': 'chair',
        'couch': 'couch',
        'bed': 'bed',
        'table': 'dining table',
        'toilet': 'toilet',
        'tv': 'tv',
        'laptop': 'laptop',
        'mouse': 'mouse',
        'keyboard': 'keyboard',
        'cell phone': 'cell phone',
        'microwave': 'microwave',
        'oven': 'oven',
        'toaster': 'toaster',
        'refrigerator': 'refrigerator',
        'book': 'book',
        'clock': 'clock',
        'vase': 'vase',
        'scissors': 'scissors',
        'teddy bear': 'teddy bear',
        'hair drier': 'hair drier',
        
        # Traffic objects
        'traffic light': 'traffic light',
        'stop sign': 'stop sign',
        'fire hydrant': 'fire hydrant',
        'parking meter': 'parking meter',
        
        # Sports equipment
        'baseball bat': 'baseball bat',
        'baseball glove': 'baseball glove',
        'skateboard': 'skateboard',
        'surfboard': 'surfboard',
        'tennis racket': 'tennis racket',
        'sports ball': 'sports ball',
        'kite': 'kite',
        'skis': 'skis',
        'snowboard': 'snowboard',
        
        # Food items
        'banana': 'banana',
        'apple': 'apple',
        'sandwich': 'sandwich',
        'orange': 'orange',
        'broccoli': 'broccoli',
        'carrot': 'carrot',
        'hot dog': 'hot dog',
        'pizza': 'pizza',
        'donut': 'donut',
        'cake': 'cake',
    }
    
    # The standard model already has these classes, so this is just a reference
    # to keep the original mapping intact while making selection easier
    return original_classes


def display_tracker_options():
    """Always use ByteTrack as the default tracker."""
    # No longer ask for tracker selection, always return True and bytetrack
    return True, "bytetrack.yaml"


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=True, tracker="bytetrack.yaml", selected_classes=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=True).
    - tracker (str): The tracker type to use (default="bytetrack.yaml").
    - selected_classes (list): List of class indices to detect. If None, all classes are detected.

    Returns:
    dict: Dictionary of class counts
    """
    if image is None:
        return {}, None

    # Resize the image maintaining aspect ratio but with a maximum width
    # This helps improve performance without losing too much quality
    MAX_WIDTH = 720
    h, w = image.shape[:2]
    if w > MAX_WIDTH:
        ratio = MAX_WIDTH / w
        dim = (MAX_WIDTH, int(h * ratio))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Always use tracking for better performance and consistency
    res = model.track(image, conf=conf, persist=True, tracker="bytetrack.yaml", classes=selected_classes)

    # Get detection results to display as text
    class_counts = {}
    if len(res) > 0 and len(res[0].boxes) > 0:
        boxes = res[0].boxes
        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )

    # Return class counts for graph updating
    return class_counts, res_plotted


def save_upload_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    if uploaded_file is None:
        return None
        
    # Create a temporary file to save the uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name
    
    return file_path


def save_detected_image(image, filename):
    """Save a detected image to a file and return the path."""
    if image is None:
        return None
    
    # Convert BGR to RGB if necessary 
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if the image is BGR (OpenCV format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Create a PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Create a temp file to save the image
    output_path = os.path.join(tempfile.gettempdir(), filename)
    pil_image.save(output_path)
    
    return output_path


def create_video_from_frames(frames, fps=30, output_filename="detected_video.mp4"):
    """
    Create a video file from a list of frames.
    
    Args:
    - frames: List of OpenCV image frames
    - fps: Frames per second
    - output_filename: Name of the output video file
    
    Returns:
    - String: Path to the created video file
    """
    if not frames:
        return None
    
    # Get video dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Create a temporary file path for the video
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in frames:
        out.write(frame)
    
    # Release the video writer
    out.release()
    
    return output_path


def create_detection_report(task_type, model_name, conf, source_type, class_counts_over_time, df=None, media_path=None):
    """
    Create a PDF report of detection results.
    
    Args:
    - task_type (str): Detection or Segmentation
    - model_name (str): Name of the model used
    - conf (float): Confidence threshold
    - source_type (str): Image, Video, Webcam
    - class_counts_over_time (dict): Dictionary with time as key and class counts as values
    - df (DataFrame): Optional dataframe with detection data for plotting
    - media_path (str): Path to the detected media file to include in report

    Returns:
    - BytesIO: PDF report as a BytesIO object
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Detection Statistics Report", styles['Title'])
    story.append(title)
    
    # Generated date
    date_info = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    story.append(date_info)
    story.append(Spacer(1, 12))
    
    # Prepare data for report
    if df is not None and not df.empty:
        # For video and webcam, use the maximum values for each class
        class_counts = {}
        for col in df.columns[1:]:  # Skip the Time column
            class_counts[col] = int(df[col].max())
    else:
        # For images, use the direct counts
        class_counts = class_counts_over_time
    
    # Calculate total objects and unique types
    total_objects = sum(class_counts.values())
    unique_types = len(class_counts)
    
    # Create summary table
    summary_data = [
        ["Total Objects Detected:", str(total_objects)],
        ["Unique Object Types:", str(unique_types)]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # Create chart for detected objects
    if class_counts:
        # Power BI style colors - vibrant and high contrast
        power_bi_colors = [
            '#01B8AA', '#374649', '#FD625E', '#F2C80F', 
            '#5F6B6D', '#8AD4EB', '#FE9666', '#A66999',
            '#3599B8', '#DFBFBF', '#4AC5BB', '#5F6B6D'
        ]
        
        # Set the style to be more like Power BI
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(6, 3.5))
        
        if df is not None and not df.empty and len(df) > 1:
            # Use time series data for video/webcam
            for i, column in enumerate(df.columns[1:]):
                color_idx = i % len(power_bi_colors)
                plt.plot(df['Time'], df[column], marker='', 
                         linewidth=3, label=column, 
                         color=power_bi_colors[color_idx])
            plt.xlabel('Time (seconds)', fontweight='bold')
            plt.ylabel('Count', fontweight='bold')
        else:
            # For images, create a line connecting the object counts
            objects = list(class_counts.keys())
            counts = [class_counts[obj] for obj in objects]
            
            # For single points, add a line graph visualization
            x_range = range(len(objects))
            for i, (x, y) in enumerate(zip(x_range, counts)):
                color_idx = i % len(power_bi_colors)
                plt.plot([x], [y], marker='o', markersize=10, 
                         color=power_bi_colors[color_idx])
                
            # Add connecting lines with gradient alpha
            if len(objects) > 1:
                for i in range(len(objects)-1):
                    idx1 = i % len(power_bi_colors)
                    idx2 = (i+1) % len(power_bi_colors)
                    plt.plot(x_range[i:i+2], counts[i:i+2], 
                             linestyle='-', linewidth=2.5, 
                             color=power_bi_colors[idx1], alpha=0.7)
            
            plt.xticks(x_range, objects, rotation=45, fontweight='bold')
            plt.xlabel('Object Type', fontweight='bold')
            plt.ylabel('Count', fontweight='bold')
        
        # Enhance the visual appearance
        plt.title('Object Detection Counts', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add a light gray background and white frame
        ax = plt.gca()
        ax.set_facecolor('#f5f5f5')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#FFFFFF')
        
        # Improve legend
        if df is not None and not df.empty and len(df) > 1:
            legend = plt.legend(frameon=True, facecolor='white', 
                     edgecolor='lightgray', fontsize=9, loc='best')
        
        # Save plot to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Add plot to report
        rl_img = RLImage(img_buffer, width=400, height=220)
        story.append(rl_img)
        story.append(Spacer(1, 15))
    
    # Create detailed table of detected objects
    if class_counts:
        data = [["Object Type", "Count"]]
        for obj, count in class_counts.items():
            data.append([obj, str(count)])
        
        table = Table(data, colWidths=[300, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    
    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer, media_path


def generate_download_link(buffer, filename, text):
    """Generate a download link for a file."""
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href


def generate_video_download_link(video_path, text):
    """Generate a download link for a video file."""
    if not video_path or not os.path.exists(video_path):
        return ""
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{os.path.basename(video_path)}">{text}</a>'
    return href


def init_detection_graph():
    """Initialize a matplotlib figure for real-time plotting."""
    # Set the style to be more like Power BI
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Enhanced styling
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Object Detection Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a light gray background and white frame
    ax.set_facecolor('#f5f5f5')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#FFFFFF')
        
    return fig, ax


def update_detection_graph(fig, ax, df):
    """Update the detection graph with new data."""
    # Power BI style colors - vibrant and high contrast
    power_bi_colors = [
        '#01B8AA', '#374649', '#FD625E', '#F2C80F', 
        '#5F6B6D', '#8AD4EB', '#FE9666', '#A66999',
        '#3599B8', '#DFBFBF', '#4AC5BB', '#5F6B6D'
    ]
    
    ax.clear()
    
    # Plot each class with a different vibrant color
    for i, column in enumerate(df.columns[1:]):  # Skip time column
        color_idx = i % len(power_bi_colors)
        ax.plot(df['Time'], df[column], marker='', 
                linewidth=3, label=column, 
                color=power_bi_colors[color_idx])
    
    # Enhance the visual appearance
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Object Detection Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a light gray background and white frame
    ax.set_facecolor('#f5f5f5')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#FFFFFF')
    
    # Improve legend
    if len(df.columns[1:]) > 0:
        legend = ax.legend(frameon=True, facecolor='white', 
                 edgecolor='lightgray', fontsize=9, loc='best')
    
    return fig


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = 0  # Use default camera (change to a number if you have multiple cameras)
    
    # Get class names and create multiselect
    class_names = get_yolo_classes(model)
    class_list = list(class_names.values())
    selected_classes_names = st.sidebar.multiselect("Select objects to detect", class_list)
    
    # Convert selected class names to indices
    selected_classes = None
    if selected_classes_names:
        selected_classes = [list(class_names.keys())[list(class_names.values()).index(name)] 
                           for name in selected_classes_names]
                           
    # Create placeholder for video display, graph and results
    st_frame = st.empty()
    st_graph = st.empty()
    results_container = st.container()
    
    # Initialize dataframe to store detection data for graph
    if "detection_data_webcam" not in st.session_state:
        st.session_state.detection_data_webcam = pd.DataFrame(columns=['Time'])
        st.session_state.webcam_detected_frames = []
    
    # Initialize detection start time
    if "detection_start_time_webcam" not in st.session_state:
        st.session_state.detection_start_time_webcam = None
    
    # Create a placeholder for the report link
    report_link_placeholder = st.empty()
    
    # Always use tracking (ByteTrack)
    is_display_tracker, tracker = True, "bytetrack.yaml"
    
    col1, col2 = st.columns(2)
    with col1:
        detect_button = st.button('Detect Objects (Webcam)')
    with col2:
        save_button = st.button('Save Results (Webcam)')
    
    if detect_button:
        # Reset data and start time
        st.session_state.detection_data_webcam = pd.DataFrame(columns=['Time'])
        st.session_state.detection_start_time_webcam = time.time()
        st.session_state.webcam_detected_frames = []
        
        # Try opening the webcam, with a more robust approach
        try:
            # Try multiple backends/sources if needed
            for src in [0, 1, 2, cv2.CAP_DSHOW, cv2.CAP_ANY]:
                vid_cap = cv2.VideoCapture(src)
                if vid_cap.isOpened():
                    st.info(f"Webcam successfully opened (source: {src})")
                    break
                vid_cap.release()
            
            if not vid_cap.isOpened():
                st.error("Could not open webcam. Please check your webcam connection.")
                return
            
            # Get camera parameters
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
            if original_fps == 0:  # If can't get FPS, use default
                original_fps = 30
                
            # Processing settings for smoother experience
            target_fps = 15  # Target processing FPS
            process_every_n_frames = max(1, int(original_fps / target_fps))
            
            # Create figure and axis for the graph
            fig, ax = init_detection_graph()
            
            # For tracking total counts
            cumulative_class_counts = {}
            
            # Show a status message
            status_msg = st.empty()
            status_msg.info("Webcam is active. Processing frames...")
            
            frame_count = 0
            processed_count = 0
            last_update_time = time.time()
            graph_update_interval = 0.5  # Update graph every 0.5 seconds instead of every frame
            last_frame_time = time.time()
            
            # Set lower resolution for processing if camera is high-res
            if width > 640:
                vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            while vid_cap.isOpened():
                # Control frame rate to avoid overwhelming the CPU
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Limit capture rate
                if elapsed < 1.0/target_fps:
                    time.sleep(max(0, (1.0/target_fps) - elapsed))
                
                # Read frame
                success, image = vid_cap.read()
                last_frame_time = time.time()
                
                if success:
                    frame_count += 1
                    
                    # Calculate time elapsed
                    time_elapsed = current_time - st.session_state.detection_start_time_webcam
                    
                    # Only process every n frames to maintain framerate
                    if frame_count % process_every_n_frames == 0:
                        processed_count += 1
                        
                        # Display detected frames
                        class_counts, detected_frame = _display_detected_frames(
                            conf, model, st_frame, image,
                            is_display_tracker, tracker, selected_classes
                        )
                        
                        # Store the frame for video creation
                        if processed_count % 2 == 0 and len(st.session_state.webcam_detected_frames) < 300:
                            st.session_state.webcam_detected_frames.append(detected_frame)
                        
                        # Update cumulative counts
                        for cls, count in class_counts.items():
                            if cls in cumulative_class_counts:
                                cumulative_class_counts[cls] = max(cumulative_class_counts[cls], count)
                            else:
                                cumulative_class_counts[cls] = count
                        
                        # Update dataframe with detection data (less frequently to reduce overhead)
                        # Only update graph data periodically to improve performance
                        current_update_time = time.time()
                        if current_update_time - last_update_time > graph_update_interval:
                            new_row = {'Time': time_elapsed}
                            for cls, count in class_counts.items():
                                if cls not in st.session_state.detection_data_webcam.columns:
                                    st.session_state.detection_data_webcam[cls] = 0
                                new_row[cls] = count
                            
                            # Fill missing classes with 0
                            for col in st.session_state.detection_data_webcam.columns:
                                if col != 'Time' and col not in new_row:
                                    new_row[col] = 0
                                    
                            # Append the new row
                            st.session_state.detection_data_webcam = pd.concat([
                                st.session_state.detection_data_webcam, 
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                            
                            # Update the graph
                            fig = update_detection_graph(fig, ax, st.session_state.detection_data_webcam)
                            st_graph.pyplot(fig)
                            last_update_time = current_update_time
                    
                    # Check if stop button is pressed or enough frames captured
                    if not detect_button or len(st.session_state.webcam_detected_frames) >= 300:
                        break
                else:
                    # Retry a few times before giving up
                    time.sleep(0.1)
                    if frame_count > 10:  # If we've been running for a while and now lost connection
                        break
            
            # Clean up
            vid_cap.release()
            status_msg.success("Webcam detection completed!")
            
            # Display final graph
            if len(st.session_state.detection_data_webcam) > 0:
                fig = update_detection_graph(fig, ax, st.session_state.detection_data_webcam)
                st_graph.pyplot(fig)
                
            # Display summary under the video/graph
            with results_container:
                st.markdown("### Detection Results")
                detection_text = ", ".join([f"{name}: {count}" for name, count in cumulative_class_counts.items()])
                if detection_text:
                    st.write(detection_text)
                else:
                    st.write("No objects detected")
                
                # Display detailed results
                if cumulative_class_counts:
                    with st.expander("View Detailed Results"):
                        # Create a DataFrame from the results
                        results_df = pd.DataFrame(list(cumulative_class_counts.items()), columns=['Object', 'Count'])
                        st.dataframe(results_df)
            
        except Exception as e:
            st.error(f"Error in webcam detection: {str(e)}")
            
    if save_button and hasattr(st.session_state, "webcam_detected_frames") and len(st.session_state.webcam_detected_frames) > 0:
        try:
            # Show processing message
            saving_status = st.empty()
            saving_status.info("Creating video file and generating report...")
            
            # Create video from captured frames
            video_path = create_video_from_frames(
                st.session_state.webcam_detected_frames, 
                output_filename=f"webcam_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            
            # Generate report
            buffer, _ = create_detection_report(
                task_type="Detection",
                model_name=str(model.model.names),
                conf=conf,
                source_type="Webcam",
                class_counts_over_time={},  # Will use dataframe instead
                df=st.session_state.detection_data_webcam,
                media_path=video_path
            )
            
            # Create download links
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # PDF report link
            report_link = generate_download_link(
                buffer,
                f"webcam_detection_report_{now}.pdf",
                "Download Detection Report"
            )
            
            # Video link
            video_link = generate_video_download_link(
                video_path,
                "Download Detected Video"
            )
            
            # Display links
            saving_status.success("Video and report ready for download!")
            report_link_placeholder.markdown(
                f"{report_link}<br>{video_link}", 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Error saving webcam results: {str(e)}")
    elif save_button and (not hasattr(st.session_state, "webcam_detected_frames") or len(st.session_state.webcam_detected_frames) == 0):
        st.warning("No webcam data to save. Please run detection first.")


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    # Allow user to upload video
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])
    
    # Get class names and create multiselect
    class_names = get_yolo_classes(model)
    class_list = list(class_names.values())
    selected_classes_names = st.sidebar.multiselect("Select objects to detect", class_list)
    
    # Convert selected class names to indices
    selected_classes = None
    if selected_classes_names:
        selected_classes = [list(class_names.keys())[list(class_names.values()).index(name)] 
                           for name in selected_classes_names]
    
    # Always use tracking (ByteTrack)
    is_display_tracker, tracker = True, "bytetrack.yaml"

    # Create placeholder for video display, graph and results
    st_frame = st.empty()
    st_graph = st.empty()
    results_container = st.container()
    
    # Initialize dataframe to store detection data for graph
    if "detection_data_video" not in st.session_state:
        st.session_state.detection_data_video = pd.DataFrame(columns=['Time'])
        st.session_state.video_detected_frames = []
    
    # Initialize detection start time
    if "detection_start_time_video" not in st.session_state:
        st.session_state.detection_start_time_video = None
    
    # Create a placeholder for the report link
    report_link_placeholder = st.empty()

    video_path = None
    if uploaded_video is not None:
        # Create a temporary file to save the uploaded video
        video_path = save_upload_file(uploaded_video)
        
        col1, col2 = st.columns(2)
        with col1:
            detect_button = st.button('Detect Video Objects')
        with col2:
            save_button = st.button('Save Results (Video)')
            
        if detect_button:
            # Reset data and start time
            st.session_state.detection_data_video = pd.DataFrame(columns=['Time'])
            st.session_state.detection_start_time_video = time.time()
            st.session_state.video_detected_frames = []
            
            try:
                vid_cap = cv2.VideoCapture(video_path)
                
                if not vid_cap.isOpened():
                    st.error("Error opening video file")
                    return
                
                # Get video info
                fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                if fps == 0:  # If can't get FPS, use default
                    fps = 30
                frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                
                # Processing settings for smoother experience
                target_fps = 15  # Target processing FPS
                process_every_n_frames = max(1, int(fps / target_fps))
                video_sampling_rate = min(1000, int(frame_count / 100))  # Sample up to 100 data points for the graph
                
                # Create progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create figure and axis for the graph
                fig, ax = init_detection_graph()
                
                # For tracking total counts
                cumulative_class_counts = {}
                frame_index = 0
                processed_count = 0
                last_update_time = time.time()
                graph_update_interval = 1.0  # seconds
                
                # Prepare a buffer of frames to process
                frames_buffer = []
                max_buffer_size = 5
                
                # Process video frames
                st.info("Processing video frames for detection...")
                
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        frame_index += 1
                        
                        # Calculate progress
                        progress = int(frame_index / frame_count * 100)
                        if frame_index % 30 == 0:  # Update progress bar periodically
                            progress_bar.progress(progress)
                            status_text.text(f"Processing video: {progress}% complete")
                        
                        # Process only selected frames to maintain desired frame rate
                        if frame_index % process_every_n_frames == 0:
                            processed_count += 1
                            current_time = frame_index / fps
                            
                            # Display detected frames
                            class_counts, detected_frame = _display_detected_frames(
                                conf, model, st_frame, image,
                                is_display_tracker, tracker, selected_classes
                            )
                            
                            # Store frame for video creation
                            if processed_count % 2 == 0:  # Store every other processed frame
                                st.session_state.video_detected_frames.append(detected_frame)
                            
                            # Update cumulative counts
                            for cls, count in class_counts.items():
                                if cls in cumulative_class_counts:
                                    cumulative_class_counts[cls] = max(cumulative_class_counts[cls], count)
                                else:
                                    cumulative_class_counts[cls] = count
                            
                            # Update graph data less frequently to improve performance
                            if processed_count % 3 == 0 or frame_index >= frame_count - process_every_n_frames:
                                # Calculate time based on video position, not real time
                                new_row = {'Time': current_time}
                                
                                # Add detection counts
                                for cls, count in class_counts.items():
                                    if cls not in st.session_state.detection_data_video.columns:
                                        st.session_state.detection_data_video[cls] = 0
                                    new_row[cls] = count
                                
                                # Fill missing classes with 0
                                for col in st.session_state.detection_data_video.columns:
                                    if col != 'Time' and col not in new_row:
                                        new_row[col] = 0
                                
                                # Add row to dataframe
                                st.session_state.detection_data_video = pd.concat([
                                    st.session_state.detection_data_video, 
                                    pd.DataFrame([new_row])
                                ], ignore_index=True)
                                
                                # Update graph less frequently
                                now = time.time()
                                if now - last_update_time > graph_update_interval:
                                    fig = update_detection_graph(fig, ax, st.session_state.detection_data_video)
                                    st_graph.pyplot(fig)
                                    last_update_time = now
                    else:
                        break
                
                # Clean up
                vid_cap.release()
                progress_bar.progress(100)
                status_text.text("Video processing complete!")
                
                # Display final graph
                if len(st.session_state.detection_data_video) > 0:
                    fig = update_detection_graph(fig, ax, st.session_state.detection_data_video)
                    st_graph.pyplot(fig)
                    
                # Display summary under the video/graph
                with results_container:
                    st.markdown("### Detection Results")
                    detection_text = ", ".join([f"{name}: {count}" for name, count in cumulative_class_counts.items()])
                    if detection_text:
                        st.write(detection_text)
                    else:
                        st.write("No objects detected")
                    
                    # Display detailed results
                    if cumulative_class_counts:
                        with st.expander("View Detailed Results"):
                            # Create a DataFrame from the results
                            results_df = pd.DataFrame(list(cumulative_class_counts.items()), columns=['Object', 'Count'])
                            st.dataframe(results_df)
            
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                
        if save_button and hasattr(st.session_state, "video_detected_frames") and len(st.session_state.video_detected_frames) > 0:
            try:
                # Show processing message
                saving_status = st.empty()
                saving_status.info("Creating video file and generating report...")
                
                # Create video from captured frames
                video_path = create_video_from_frames(
                    st.session_state.video_detected_frames, 
                    output_filename=f"video_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                
                # Generate report
                buffer, _ = create_detection_report(
                    task_type="Detection",
                    model_name=str(model.model.names),
                    conf=conf,
                    source_type="Video",
                    class_counts_over_time={},  # Will use dataframe instead
                    df=st.session_state.detection_data_video,
                    media_path=video_path
                )
                
                # Create download links
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # PDF report link
                report_link = generate_download_link(
                    buffer,
                    f"video_detection_report_{now}.pdf",
                    "Download Detection Report"
                )
                
                # Video link
                video_link = generate_video_download_link(
                    video_path,
                    "Download Detected Video"
                )
                
                # Display links
                saving_status.success("Video and report ready for download!")
                report_link_placeholder.markdown(
                    f"{report_link}<br>{video_link}", 
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error saving video results: {str(e)}")
        elif save_button and (not hasattr(st.session_state, "video_detected_frames") or len(st.session_state.video_detected_frames) == 0):
            st.warning("No video data to save. Please run detection first.")
    else:
        st.sidebar.warning("Please upload a video file")
