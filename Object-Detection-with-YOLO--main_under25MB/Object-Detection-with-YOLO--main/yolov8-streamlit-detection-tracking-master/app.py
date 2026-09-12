# Python In-built packages
from pathlib import Path
import PIL
import os
import hashlib
import json

# External packages
import streamlit as st

# Set page configuration first - this must come before any other Streamlit commands
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Other imports after st.set_page_config()
import cv2
import numpy as np
import base64
import tempfile
import time
import pandas as pd
from datetime import datetime

# Local Modules
import settings
import helper

# Authentication functionality
def check_password():
    """Returns `True` if the user had the correct password."""
    # Default credentials - in production you should use a more secure method
    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "admin123"

    def credentials_entered():
        """Checks whether the credentials entered by the user are correct."""
        if (st.session_state["username"] == DEFAULT_USERNAME and 
            st.session_state["password"] == DEFAULT_PASSWORD):
            st.session_state["password_correct"] = True
            del st.session_state["username"]  # don't store credentials
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for credentials
        st.title("Login")
        st.text_input("Username", on_change=credentials_entered, key="username")
        st.text_input("Password", type="password", on_change=credentials_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Credentials not correct, show inputs + error
        st.title("Login")
        st.text_input("Username", on_change=credentials_entered, key="username")
        st.text_input("Password", type="password", on_change=credentials_entered, key="password")
        st.error("😕 Invalid username or password")
        return False
    else:
        # Credentials correct
        return True

def main_app():
    """The main application that runs after authentication"""
    # Main page heading
    st.title("Object Detection And Tracking using YOLOv8")

    # Sidebar
    st.sidebar.header("ML Model Config")
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state["password_correct"] = False
        st.experimental_rerun()

    # Model Options
    model_type = st.sidebar.selectbox(
        "Select Task",
        ["Detection", "Segmentation"]
    )

    confidence = float(st.sidebar.slider(
        "Select Confidence",
        25, 100, 40)) / 100

    # Selecting Detection Or Segmentation
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return

    # Display model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    model_info = f"Task: {model_type}<br>Model: {model_path.name}<br>Confidence: {confidence:.2f}"
    st.sidebar.markdown(model_info, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.sidebar.header("Image/Video Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source", [settings.IMAGE, settings.VIDEO, settings.WEBCAM])

    source_img = None
    # If image is selected
    if source_selectbox == settings.IMAGE:
        # Get YOLO classes for selection
        class_names = helper.get_yolo_classes(model)
        class_list = list(class_names.values())
        
        # Add class selection
        selected_classes_names = st.sidebar.multiselect("Select objects to detect", class_list)
        
        # Convert selected class names to indices
        selected_classes = None
        if selected_classes_names:
            selected_classes = [list(class_names.keys())[list(class_names.values()).index(name)] 
                               for name in selected_classes_names]
        
        # File uploader for image
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
        )

        col1, col2 = st.columns(2)
        with col1:
            detect_button = st.button("Detect Objects")
        with col2:
            save_button = st.button("Save Results")
        
        # Report link placeholder
        report_link_placeholder = st.empty()
        
        # Image display container - will show either original or detected
        image_container = st.empty()
        # Results container - will show below the image
        results_container = st.container()
            
        if source_img is None:
            # Display instructions instead of default image
            image_container.info("👆 Please upload an image using the file uploader in the sidebar.")
        else:
            uploaded_image = PIL.Image.open(source_img)
            image_container.image(source_img, caption="Uploaded Image",
                     use_container_width=True)

        if detect_button and source_img is not None:
            try:
                # Convert PIL Image to numpy array for processing
                image_array = np.array(uploaded_image)
                
                # Process the image with the YOLO model
                res = model.predict(image_array, conf=confidence, classes=selected_classes)
                
                # Get detection results
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
                
                # Store results in session state
                st.session_state.image_detection_results = class_counts
                
                # Format the detection results
                detection_text = ", ".join([f"{name}: {count}" for name, count in class_counts.items()])
                if not detection_text:
                    detection_text = "No objects detected"
                
                # Plot the detected objects on the image
                res_plotted = res[0].plot()
                
                # Store the detected image in session state for later use
                st.session_state.last_detection_result = res_plotted
                
                # Replace the original image with the detected image
                image_container.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
                # Display detection summary under the image
                with results_container:
                    st.markdown("### Detection Results")
                    st.write(detection_text)
                    
                    # Display detailed results
                    if class_counts:
                        with st.expander("View Detailed Results"):
                            # Create a DataFrame from the results
                            results_df = pd.DataFrame(list(class_counts.items()), columns=['Object', 'Count'])
                            st.dataframe(results_df)
                
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
        elif detect_button and source_img is None:
            st.error("Please upload an image first")
            
        if save_button and "image_detection_results" in st.session_state and source_img is not None:
            try:
                # Save the detected image
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                detected_image_path = None
                
                if "last_detection_result" in st.session_state and st.session_state.last_detection_result is not None:
                    # Get the image with detections
                    detected_image = st.session_state.last_detection_result
                    # Save it to a file
                    detected_image_path = helper.save_detected_image(
                        detected_image, 
                        f"detected_image_{now}.jpg"
                    )
                
                # Generate report PDF
                buffer, _ = helper.create_detection_report(
                    task_type=model_type,
                    model_name=model_path.name,
                    conf=confidence,
                    source_type="Image",
                    class_counts_over_time=st.session_state.image_detection_results,
                    media_path=detected_image_path
                )
                
                # Create download links
                report_link = helper.generate_download_link(
                    buffer,
                    f"image_detection_report_{now}.pdf",
                    "Download Detection Report"
                )
                
                # Create image download link if available
                if detected_image_path and os.path.exists(detected_image_path):
                    with open(detected_image_path, "rb") as f:
                        image_bytes = f.read()
                    
                    b64 = base64.b64encode(image_bytes).decode()
                    image_link = f'<a href="data:image/jpeg;base64,{b64}" download="detected_image_{now}.jpg">Download Detected Image</a>'
                    
                    # Display both links
                    report_link_placeholder.markdown(
                        f"{report_link}<br>{image_link}", 
                        unsafe_allow_html=True
                    )
                else:
                    # Just display the report link
                    report_link_placeholder.markdown(report_link, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error saving results: {str(e)}")
        elif save_button and (source_img is None or "image_detection_results" not in st.session_state):
            st.error("No detection results to save. Please upload an image and run detection first.")

    elif source_selectbox == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_selectbox == settings.WEBCAM:
        helper.play_webcam(confidence, model)

    else:
        st.error("Please select a valid source type!")

# Main execution
if __name__ == "__main__":
    # Initialize session state for detection results
    if "image_detection_results" not in st.session_state:
        st.session_state.image_detection_results = {}
    
    # Check password and run app
    if check_password():
        main_app()
