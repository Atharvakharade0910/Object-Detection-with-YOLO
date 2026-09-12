import subprocess
import webbrowser
import time
import os
import sys

def run_app():
    """Runs the main app and admin page"""
    print("Starting YOLOv8 Object Detection App with Authentication...")
    
    # Run the main app
    main_app_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
    
    # Wait a moment for the main app to start
    time.sleep(2)
    
    # Run the admin page
    admin_app_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "admin.py", "--server.port=8502"])
    
    # Open the main app in browser
    webbrowser.open("http://localhost:8501")
    print("Main app running at: http://localhost:8501")
    print("Admin panel running at: http://localhost:8502")
    
    try:
        # Keep the script running
        main_app_process.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        main_app_process.terminate()
        admin_app_process.terminate()
        print("Servers have been shut down.")

if __name__ == "__main__":
    run_app() 