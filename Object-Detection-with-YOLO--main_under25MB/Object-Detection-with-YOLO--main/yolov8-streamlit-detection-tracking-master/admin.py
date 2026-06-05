# Python In-built packages
import hashlib
import json
import os
from pathlib import Path

# External packages
import streamlit as st

# Local imports
import settings

# Setting page layout
st.set_page_config(
    page_title="Admin Panel - YOLOv8 Detection",
    page_icon="🔐",
    layout="wide",
)

# File path for storing users (in a real app, use a database)
USERS_FILE = Path("users.json")

# Initialize users file if it doesn't exist
def initialize_users_file():
    if not USERS_FILE.exists():
        default_users = {
            "admin": hashlib.sha256("admin123".encode()).hexdigest()
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f)
    
    # Load existing users
    with open(USERS_FILE, "r") as f:
        return json.load(f)

# Save users to file
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Authentication check
def check_admin_password():
    """Returns `True` if the user has admin credentials."""
    
    # Initialize session state variables
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
    
    # If already authenticated, return True
    return st.session_state["admin_authenticated"]

def admin_login_form():
    """Display the admin login form"""
    st.title("Admin Login")
    
    # Create login form
    with st.form("admin_login_form"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            validate_admin_login()
    
    # Add link back to main app
    st.markdown("Return to [main application](/?)")

def validate_admin_login():
    """Validates the admin credentials."""
    users = initialize_users_file()
    
    if (
        st.session_state["username"] == "admin" and
        users["admin"] == hashlib.sha256(st.session_state["password"].encode()).hexdigest()
    ):
        st.session_state["admin_authenticated"] = True
        st.rerun()  # Force rerun to update UI
        return True
    else:
        st.session_state["admin_authenticated"] = False
        st.error("Invalid admin credentials")
        return False

def admin_panel():
    """The main admin panel that runs after authentication"""
    st.title("User Management")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["admin_authenticated"] = False
        st.rerun()
    
    # Load existing users
    users = initialize_users_file()
    
    # Display existing users
    st.subheader("Existing Users")
    
    # Create a table with usernames
    user_list = list(users.keys())
    if user_list:
        st.write("Users:")
        for user in user_list:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 {user}")
            with col2:
                if user != "admin":  # Prevent deleting the admin user
                    if st.button(f"Delete {user}", key=f"del_{user}"):
                        del users[user]
                        save_users(users)
                        st.success(f"User {user} deleted!")
                        st.rerun()
    else:
        st.info("No users found.")
    
    # Add new user form
    st.subheader("Add New User")
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submit = st.form_submit_button("Add User")
        
        if submit:
            if not new_username:
                st.error("Username cannot be empty")
            elif new_username in users:
                st.error(f"User '{new_username}' already exists")
            elif not new_password:
                st.error("Password cannot be empty")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                # Add new user
                users[new_username] = hashlib.sha256(new_password.encode()).hexdigest()
                save_users(users)
                st.success(f"User '{new_username}' added successfully!")
                st.rerun()
    
    # Change admin password
    st.subheader("Change Admin Password")
    with st.form("change_admin_form"):
        current_password = st.text_input("Current Password", type="password")
        new_admin_password = st.text_input("New Password", type="password")
        confirm_admin_password = st.text_input("Confirm New Password", type="password")
        
        submit = st.form_submit_button("Change Password")
        
        if submit:
            if not current_password or not new_admin_password:
                st.error("Passwords cannot be empty")
            elif hashlib.sha256(current_password.encode()).hexdigest() != users["admin"]:
                st.error("Current password is incorrect")
            elif new_admin_password != confirm_admin_password:
                st.error("New passwords do not match")
            else:
                # Update admin password
                users["admin"] = hashlib.sha256(new_admin_password.encode()).hexdigest()
                save_users(users)
                st.success("Admin password changed successfully!")

# Main execution
if check_admin_password():
    admin_panel()
else:
    admin_login_form() 