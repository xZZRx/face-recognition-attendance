import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from PIL import Image
import pickle
import os
import datetime
import pandas as pd
import json
import time

# Page config for mobile responsiveness
st.set_page_config(
    page_title="College Event Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #ff6b6b;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 16px;
        font-weight: bold;
    }
    .attendance-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .student-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        color: #333;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize face detector
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces(image, face_cascade):
    """Detect faces using Haar Cascade with better filtering"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization for better detection
    gray = cv2.equalizeHist(gray)
    
    # More strict parameters to reduce false positives
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(80, 80),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filter faces by aspect ratio to remove false positives
    valid_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.7 <= aspect_ratio <= 1.3:
            image_area = image.shape[0] * image.shape[1]
            face_area = w * h
            if face_area / image_area > 0.01:
                valid_faces.append((x, y, x+w, y+h))
    
    return valid_faces

def extract_face_features(image, face_coords):
    """Extract simple features from face region"""
    x1, y1, x2, y2 = face_coords
    face_roi = image[y1:y2, x1:x2]
    
    if face_roi.size == 0:
        return None
    
    face_roi = cv2.resize(face_roi, (100, 100))
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    features = gray_face.flatten().astype(np.float32)
    features = features / 255.0
    
    return features

def compare_faces(known_features, unknown_features, threshold=0.6):
    """Compare face features using cosine similarity"""
    if known_features is None or unknown_features is None:
        return False, 0.0
    
    dot_product = np.dot(known_features, unknown_features)
    norm_a = np.linalg.norm(known_features)
    norm_b = np.linalg.norm(unknown_features)
    
    if norm_a == 0 or norm_b == 0:
        return False, 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return similarity > threshold, similarity

def save_attendance_record(student_name, event_name):
    """Save attendance record to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    record = {
        'student_name': student_name,
        'event_name': event_name,
        'timestamp': timestamp,
        'date': datetime.datetime.now().strftime("%Y-%m-%d")
    }
    
    try:
        with open('attendance_records.json', 'r') as f:
            records = json.load(f)
    except FileNotFoundError:
        records = []
    
    records.append(record)
    
    with open('attendance_records.json', 'w') as f:
        json.dump(records, f, indent=2)

def get_attendance_summary(event_name):
    """Get attendance summary for an event"""
    try:
        with open('attendance_records.json', 'r') as f:
            records = json.load(f)
        
        event_records = [r for r in records if r['event_name'] == event_name]
        return event_records
    except FileNotFoundError:
        return []

# Initialize session state
if "students" not in st.session_state:
    st.session_state.students = {}

if "current_event" not in st.session_state:
    st.session_state.current_event = ""

if "attendance_today" not in st.session_state:
    st.session_state.attendance_today = []

# Automatically load student data from file on startup
try:
    with open("student_database.pkl", "rb") as f:
        loaded_students = pickle.load(f)
        st.session_state.students.update(loaded_students)
except FileNotFoundError:
    pass  # No student data file exists yet, so continue with empty students dict

# Load face detector
try:
    face_cascade = load_face_detector()
    if face_cascade.empty():
        st.error("❌ Failed to load face detection system")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading face detection: {e}")
    st.stop()

# Header
st.markdown("""
<div class="attendance-card">
    <h1>🎓 College Event Attendance System</h1>
    <p>AI-Powered Face Recognition for Student Attendance</p>
</div>
""", unsafe_allow_html=True)

# Event Setup
st.subheader("📅 Event Configuration")
col1, col2 = st.columns(2)
with col1:
    event_name = st.text_input("Event Name", value="Computer Science Seminar 2024")
with col2:
    event_date = st.date_input("Event Date", value=datetime.date.today())

if event_name:
    st.session_state.current_event = event_name

# Tabs for different functions
tab1, tab2, tab3, tab4 = st.tabs(["👥 Student Registration", "✅ Take Attendance", "📊 View Records", "⚙️ System Management"])

with tab1:
    st.header("👥 Student Registration")
    st.info("Register students using webcam or upload photos")
    
    reg_method = st.radio("Choose registration method:", ["📸 Upload Photo", "📹 Use Webcam"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", placeholder="e.g., CS-2024-001")
    with col2:
        student_name = st.text_input("Student Name", placeholder="e.g., John Doe")
    
    if reg_method == "📸 Upload Photo":
        uploaded_file = st.file_uploader("Upload student photo", type=["jpg", "png", "jpeg"])
        
        if uploaded_file and student_id and student_name:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_array = np.array(pil_image)
                
                faces = detect_faces(image_array, face_cascade)
                
                if faces:
                    face_coords = faces[0]
                    features = extract_face_features(image_array, face_coords)
                    
                    if features is not None:
                        st.session_state.students[student_id] = {
                            "name": student_name,
                            "features": features
                        }
                        
                        # Save student data to file
                        with open("student_database.pkl", "wb") as f:
                            pickle.dump(st.session_state.students, f)
                        
                        result_image = image_array.copy()
                        x1, y1, x2, y2 = face_coords
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(result_image, f"{student_name}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ <strong>Student registered successfully!</strong><br>
                            ID: {student_id}<br>
                            Name: {student_name}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(result_image, caption=f"Registered: {student_name}", width=300)
                else:
                    st.error("❌ No face detected. Please use a clear photo with visible face.")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    else:
        if not student_id or not student_name:
            st.warning("⚠️ Please enter Student ID and Name first!")
        else:
            st.info("📹 Look at the camera and click 'Capture & Register' when ready")
            
            if "webcam_key" not in st.session_state:
                st.session_state.webcam_key = 0
            
            class RegistrationProcessor:
                def __init__(self):
                    self.latest_frame = None
                    
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        self.latest_frame = rgb_img.copy()
                        
                        faces = detect_faces(rgb_img, face_cascade)
                        
                        for face_coords in faces:
                            x1, y1, x2, y2 = face_coords
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            if face_width > 60 and face_height > 60:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, "Face detected - Ready!", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(img, "Too small", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        if len(faces) == 0:
                            cv2.putText(img, "Please position your face in view", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                    except Exception as e:
                        print(f"Registration processor error: {e}")
                        return frame
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                webrtc_ctx = webrtc_streamer(
                    key=f"registration_{st.session_state.webcam_key}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    video_processor_factory=RegistrationProcessor,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 640, "ideal": 1280, "max": 1920},
                            "height": {"min": 480, "ideal": 720, "max": 1080},
                            "frameRate": {"min": 10, "ideal": 15, "max": 30}
                        }, 
                        "audio": False
                    },
                    async_processing=True
                )
            
            with col2:
                st.markdown("### 📸 Registration")
                
                if webrtc_ctx.video_processor:
                    st.success("✅ Camera active")
                    
                    if st.button("📸 Capture & Register Student", type="primary", use_container_width=True):
                        if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'latest_frame'):
                            if webrtc_ctx.video_processor.latest_frame is not None:
                                try:
                                    image_array = webrtc_ctx.video_processor.latest_frame
                                    faces = detect_faces(image_array, face_cascade)
                                    
                                    if faces:
                                        face_coords = faces[0]
                                        features = extract_face_features(image_array, face_coords)
                                        
                                        if features is not None:
                                            st.session_state.students[student_id] = {
                                                "name": student_name,
                                                "features": features
                                            }
                                            
                                            # Save student data to file
                                            with open("student_database.pkl", "wb") as f:
                                                pickle.dump(st.session_state.students, f)
                                            
                                            st.success(f"✅ {student_name} registered!")
                                            
                                            result_image = image_array.copy()
                                            x1, y1, x2, y2 = face_coords
                                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                            st.image(result_image, caption=f"Captured: {student_name}", width=250)
                                            
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("❌ Could not extract face features")
                                    else:
                                        st.error("❌ No face detected in capture!")
                                        
                                except Exception as e:
                                    st.error(f"❌ Capture failed: {str(e)}")
                            else:
                                st.error("❌ No frame available - wait a moment")
                        else:
                            st.error("❌ Camera not ready")
                else:
                    st.info("📹 Starting camera...")
                
                if st.button("🔄 Reset Camera", key="reset_registration_cam", help="Click if camera is not working"):
                    st.session_state.webcam_key += 1
                    st.rerun()
    
    if st.session_state.students:
        st.markdown("### 📋 Registered Students")
        for student_id, data in st.session_state.students.items():
            st.markdown(f"""
            <div class="student-list">
                🎓 <strong>{data['name']}</strong> (ID: {student_id})
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("✅ Take Attendance")
    
    debug_mode = st.checkbox("🔍 Debug Mode (Show detection details)")
    
    if not st.session_state.current_event:
        st.warning("⚠️ Please set an event name first!")
    elif not st.session_state.students:
        st.warning("⚠️ Please register students first!")
    else:
        st.info(f"📅 Taking attendance for: **{st.session_state.current_event}**")
        
        if debug_mode:
            st.info(f"📊 Registered students: {len(st.session_state.students)}")
        
        att_method = st.radio("Choose attendance method:", ["📸 Upload Photo", "📹 Live Webcam"], horizontal=True)
        
        if att_method == "📸 Upload Photo":
            uploaded_attendance = st.file_uploader("Upload photo for attendance", type=["jpg", "png", "jpeg"], key="attendance")
            
            if uploaded_attendance:
                try:
                    pil_image = Image.open(uploaded_attendance).convert("RGB")
                    image_array = np.array(pil_image)
                    
                    faces = detect_faces(image_array, face_cascade)
                    
                    if faces:
                        result_image = image_array.copy()
                        recognized_students = []
                        
                        for face_coords in faces:
                            features = extract_face_features(image_array, face_coords)
                            
                            if features is not None:
                                best_match = None
                                best_similarity = 0
                                
                                for student_id, data in st.session_state.students.items():
                                    is_match, similarity = compare_faces(data["features"], features, threshold=0.5)
                                    if is_match and similarity > best_similarity:
                                        best_similarity = similarity
                                        best_match = (student_id, data["name"])
                                
                                x1, y1, x2, y2 = face_coords
                                if best_match:
                                    student_id, name = best_match
                                    recognized_students.append((student_id, name))
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(result_image, f"{name}", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                                    cv2.putText(result_image, "Unknown", (x1, y1 - 10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        st.image(result_image, caption="Attendance Recognition Result", width=500)
                        
                        if recognized_students:
                            st.markdown("### ✅ Attendance Marked For:")
                            for student_id, name in recognized_students:
                                save_attendance_record(name, st.session_state.current_event)
                                st.markdown(f"""
                                <div class="success-box">
                                    ✅ <strong>{name}</strong> (ID: {student_id}) - Present
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No registered students recognized in the photo.")
                    else:
                        st.warning("No faces detected in the image.")
                        
                except Exception as e:
                    st.error(f"Error processing attendance: {str(e)}")
        
        else:
            st.info("📹 Look at the camera and click 'Mark Attendance' when ready")
            
            if "attendance_webcam_key" not in st.session_state:
                st.session_state.attendance_webcam_key = 0
            
            if "attendance_session" not in st.session_state:
                st.session_state.attendance_session = []
            
            class AttendanceProcessor:
                def __init__(self):
                    self.latest_frame = None
                    
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        self.latest_frame = rgb_img.copy()
                        
                        faces = detect_faces(rgb_img, face_cascade)
                        
                        for face_coords in faces:
                            x1, y1, x2, y2 = face_coords
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            if face_width > 60 and face_height > 60:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img, "Face detected - Ready!", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
                                cv2.putText(img, "Too small", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        if len(faces) == 0:
                            cv2.putText(img, "Please position your face in view", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                    except Exception as e:
                        print(f"Attendance processor error: {e}")
                        return frame
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                webrtc_ctx = webrtc_streamer(
                    key=f"attendance_{st.session_state.attendance_webcam_key}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    video_processor_factory=AttendanceProcessor,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 640, "ideal": 1280, "max": 1920},
                            "height": {"min": 480, "ideal": 720, "max": 1080},
                            "frameRate": {"min": 10, "ideal": 15, "max": 30}
                        }, 
                        "audio": False
                    },
                    async_processing=True
                )
            
            with col2:
                st.markdown("### 📸 Mark Attendance")
                
                if webrtc_ctx.video_processor:
                    st.success("✅ Camera active")
                    
                    if st.button("✅ Mark My Attendance", type="primary", use_container_width=True):
                        if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'latest_frame'):
                            if webrtc_ctx.video_processor.latest_frame is not None:
                                try:
                                    image_array = webrtc_ctx.video_processor.latest_frame
                                    faces = detect_faces(image_array, face_cascade)
                                    
                                    if faces:
                                        face_coords = faces[0]
                                        features = extract_face_features(image_array, face_coords)
                                        
                                        if features is not None:
                                            best_match = None
                                            best_similarity = 0
                                            
                                            for student_id, data in st.session_state.students.items():
                                                is_match, similarity = compare_faces(data["features"], features, threshold=0.4)
                                                if is_match and similarity > best_similarity:
                                                    best_similarity = similarity
                                                    best_match = (student_id, data["name"])
                                            
                                            if best_match and best_similarity > 0.3:
                                                student_id, name = best_match
                                                save_attendance_record(name, st.session_state.current_event)
                                                current_time = datetime.datetime.now()
                                                st.session_state.attendance_session.append((student_id, name, current_time.strftime("%H:%M:%S")))
                                                st.success(f"✅ Attendance marked for {name}!")
                                                st.balloons()
                                                result_image = image_array.copy()
                                                x1, y1, x2, y2 = face_coords
                                                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                                cv2.putText(result_image, f"{name} ({best_similarity:.2f})", (x1, y1 - 10), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                                st.image(result_image, caption=f"Attendance: {name}", width=250)
                                            else:
                                                st.error("❌ Face not recognized! Please register first.")
                                        else:
                                            st.error("❌ Could not extract face features")
                                    else:
                                        st.error("❌ No face detected in capture!")
                                        
                                except Exception as e:
                                    st.error(f"❌ Attendance failed: {str(e)}")
                            else:
                                st.error("❌ No frame available - wait a moment")
                        else:
                            st.error("❌ Camera not ready")
                else:
                    st.info("📹 Starting camera...")
                
                if st.button("🔄 Reset Camera", key="reset_attendance_cam", help="Click if camera is not working"):
                    st.session_state.attendance_webcam_key += 1
                    st.rerun()
                
                st.markdown("### 📋 Recent Attendance")
                if st.session_state.attendance_session:
                    for student_id, name, time in st.session_state.attendance_session[-5:]:
                        st.markdown(f"""
                        <div style="background-color: #d4edda; padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; font-size: 12px; color: #333;">
                            ✅ <strong>{name}</strong><br>
                            🕐 {time}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No attendance marked yet")
                
                if st.button("🔄 Clear Session", key="clear_attendance_session", help="Clear current session attendance display"):
                    st.session_state.attendance_session = []
                    st.rerun()

with tab3:
    st.header("📊 Attendance Records")
    
    if st.session_state.current_event:
        records = get_attendance_summary(st.session_state.current_event)
        
        if records:
            st.markdown(f"### 📅 {st.session_state.current_event}")
            
            df = pd.DataFrame(records)
            st.markdown(f"**Total Attendance: {len(records)} students**")
            
            for record in records:
                st.markdown(f"""
                <div class="student-list">
                    ✅ <strong>{record['student_name']}</strong><br>
                    🕐 {record['timestamp']}
                </div>
                """, unsafe_allow_html=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Attendance Report (CSV)",
                data=csv,
                file_name=f"attendance_{st.session_state.current_event}_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("No attendance records found for this event.")
    else:
        st.warning("Please select an event to view records.")

with tab4:
    st.header("⚙️ System Management")
    
    st.info("ℹ️ Student data is automatically saved after each registration and loaded on app startup.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Student Data (Manual)"):
            if st.session_state.students:
                with open("student_database.pkl", "wb") as f:
                    pickle.dump(st.session_state.students, f)
                st.success("✅ Student data saved!")
            else:
                st.warning("No student data to save")
    
    with col2:
        if st.button("📂 Load Student Data (Manual)"):
            try:
                with open("student_database.pkl", "rb") as f:
                    loaded_students = pickle.load(f)
                    st.session_state.students.update(loaded_students)
                st.success("✅ Student data loaded!")
                st.rerun()
            except FileNotFoundError:
                st.error("No saved student data found")
    
    with col3:
        if st.button("🗑️ Clear All Data"):
            st.session_state.students = {}
            st.session_state.attendance_today = []
            if os.path.exists("student_database.pkl"):
                os.remove("student_database.pkl")
            st.success("✅ All data cleared!")
            st.rerun()
    
    st.markdown("### 📈 System Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("👥 Registered Students", len(st.session_state.students))
    
    with col2:
        if st.session_state.current_event:
            records = get_attendance_summary(st.session_state.current_event)
            st.metric("✅ Today's Attendance", len(records))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 College Event Attendance System - Face Recognition Prototype</p>
    <p>Built with Streamlit • OpenCV • AI-Powered Recognition</p>
</div>
""", unsafe_allow_html=True)