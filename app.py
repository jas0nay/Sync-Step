import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# --- 1. PAGE CONFIGURATION (Professional UI) ---
st.set_page_config(
    page_title="Sync Step | Clinical Gait Analysis",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (Medical Design System) ---
st.markdown("""
<style>
    /* Force Dark Theme & Professional Fonts */
    .stApp {
        background-color: #0e1117;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6edf3;
        font-weight: 600;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# --- 4. MATH & LOGIC FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates angle between three points (e.g., Hip, Knee, Ankle)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def calculate_hip_drop(l_hip, r_hip):
    """Calculates vertical vertical difference between hips (Front View)."""
    # Y-coordinate difference (Vertical drop)
    return abs(l_hip[1] - r_hip[1]) * 100  # Multiplied by 100 for readable scale

# --- 5. VISUALIZATION FUNCTIONS ---

def draw_medical_hud(image, metrics, mode):
    """Draws a sterile, medical-grade overlay on the video feed."""
    height, width, _ = image.shape
    
    # 1. Sidebar Background (Semi-transparent black)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (300, height), (10, 10, 10), -1)
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # 2. Text Configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (240, 240, 240)
    teal = (0, 255, 255)
    
    # 3. Header
    cv2.putText(image, "SYNC STEP DIAGNOSTICS", (20, 40), font, 0.6, teal, 2, cv2.LINE_AA)
    cv2.putText(image, f"MODE: {mode.upper()}", (20, 70), font, 0.5, white, 1, cv2.LINE_AA)
    
    # 4. Dynamic Metrics Based on Mode
    y_pos = 120
    for key, value in metrics.items():
        cv2.putText(image, f"{key}:", (20, y_pos), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(image, str(value), (180, y_pos), font, 0.5, white, 1, cv2.LINE_AA)
        y_pos += 30
        
    return image

# --- 6. CORE APP LOGIC ---

def main():
    # Sidebar Navigation
    with st.sidebar:
        st.title("SYNC STEP")
        st.markdown("### Clinical Control Panel")
        
        mode = st.selectbox("Analysis Protocol", 
                            ["Sagittal (Side View)", "Coronal (Front View)"])
        
        source = st.radio("Input Source", ["Upload Video", "Live Camera"])
        
        st.divider()
        st.markdown("**Sensitivity Calibration**")
        threshold = st.slider("Deviation Threshold", 5, 40, 15)

    # Main Header
    st.title("Gait Analysis Laboratory")
    st.markdown(f"**Active Protocol:** {mode} | **Input:** {source}")

    # --- VIDEO PROCESSING ---
    cap = None
    tfile = None
    
    # Handle Input Source
    if source == "Upload Video":
        uploaded_file = st.file_uploader("Upload Patient Footage", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
    elif source == "Live Camera":
        if st.button("Start Live Session"):
            cap = cv2.VideoCapture(0) # 0 is usually the default webcam
            st.info("Live Feed Running in separate window. Press 'q' to quit.")

    # --- MAIN LOOP ---
    if cap is not None:
        stframe = st.empty()
        
        # Tracking Variables
        min_l_flex = 180  # Sagittal
        min_r_flex = 180  # Sagittal
        max_hip_diff = 0  # Coronal
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Formatting
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Dictionary to store current frame metrics for the HUD
            live_metrics = {}
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # ==========================
                # MODE 1: SAGITTAL (Side)
                # ==========================
                if mode == "Sagittal (Side View)":
                    # Get Coordinates
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    # Calculate Angles
                    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                    
                    # Update Peaks (Deepest Bend)
                    if l_angle < min_l_flex: min_l_flex = l_angle
                    if r_angle < min_r_flex: min_r_flex = r_angle
                    
                    live_metrics = {
                        "L Flexion (Deg)": int(l_angle),
                        "R Flexion (Deg)": int(r_angle),
                        "L Peak": int(min_l_flex),
                        "R Peak": int(min_r_flex)
                    }

                # ==========================
                # MODE 2: CORONAL (Front)
                # ==========================
                elif mode == "Coronal (Front View)":
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    
                    # Calculate Hip Drop (Trendelenburg Sign)
                    # We normalize simply by multiplying by 100 to get a "score"
                    current_hip_diff = abs(l_hip[1] - r_hip[1]) * 100
                    
                    if current_hip_diff > max_hip_diff:
                        max_hip_diff = current_hip_diff
                        
                    live_metrics = {
                        "Hip Drop Score": f"{current_hip_diff:.2f}",
                        "Max Hip Asym": f"{max_hip_diff:.2f}",
                        "Shoulder Align": "Stable" if abs(l_shoulder[1]-r_shoulder[1]) < 0.05 else "Unstable"
                    }

                # Draw Medical HUD
                image = draw_medical_hud(image, live_metrics, mode)
                
                # Draw Skeleton
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))
            
            # RENDER OUTPUT
            if source == "Live Camera":
                cv2.imshow('Sync Step Live Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                stframe.image(image, channels='BGR', use_container_width=True)

        cap.release()
        if source == "Live Camera":
            cv2.destroyAllWindows()
            
        # --- FINAL REPORT GENERATION ---
        if source == "Upload Video":
            st.divider()
            st.subheader("📋 Automated Clinical Report")
            
            c1, c2, c3 = st.columns(3)
            
            # --- DIAGNOSIS LOGIC FOR SAGITTAL ---
            if mode == "Sagittal (Side View)":
                diff = abs(min_l_flex - min_r_flex)
                
                c1.metric("L Peak Flexion", f"{int(min_l_flex)}°")
                c2.metric("R Peak Flexion", f"{int(min_r_flex)}°")
                c3.metric("Asymmetry Delta", f"{int(diff)}°")
                
                # CLASSIFIER
                if diff > threshold:
                    st.error(f"### 🔴 DETECTED: Antalgic Gait Pattern")
                    st.write(f"Significant asymmetry ({int(diff)}°) suggests pain avoidance or weakness in the limb with less flexion (Peak: {int(max(min_l_flex, min_r_flex))}°).")
                elif min_l_flex > 140 and min_r_flex > 140:
                    st.warning(f"### 🟡 DETECTED: Stiff-Knee Gait")
                    st.write("Bilateral reduction in knee flexion. Common in post-operative recovery or extensor spasticity.")
                else:
                    st.success(f"### 🟢 RESULT: Normal Gait Pattern")
                    st.write("Symmetrical flexion within healthy range.")

            # --- DIAGNOSIS LOGIC FOR CORONAL ---
            elif mode == "Coronal (Front View)":
                c1.metric("Max Hip Drop Score", f"{max_hip_diff:.2f}")
                
                # Thresholds for Hip Drop (These are heuristic/estimated values)
                if max_hip_diff > 5.0:  # 5.0 is an arbitrary 'high' number for the coordinate diff
                    st.error("### 🔴 DETECTED: Trendelenburg Sign")
                    st.write("Excessive pelvic drop during swing phase. Indicates weakness in the Gluteus Medius muscles.")
                else:
                    st.success("### 🟢 RESULT: Stable Pelvis")
                    st.write("No significant lateral pelvic tilt detected.")

if __name__ == '__main__':
    main()