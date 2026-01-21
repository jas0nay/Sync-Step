import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
import pandas as pd
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Sync Step Diagnostics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #ffffff; font-family: 'Roboto', sans-serif; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 15px; color: #3c4043;
    }
    
    /* Loading Spinner */
    .loader {
      border: 5px solid #f3f3f3;
      border-radius: 50%;
      border-top: 5px solid #1a73e8;
      width: 50px;
      height: 50px;
      -webkit-animation: spin 1s linear infinite; /* Safari */
      animation: spin 1s linear infinite;
      margin: 0 auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    /* Status Box */
    .status-box {
        padding: 20px;
        background-color: #f1f3f4;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border-left: 5px solid #1a73e8;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE SETUP ---
if 'analysis_data' not in st.session_state:
    st.session_state['analysis_data'] = None
if 'processed_video_path' not in st.session_state:
    st.session_state['processed_video_path'] = None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- CRITICAL FIX: CHANGED TO MODEL 1 TO FIX PERMISSION ERROR ---
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# --- 3. HELPER FUNCTIONS ---

def get_logo_path():
    if os.path.exists("logo.jpg"): return "logo.jpg"
    home = os.path.expanduser("~")
    downloads_path = os.path.join(home, "Downloads", "logo.jpg")
    if os.path.exists(downloads_path): return downloads_path
    return None

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_coordinates(landmarks):
    l = landmarks
    return {
        "Nose": [l[mp_pose.PoseLandmark.NOSE.value].x, l[mp_pose.PoseLandmark.NOSE.value].y],
        "L_Hip": [l[mp_pose.PoseLandmark.LEFT_HIP.value].x, l[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        "R_Hip": [l[mp_pose.PoseLandmark.RIGHT_HIP.value].x, l[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        "L_Knee": [l[mp_pose.PoseLandmark.LEFT_KNEE.value].x, l[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        "R_Knee": [l[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, l[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        "L_Ankle": [l[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, l[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
        "R_Ankle": [l[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, l[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
        "L_Shoulder": [l[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, l[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        "R_Shoulder": [l[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, l[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    }

def analyze_gait_pathology(metrics):
    # Failsafe
    if metrics['frames_with_legs'] < 10:
        return [{
            "name": "Gait Pattern Undetermined",
            "desc": "Insufficient body visibility. Please ensure full body (hips to ankles) is in frame.",
            "severity": 0,
            "confidence": 0,
            "steps": "Reposition camera to capture full body profile."
        }]

    diagnoses = []
    
    # 1. Hemiplegic (Stroke Pattern)
    if (metrics['l_knee_rom'] < 30 and metrics['r_knee_rom'] > 50) or (metrics['r_knee_rom'] < 30 and metrics['l_knee_rom'] > 50):
        diagnoses.append({
            "name": "Hemiplegic Pattern", 
            "desc": "Unilateral stiff knee detected with circumduction motion.", 
            "severity": 3, 
            "confidence": 95,
            "steps": """
            **1. Range of Motion (ROM):** Perform seated knee extensions (3 sets of 10) to improve flexibility.
            **2. Gait Training:** Practice 'High Knee' marching in place to force knee flexion.
            **3. Spasticity Management:** Consult a Physical Therapist about daily stretching routines for the calf and quadriceps.
            **4. Ankle Stability:** Use a resistance band for ankle dorsiflexion strengthening.
            """
        })

    # 2. Parkinsonian (Shuffling)
    if metrics['avg_knee_rom'] < 45 and metrics['arm_swing_magnitude'] < 0.05:
        diagnoses.append({
            "name": "Parkinsonian Gait", 
            "desc": "Shuffling steps and lack of arm swing.", 
            "severity": 2, 
            "confidence": 85,
            "steps": """
            **1. Visual Cues:** Place strips of tape on the floor 18 inches apart and practice stepping *over* them, not shuffling.
            **2. Arm Swing Drills:** Use walking poles (Nordic walking) to force exaggerated arm movement.
            **3. Rhythmic Auditory Stimulation:** Walk to a metronome beat (set to 100-110 BPM) to regulate stride.
            **4. Posture:** Practice 'chin tucks' and scapular retractions to prevent forward stooping.
            """
        })

    # 3. Trendelenburg (Hip Drop)
    if metrics['max_hip_drop'] > 8.0:
        diagnoses.append({
            "name": "Trendelenburg Gait", 
            "desc": "Significant pelvic drop detected (Gluteus Medius weakness).", 
            "severity": 2, 
            "confidence": 80,
            "steps": """
            **1. Clamshells:** Lay on side, knees bent, open top knee like a clam (3 sets of 15).
            **2. Side-Lying Leg Lifts:** Lay on side, keep leg straight, lift towards ceiling.
            **3. Single-Leg Stance:** Practice balancing on one foot for 30 seconds (hold a chair for safety).
            **4. Hip Hiking:** Stand on a step with one leg hanging off; lift the hanging hip up using waist muscles.
            """
        })

    # 4. Antalgic (Limp)
    if metrics['asymmetry_score'] > 15:
        # Check if secondary
        is_secondary = any(d['severity'] == 3 for d in diagnoses)
        if not is_secondary:
            diagnoses.append({
                "name": "Antalgic Gait", 
                "desc": "Asymmetry detected. Patient is favoring one side (Limp).", 
                "severity": 1, 
                "confidence": 75,
                "steps": """
                **1. Pain Identification:** Consult a specialist to rule out acute injury (stress fracture, sprain).
                **2. Aquatic Therapy:** Walking in a pool reduces weight bearing by 50-80%, allowing gait normalization.
                **3. Isometric Holds:** Perform static wall sits to build strength without joint impact.
                **4. Weight Shifting:** Stand with feet shoulder-width apart and shift weight slowly left to right.
                """
            })

    # Default: Healthy
    if not diagnoses:
        diagnoses.append({
            "name": "Healthy / Normal Gait", 
            "desc": "No significant deviations detected.", 
            "severity": 0, 
            "confidence": 98,
            "steps": """
            **1. Maintenance:** Continue current activity levels (150 mins moderate activity/week).
            **2. Strength:** Incorporate squats and lunges 2x per week.
            **3. Flexibility:** Stretch hamstrings and hip flexors daily.
            """
        })
        
    diagnoses.sort(key=lambda x: x['severity'], reverse=True)
    return diagnoses

def draw_overlay(image, live_data, frame_count, w_img):
    h, _, _ = image.shape
    
    # Analyzing Bar
    bar_width = int((frame_count % 45) / 45 * w_img)
    cv2.rectangle(image, (0, 0), (w_img, 4), (230, 230, 230), -1)
    cv2.rectangle(image, (0, 0), (bar_width, 4), (26, 115, 232), -1)
    
    cv2.putText(image, "SYNC STEP AI ENGINE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (26, 115, 232), 1, cv2.LINE_AA)
    
    # Data Box
    box_w = 220; box_h = 130
    x_start = w_img - box_w - 20; y_start = h - box_h - 20
    overlay = image.copy()
    cv2.rectangle(overlay, (x_start, y_start), (w_img - 20, h - 20), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
    y = y_start + 25
    for key, val in live_data.items():
        cv2.putText(image, key, (x_start + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(image, str(val), (x_start + 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y += 20
    return image

# --- 4. MAIN APPLICATION ---
def main():
    c1, c2 = st.columns([1, 8])
    with c1:
        logo_file = get_logo_path()
        if logo_file:
            st.image(logo_file, width=90)
        else:
            st.markdown("### [LOGO]")
    with c2:
        st.title("Sync Step Diagnostics")
        st.markdown("**Automated Biomechanical Assessment System**")

    # --- INSTRUCTIONS SECTION ---
    st.info("""
    **Instructions for Optimal Analysis:**
    * **Side View (Walking Left ↔ Right):** Best for analyzing **Knee Flexion**, **Limping**, and **Leg Stiffness**.
    * **Front View (Walking Towards Camera):** Best for analyzing **Hip Drop**, **Balance**, and **Trunk Lean**.
    * **Setup:** Ensure camera is at waist height. Lighting must be bright. Full body must be visible.
    """)
    
    st.divider()
    
    # Main Layout
    col_video, col_info = st.columns([2, 1])
    
    # --- IF NO ANALYSIS EXISTS, SHOW INPUT OPTIONS ---
    if st.session_state['analysis_data'] is None:
        with col_video:
            mode = st.radio("Input Source", ["Live Webcam", "Upload Video"], horizontal=True)
            
            # --- INPUT: LIVE WEBCAM ---
            if mode == "Live Webcam":
                st.info("System will record and analyze for exactly 30 seconds.")
                if st.button("Start 30s Recording & Analysis", type="primary"):
                    
                    # RIGHT COLUMN: SHOW SPINNER
                    with col_info:
                        st.markdown("""
                            <div class="status-box">
                                <div class="loader"></div>
                                <h3 style="color: #1a73e8; margin-top: 15px;">Initializing Neural Engine...</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Webcam not accessible.")
                    else:
                        # Setup File
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        st.session_state['processed_video_path'] = temp_file.name
                        
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        out = cv2.VideoWriter(st.session_state['processed_video_path'], cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                        
                        stframe = st.empty()
                        metrics_log = {"l_knee_angles": [], "r_knee_angles": [], "hip_drops": [], "frames_with_legs": 0}
                        
                        start_time = time.time()
                        frame_count = 0
                        status_placeholder = col_info.empty()
                        prev_live_data = {"Status": "Calibrating"}
                        prev_results = None

                        # RECORD LOOP
                        while (time.time() - start_time) < 30:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_count += 1
                            
                            # Update Right Side Spinner
                            status_placeholder.markdown(f"""
                                <div class="status-box">
                                    <div class="loader"></div>
                                    <h4 style="color: #1a73e8; margin-top: 15px;">Analyzing Frame {frame_count}</h4>
                                    <p>Do not close this tab.</p>
                                </div>
                            """, unsafe_allow_html=True)

                            # --- PERFORMANCE OPTIMIZATION ---
                            # 1. AI Vision: Downscale to 320px width (Very Fast)
                            h_orig, w_orig, _ = frame.shape
                            ai_frame = cv2.resize(frame, (320, int(h_orig * (320/w_orig))))
                            ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                            ai_frame.flags.writeable = False
                            
                            # 2. Frame Skipping: Analyze every 3rd frame
                            if frame_count % 3 == 0:
                                results = pose.process(ai_frame)
                                prev_results = results
                            else:
                                results = prev_results

                            # 3. Human Vision: Use 640px width (Clearer)
                            display_frame = cv2.resize(frame, (640, int(h_orig * (640/w_orig))))
                            h_disp, w_disp, _ = display_frame.shape

                            if results and results.pose_landmarks:
                                l_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility
                                r_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
                                
                                if l_vis > 0.5 and r_vis > 0.5:
                                    # Update metrics only on AI frames to save computation
                                    if frame_count % 3 == 0:
                                        metrics_log['frames_with_legs'] += 1
                                        coords = get_coordinates(results.pose_landmarks.landmark)
                                        l_knee = calculate_angle(coords['L_Hip'], coords['L_Knee'], coords['L_Ankle'])
                                        r_knee = calculate_angle(coords['R_Hip'], coords['R_Knee'], coords['R_Ankle'])
                                        hip_drop = abs(coords['L_Hip'][1] - coords['R_Hip'][1]) * 100
                                        
                                        metrics_log['l_knee_angles'].append(l_knee)
                                        metrics_log['r_knee_angles'].append(r_knee)
                                        metrics_log['hip_drops'].append(hip_drop)
                                        prev_live_data = {"L Flex": f"{int(l_knee)}", "R Flex": f"{int(r_knee)}"}

                                # Draw on Display Frame (Not AI Frame)
                                mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                                          mp_drawing.DrawingSpec(color=(26, 115, 232), thickness=2, circle_radius=2))

                            # Draw Overlay using cached data
                            display_frame = draw_overlay(display_frame, prev_live_data, frame_count, w_disp)
                            
                            # Countdown on Video
                            remaining = int(30 - (time.time() - start_time))
                            cv2.putText(display_frame, f"REC: {remaining}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            
                            # Save the High Res Display Frame to video file
                            # Note: We need to resize display_frame back to the VideoWriter size (w, h) or init VideoWriter with (w_disp, h_disp)
                            # To be safe, we resize to the writer dimensions
                            final_out = cv2.resize(display_frame, (w, h))
                            out.write(final_out)
                            stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        cap.release()
                        out.release()
                        stframe.empty()
                        
                        st.session_state['analysis_data'] = metrics_log
                        st.rerun()

            # --- INPUT: UPLOAD VIDEO ---
            elif mode == "Upload Video":
                uploaded_file = st.file_uploader("Upload File", type=['mp4','mov'])
                if uploaded_file and st.button("Analyze Upload"):
                    
                    with col_info:
                        st.markdown("""
                            <div class="status-box">
                                <div class="loader"></div>
                                <h3 style="color: #1a73e8; margin-top: 15px;">Processing Video Data...</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    
                    cap = cv2.VideoCapture(tfile.name)
                    
                    # Prepare Output File
                    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    st.session_state['processed_video_path'] = temp_out.name
                    
                    w = int(cap.get(3))
                    h = int(cap.get(4))
                    out = cv2.VideoWriter(st.session_state['processed_video_path'], cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                    
                    metrics_log = {"l_knee_angles": [], "r_knee_angles": [], "hip_drops": [], "frames_with_legs": 0}
                    stframe = st.empty()
                    frame_count = 0
                    status_placeholder = col_info.empty()
                    
                    prev_live_data = {"Status": "Initializing"}
                    prev_results = None

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_count += 1
                        
                        if frame_count % 5 == 0:
                            status_placeholder.markdown(f"""
                                <div class="status-box">
                                    <div class="loader"></div>
                                    <h4 style="color: #1a73e8; margin-top: 15px;">Analyzing Frame {frame_count}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # --- PERFORMANCE OPTIMIZATION ---
                        # 1. AI Vision: Downscale to 320px
                        h_orig, w_orig, _ = frame.shape
                        ai_frame = cv2.resize(frame, (320, int(h_orig * (320/w_orig))))
                        ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                        
                        # 2. Frame Skipping (Every 3rd)
                        if frame_count % 3 == 0:
                            results = pose.process(ai_frame)
                            prev_results = results
                        else:
                            results = prev_results
                        
                        # 3. Human Vision: 640px
                        display_frame = cv2.resize(frame, (640, int(h_orig * (640/w_orig))))
                        h_disp, w_disp, _ = display_frame.shape
                        
                        if results and results.pose_landmarks:
                            l_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility
                            r_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
                            
                            if l_vis > 0.5 and r_vis > 0.5:
                                if frame_count % 3 == 0:
                                    metrics_log['frames_with_legs'] += 1
                                    coords = get_coordinates(results.pose_landmarks.landmark)
                                    l_knee = calculate_angle(coords['L_Hip'], coords['L_Knee'], coords['L_Ankle'])
                                    r_knee = calculate_angle(coords['R_Hip'], coords['R_Knee'], coords['R_Ankle'])
                                    hip_drop = abs(coords['L_Hip'][1] - coords['R_Hip'][1]) * 100
                                    
                                    metrics_log['l_knee_angles'].append(l_knee)
                                    metrics_log['r_knee_angles'].append(r_knee)
                                    metrics_log['hip_drops'].append(hip_drop)
                                    prev_live_data = {"L Flex": f"{int(l_knee)}", "R Flex": f"{int(r_knee)}"}

                            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                                                      mp_drawing.DrawingSpec(color=(26, 115, 232), thickness=2, circle_radius=2))

                        display_frame = draw_overlay(display_frame, prev_live_data, frame_count, w_disp)
                        
                        final_out = cv2.resize(display_frame, (w, h))
                        out.write(final_out)
                        stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    cap.release()
                    out.release()
                    st.session_state['analysis_data'] = metrics_log
                    st.rerun()

    # --- STATE 2: REPORT & REPLAY ---
    else:
        # VIDEO REPLAY COLUMN
        with col_video:
            st.subheader("Session Recording")
            if st.session_state['processed_video_path']:
                st.video(st.session_state['processed_video_path'])
            
            if st.button("Start New Session", type="primary"):
                st.session_state['analysis_data'] = None
                st.session_state['processed_video_path'] = None
                st.rerun()

        # REPORT COLUMN (Right Side)
        with col_info:
            st.subheader("Diagnostic Report")
            metrics = st.session_state['analysis_data']
            
            if metrics and metrics['l_knee_angles']:
                max_l = min(metrics['l_knee_angles'])
                max_r = min(metrics['r_knee_angles'])
                summary = {
                    'frames_with_legs': metrics['frames_with_legs'],
                    'max_hip_drop': np.max(metrics['hip_drops']),
                    'asymmetry_score': abs(max_l - max_r),
                    'l_knee_rom': 180 - max_l,
                    'r_knee_rom': 180 - max_r,
                    'min_knee_angle_during_swing': min(max_l, max_r),
                    'arm_swing_magnitude': 0.1,
                    'avg_knee_rom': (360-max_l-max_r)/2
                }
            else:
                summary = {'frames_with_legs': 0}

            findings = analyze_gait_pathology(summary)
            primary = findings[0]
            
            if primary['name'] == "Gait Pattern Undetermined":
                st.warning(f"### {primary['name']}")
                st.write(primary['desc'])
            else:
                color = "#0f9d58" if primary['severity'] == 0 else "#d93025"
                st.markdown(f"""
                <div style="border: 1px solid {color}; border-left: 8px solid {color}; padding: 20px; background: #fff; border-radius: 5px;">
                    <h3 style="color: {color}; margin:0;">PRIMARY DIAGNOSIS:</h3>
                    <h2 style="color: #333; margin:0;">{primary['name']}</h2>
                    <p style="margin-top:10px;">{primary['desc']}</p>
                    <hr>
                    <p><strong>Confidence:</strong> {primary['confidence']}%</p>
                </div>""", unsafe_allow_html=True)
                
                st.markdown("#### Recommended Correction Steps")
                st.info(primary['steps'])
                
                st.markdown("#### Biometric Telemetry")
                df = pd.DataFrame({
                    "Metric": ["L Knee Flexion", "R Knee Flexion", "Asymmetry", "Hip Drop"],
                    "Value": [f"{int(summary['l_knee_rom'])} deg", f"{int(summary['r_knee_rom'])} deg", 
                              f"{int(summary['asymmetry_score'])} deg", f"{summary['max_hip_drop']:.1f}"]
                })
                st.dataframe(df, use_container_width=True)

if __name__ == '__main__':
    main()
