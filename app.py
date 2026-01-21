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
    div[data-testid="metric-container"] {
        background-color: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 15px; color: #3c4043;
    }
    .status-box {
        padding: 20px;
        background-color: #f1f3f4;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border-left: 5px solid #1a73e8;
    }
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
# MODEL COMPLEXITY 1 to avoid download permission errors on Cloud
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

# --- 3. HELPER FUNCTIONS ---

def get_logo_path():
    if os.path.exists("logo.jpg"): return "logo.jpg"
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
    if metrics['frames_with_legs'] < 10:
        return [{
            "name": "Gait Pattern Undetermined",
            "desc": "Insufficient body visibility. Please ensure full body (hips to ankles) is in frame.",
            "severity": 0, "confidence": 0,
            "steps": "Reposition camera to capture full body profile."
        }]

    diagnoses = []
    
    # 1. Hemiplegic
    if (metrics['l_knee_rom'] < 30 and metrics['r_knee_rom'] > 50) or (metrics['r_knee_rom'] < 30 and metrics['l_knee_rom'] > 50):
        diagnoses.append({
            "name": "Hemiplegic Pattern", "desc": "Unilateral stiff knee detected with circumduction motion.", "severity": 3, "confidence": 95,
            "steps": "**1. Range of Motion (ROM):** Perform seated knee extensions (3 sets of 10).\n**2. Gait Training:** Practice 'High Knee' marching."
        })

    # 2. Parkinsonian
    if metrics['avg_knee_rom'] < 45 and metrics['arm_swing_magnitude'] < 0.05:
        diagnoses.append({
            "name": "Parkinsonian Gait", "desc": "Shuffling steps and lack of arm swing.", "severity": 2, "confidence": 85,
            "steps": "**1. Visual Cues:** Tape lines on floor to step over.\n**2. Arm Swing Drills:** Use walking poles."
        })

    # 3. Trendelenburg
    if metrics['max_hip_drop'] > 8.0:
        diagnoses.append({
            "name": "Trendelenburg Gait", "desc": "Significant pelvic drop detected (Gluteus Medius weakness).", "severity": 2, "confidence": 80,
            "steps": "**1. Clamshells:** 3 sets of 15 reps.\n**2. Single-Leg Stance:** Balance for 30s."
        })

    # 4. Antalgic
    if metrics['asymmetry_score'] > 15:
        is_secondary = any(d['severity'] == 3 for d in diagnoses)
        if not is_secondary:
            diagnoses.append({
                "name": "Antalgic Gait", "desc": "Asymmetry detected. Patient is favoring one side (Limp).", "severity": 1, "confidence": 75,
                "steps": "**1. Aquatic Therapy:** Pool walking.\n**2. Isometric Holds:** Static wall sits."
            })

    if not diagnoses:
        diagnoses.append({
            "name": "Healthy / Normal Gait", "desc": "No significant deviations detected.", "severity": 0, "confidence": 98,
            "steps": "**1. Maintenance:** Continue current activity levels.\n**2. Strength:** Squats/Lunges 2x/week."
        })
        
    diagnoses.sort(key=lambda x: x['severity'], reverse=True)
    return diagnoses

def draw_overlay(image, live_data, frame_count, w_img):
    h, _, _ = image.shape
    bar_width = int((frame_count % 45) / 45 * w_img)
    cv2.rectangle(image, (0, 0), (w_img, 4), (230, 230, 230), -1)
    cv2.rectangle(image, (0, 0), (bar_width, 4), (26, 115, 232), -1)
    cv2.putText(image, "SYNC STEP AI ENGINE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (26, 115, 232), 1, cv2.LINE_AA)
    
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
        if logo_file: st.image(logo_file, width=90)
        else: st.markdown("### [LOGO]")
    with c2:
        st.title("Sync Step Diagnostics")
        st.markdown("**Automated Biomechanical Assessment System**")

    st.info("""
    **Instructions:**
    * **Side View:** Best for Knee Flexion & Limp detection.
    * **Front View:** Best for Hip Drop & Balance detection.
    * **Note:** Processing happens in the background to ensure smooth playback. Please wait for the progress bar to finish.
    """)
    st.divider()
    
    col_video, col_info = st.columns([2, 1])
    
    if st.session_state['analysis_data'] is None:
        with col_video:
            mode = st.radio("Input Source", ["Live Webcam", "Upload Video"], horizontal=True)
            
            # --- LIVE WEBCAM ---
            if mode == "Live Webcam":
                st.info("System will record for 30s. Please stand back.")
                if st.button("Start Recording", type="primary"):
                    with col_info:
                        st.markdown('<div class="status-box"><h3>🎥 Recording...</h3></div>', unsafe_allow_html=True)
                    
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened(): st.error("Webcam error.")
                    else:
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        st.session_state['processed_video_path'] = temp_file.name
                        w = int(cap.get(3)); h = int(cap.get(4))
                        out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                        
                        start_time = time.time()
                        frame_count = 0
                        stframe = st.empty()
                        metrics_log = {"l_knee_angles": [], "r_knee_angles": [], "hip_drops": [], "frames_with_legs": 0}
                        
                        while (time.time() - start_time) < 30:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_count += 1
                            
                            # PROCESSING
                            h_orig, w_orig, _ = frame.shape
                            ai_frame = cv2.resize(frame, (320, int(h_orig * (320/w_orig))))
                            ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(ai_frame)
                            
                            disp_frame = frame.copy()
                            prev_live_data = {"Status": "Recording"}
                            
                            if results.pose_landmarks:
                                metrics_log['frames_with_legs'] += 1
                                coords = get_coordinates(results.pose_landmarks.landmark)
                                l_knee = calculate_angle(coords['L_Hip'], coords['L_Knee'], coords['L_Ankle'])
                                r_knee = calculate_angle(coords['R_Hip'], coords['R_Knee'], coords['R_Ankle'])
                                hip_drop = abs(coords['L_Hip'][1] - coords['R_Hip'][1]) * 100
                                metrics_log['l_knee_angles'].append(l_knee)
                                metrics_log['r_knee_angles'].append(r_knee)
                                metrics_log['hip_drops'].append(hip_drop)
                                prev_live_data = {"L Flex": f"{int(l_knee)}", "R Flex": f"{int(r_knee)}"}
                                
                                mp_drawing.draw_landmarks(disp_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                            disp_frame = draw_overlay(disp_frame, prev_live_data, frame_count, w_orig)
                            out.write(disp_frame)
                            stframe.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        cap.release()
                        out.release()
                        st.session_state['analysis_data'] = metrics_log
                        st.rerun()

            # --- UPLOAD VIDEO (BATCH PROCESSING FIX) ---
            elif mode == "Upload Video":
                uploaded_file = st.file_uploader("Upload File", type=['mp4','mov'])
                if uploaded_file and st.button("Analyze Upload", type="primary"):
                    
                    # 1. Setup Status UI
                    with col_info:
                        status_box = st.empty()
                        progress_bar = st.progress(0)
                    
                    status_box.markdown('<div class="status-box"><h3>⚙️ Processing...</h3><p>Please wait.</p></div>', unsafe_allow_html=True)
                    
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0: total_frames = 1
                    
                    # Output File
                    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    st.session_state['processed_video_path'] = temp_out.name
                    w = int(cap.get(3)); h = int(cap.get(4))
                    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                    
                    metrics_log = {"l_knee_angles": [], "r_knee_angles": [], "hip_drops": [], "frames_with_legs": 0}
                    frame_idx = 0
                    prev_results = None
                    prev_live_data = {"Status": "Initializing"}

                    # 2. BATCH PROCESS LOOP (No st.image here = Fast!)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_idx += 1
                        
                        # Progress Update (Every 5 frames to save UI redraws)
                        if frame_idx % 5 == 0:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        
                        # Resize for AI (Speed)
                        h_orig, w_orig, _ = frame.shape
                        ai_frame = cv2.resize(frame, (320, int(h_orig * (320/w_orig))))
                        ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                        
                        # Skip Logic
                        if frame_idx % 3 == 0:
                            results = pose.process(ai_frame)
                            prev_results = results
                        else:
                            results = prev_results
                        
                        # High Res Display Frame
                        display_frame = cv2.resize(frame, (640, int(h_orig * (640/w_orig))))
                        h_disp, w_disp, _ = display_frame.shape

                        if results and results.pose_landmarks:
                            if frame_idx % 3 == 0:
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

                        display_frame = draw_overlay(display_frame, prev_live_data, frame_idx, w_disp)
                        
                        # Resize back to original writer dims to prevent corruption
                        final_out = cv2.resize(display_frame, (w, h))
                        out.write(final_out)

                    cap.release()
                    out.release()
                    
                    status_box.success("Processing Complete!")
                    st.session_state['analysis_data'] = metrics_log
                    st.rerun()

    # --- STATE 2: REPORT & REPLAY ---
    else:
        with col_video:
            st.subheader("Session Recording")
            if st.session_state['processed_video_path']:
                st.video(st.session_state['processed_video_path'])
            
            if st.button("Start New Session", type="primary"):
                st.session_state['analysis_data'] = None
                st.session_state['processed_video_path'] = None
                st.rerun()

        with col_info:
            st.subheader("Diagnostic Report")
            metrics = st.session_state['analysis_data']
            
            if metrics and metrics['l_knee_angles']:
                max_l = min(metrics['l_knee_angles']); max_r = min(metrics['r_knee_angles'])
                summary = {
                    'frames_with_legs': metrics['frames_with_legs'],
                    'max_hip_drop': np.max(metrics['hip_drops']),
                    'asymmetry_score': abs(max_l - max_r),
                    'l_knee_rom': 180 - max_l, 'r_knee_rom': 180 - max_r,
                    'min_knee_angle_during_swing': min(max_l, max_r),
                    'arm_swing_magnitude': 0.1, 'avg_knee_rom': (360-max_l-max_r)/2
                }
            else: summary = {'frames_with_legs': 0}

            findings = analyze_gait_pathology(summary)
            primary = findings[0]
            
            if primary['name'] == "Gait Pattern Undetermined":
                st.warning(f"### {primary['name']}")
                st.write(primary['desc'])
            else:
                color = "#0f9d58" if primary['severity'] == 0 else "#d93025"
                st.markdown(f"""
                <div style="border-left: 8px solid {color}; padding: 20px; background: #fff; border-radius: 5px;">
                    <h3 style="color: {color}; margin:0;">PRIMARY DIAGNOSIS:</h3>
                    <h2 style="color: #333; margin:0;">{primary['name']}</h2>
                    <p style="margin-top:10px;">{primary['desc']}</p>
                    <hr><p><strong>Confidence:</strong> {primary['confidence']}%</p>
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
