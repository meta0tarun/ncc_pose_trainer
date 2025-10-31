import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import base64
import av  # ✅ Needed for converting frames
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# ─── App Setup ─────────────────────────────────────────────
st.set_page_config(page_title="NCC Posture Trainer", layout="wide")

# ─── MediaPipe Setup ─────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ─── Utility Functions ───────────────────────────────────────────
def calculate_angle(a, b, c):
    """Returns the angle (in degrees) formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def angle_accuracy(actual, ideal, tolerance=15):
    diff = abs(actual - ideal)
    if diff > tolerance:
        return 0
    return round(100 - (diff / tolerance) * 100, 1)

def play_audio(sound_bytes):
    """Plays a short embedded tone."""
    b64 = base64.b64encode(sound_bytes).decode()
    md = f"""
    <audio autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)

# ─── Embedded Beep Sounds ───────────────────────────────────────
correct_tone = base64.b64decode(
    "UklGRhwAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQwAAAAA//8AAP//AAD//wAA//8AAP//AAD//wAA"
)
wrong_tone = base64.b64decode(
    "UklGRhwAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQwAAAAA//8A////AP///wD///8A////AP///wD///8A"
)

# ─── Page Layout ─────────────────────────────────────────────
st.title("🎖️ NCC Drill Trainer")
st.markdown("""
Choose a drill to begin real-time AI Training.
""")

pose_choice = st.selectbox(
    "Select a Drill to Practice 👇",
    ["Select", "Salute", "Attention", "Stand-at-Ease"]
)

# ─── WebRTC Configuration ───────────────────────────────────────
rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:relay1.expressturn.com:3478"],
         "username": "efYqJb8ZxvM3WkDElO",
         "credential": "A9W6y5P1sH7uGzK2"}
    ]
})

# ─── Salute Pose Detection ───────────────────────────────────────
if pose_choice == "Salute":
    col1, col2 = st.columns(2)
    with col1:
        salute_goal = st.number_input("Enter number of salutes to perform:", min_value=1, value=5, step=1)
    with col2:
        st.markdown("### ⚙️ Detection Flexibility")
        shoulder_flex = st.slider("Shoulder Flexibility (°)", 5, 30, 20)
        elbow_flex = st.slider("Elbow Flexibility (°)", 5, 20, 10)
        wrist_flex = st.slider("Wrist–Eye Proximity", 0.02, 0.1, 0.06, step=0.01)

    st.caption(f"🎯 Shoulder target: {90 - shoulder_flex}° → {90 + shoulder_flex}°")
    st.caption(f"🎯 Elbow target: {55 - elbow_flex}° → {55 + elbow_flex}°")

    # ─── Video Processor ─────────────────────────────────────────
    class SaluteProcessor(VideoProcessorBase):
        def __init__(self):
            self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.salute_count = 0
            self.last_salute_time = 0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                shoulder_angle = calculate_angle(elbow, shoulder, [shoulder[0], shoulder[1]-0.1])

                elbow_acc = angle_accuracy(elbow_angle, 55, elbow_flex)
                shoulder_acc = angle_accuracy(shoulder_angle, 90, shoulder_flex)
                overall_acc = round((elbow_acc + shoulder_acc) / 2, 1)

                elbow_ok = (55 - elbow_flex) <= elbow_angle <= (55 + elbow_flex)
                shoulder_ok = (90 - shoulder_flex) <= shoulder_angle <= (90 + shoulder_flex)
                wrist_near_head = abs(wrist[1] - eye[1]) < wrist_flex

                overlay = img.copy()
                cv2.rectangle(overlay, (10, 20), (340, 170), (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

                cv2.putText(img, f"Elbow: {int(elbow_angle)}°", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(img, f"Shoulder: {int(shoulder_angle)}°", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(img, f"Accuracy: {overall_acc}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if elbow_ok and shoulder_ok and wrist_near_head and overall_acc >= 60:
                    cv2.putText(img, "✅ Correct Salute!", (20, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 3)
                    current_time = time.time()
                    if current_time - self.last_salute_time > 5:
                        self.salute_count += 1
                        self.last_salute_time = current_time
                        play_audio(correct_tone)
                else:
                    cv2.putText(img, "⚠️ Incorrect Form", (20, 145),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(img, f"Salutes: {self.salute_count}/{salute_goal}",
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ─── Stream video feed ─────────────────────────────────────
    webrtc_streamer(
        key="salute",
        mode=WebRtcMode.SENDRECV,  # ✅ Correct mode
        rtc_configuration=rtc_config,
        video_processor_factory=SaluteProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

elif pose_choice == "Attention":
    st.info("🪖 Attention mode coming soon — posture alignment tracking in development.")
elif pose_choice == "Stand-at-Ease":
    st.info("🧍 Stand-at-Ease mode coming soon — body relaxation stability check coming next.")
