import cv2
import numpy as np
import dlib
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import time
import datetime
from threading import Thread
import pygame
from scipy.spatial import distance as dist

# Initialize pygame for sound
pygame.init()
pygame.mixer.init()

# ====== CONFIGURATION ======
# Update these paths as needed
DLIB_MODEL_PATH = r************************************"  # Replace with your dlib model path
HF_TOKEN = "**********************************"  # Replace with your Hugging Face token
ALERTS_FOLDER = os.path.join(os.path.dirname(__file__), "dms_alerts")

# Verify all required files exist
def verify_files():
    required_files = {
        "dlib model": DLIB_MODEL_PATH,
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at: {path}")

    # Create alerts folder if needed
    os.makedirs(ALERTS_FOLDER, exist_ok=True)

# ====== INITIALIZATION ======
try:
    verify_files()
    
    # Initialize sound
    try:
        alert_sound = pygame.mixer.Sound("alert_beep.wav")
        print("Loaded alert sound file successfully")
    except Exception as e:
        print(f"Couldn't load sound file: {e}")
        def play_alert():
            duration = 1  # seconds
            freq = 440  # Hz
            os.system(f'powershell -c (New-Object System.Media.SoundPlayer).PlaySync(); [console]::beep({freq},{duration * 1000})')
    
    # ====== DETECTION PARAMETERS ======
    YAWN_THRESHOLD = 0.37
    YAWN_CONSEC_FRAMES = 10
    MOUTH_ROI_RELATIVE = [0.3, 0.6, 0.7, 0.9]
    
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 20
    
    LOOK_LEFT_THRESHOLD = -72
    LOOK_RIGHT_THRESHOLD = 72
    NEUTRAL_ZONE = 20
    ALERT_FRAMES_REQUIRED = 35
    SMOOTHING_WINDOW = 20
    
    ALERT_DURATION = 3

except Exception as e:
    print(f"Initialization failed: {e}")
    exit(1)

# ====== FUNCTIONS ======
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_yawn(face_roi):
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_roi = clahe.apply(gray_roi)
    
    thresh = cv2.adaptiveThreshold(
        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    height, width = face_roi.shape[:2]
    x1 = int(MOUTH_ROI_RELATIVE[0] * width)
    y1 = int(MOUTH_ROI_RELATIVE[1] * height)
    x2 = int(MOUTH_ROI_RELATIVE[2] * width)
    y2 = int(MOUTH_ROI_RELATIVE[3] * height)
    mouth_roi = thresh[y1:y2, x1:x2]
    
    try:
        ratio = cv2.countNonZero(mouth_roi) / (mouth_roi.shape[0] * mouth_roi.shape[1])
        cv2.rectangle(face_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return ratio, mouth_roi
    except:
        return 0, None

def play_alert_sound():
    try:
        alert_sound.play()
        pygame.time.wait(1000)
    except:
        play_alert()

def save_alert_screenshot(frame, alert_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(ALERTS_FOLDER, f"{alert_type}_alert_{timestamp}.jpg")
    cv2.imwrite(screenshot_path, frame)
    print(f"ALERT: Screenshot saved to {screenshot_path}")

def estimate_head_pose(landmarks, frame_width):
    nose_tip = landmarks[30]
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    face_center = (left_eye[0] + right_eye[0]) // 2
    return (face_center - frame_width//2) / (frame_width//2) * 100

# ====== MAIN FUNCTION ======
def main():
    print("Initializing driver monitoring system...")
    
    try:
        # Initialize models
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_MODEL_PATH)
        print("Loaded dlib facial landmark detector")
        
        face_model = YOLO(hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt",
            use_auth_token=HF_TOKEN
        ))
        print("Loaded YOLO face detection model")
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video capture")
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        # State variables
        yawn_counter = drowsy_counter = head_turn_counter = 0
        yaw_history = []
        alert_active = False
        alert_start_time = 0
        alert_type = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            alert_frame = frame.copy()
            driver_status = "Awake"
            gaze_status = "Looking Forward"
            ear = 0.3
            yaw = 0
            mouth_ratio = 0
            
            # Face detection
            results = face_model(frame)
            if results and len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size == 0:
                        continue
                    
                    # Facial landmarks
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    landmarks = predictor(gray, dlib.rectangle(0, 0, *gray.shape[::-1]))
                    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                    
                    # Yawn detection
                    mouth_ratio, _ = detect_yawn(face_roi)
                    if mouth_ratio > YAWN_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_CONSEC_FRAMES:
                            driver_status = "Yawning"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "YAWNING DETECTED", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            if not alert_active:
                                alert_active = True
                                alert_type = "yawn"
                                alert_start_time = time.time()
                                Thread(target=play_alert_sound).start()
                                save_alert_screenshot(frame, "yawn")
                    else:
                        yawn_counter = max(0, yawn_counter - 1)
                    
                    # Sleep detection
                    left_ear = eye_aspect_ratio(landmarks[36:42])
                    right_ear = eye_aspect_ratio(landmarks[42:48])
                    ear = (left_ear + right_ear) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        drowsy_counter += 1
                        if drowsy_counter >= EYE_AR_CONSEC_FRAMES:
                            driver_status = "Sleep"
                            if not alert_active:
                                alert_active = True
                                alert_type = "sleep"
                                alert_start_time = time.time()
                                Thread(target=play_alert_sound).start()
                                save_alert_screenshot(frame, "sleep")
                    else:
                        drowsy_counter = 0
                    
                    # Head pose
                    yaw = estimate_head_pose(landmarks, frame_width)
                    yaw_history.append(yaw)
                    if len(yaw_history) > SMOOTHING_WINDOW:
                        yaw_history.pop(0)
                    smooth_yaw = np.mean(yaw_history)
                    
                    if smooth_yaw < LOOK_LEFT_THRESHOLD:
                        head_turn_counter += 1
                        if head_turn_counter > ALERT_FRAMES_REQUIRED:
                            gaze_status = "Looking Left!"
                            if not alert_active:
                                alert_active = True
                                alert_type = "head_turn"
                                alert_start_time = time.time()
                                Thread(target=play_alert_sound).start()
                                save_alert_screenshot(frame, "head_turn")
                    elif smooth_yaw > LOOK_RIGHT_THRESHOLD:
                        head_turn_counter += 1
                        if head_turn_counter > ALERT_FRAMES_REQUIRED:
                            gaze_status = "Looking Right!"
                            if not alert_active:
                                alert_active = True
                                alert_type = "head_turn"
                                alert_start_time = time.time()
                                Thread(target=play_alert_sound).start()
                                save_alert_screenshot(frame, "head_turn")
                    elif abs(smooth_yaw) <= NEUTRAL_ZONE:
                        head_turn_counter = max(0, head_turn_counter - 2)
                    else:
                        head_turn_counter = max(0, head_turn_counter - 1)
                    
                    # Visual feedback
                    head_pose_x = int(frame_width/2 + (frame_width/4 * (smooth_yaw/100)))
                    cv2.line(frame, (frame_width//2, frame_height-50), 
                            (head_pose_x, frame_height-50), (0,255,0), 4)
                    
                    for (x, y) in landmarks:
                        cv2.circle(frame, (x1 + x, y1 + y), 1, (0, 255, 255), -1)
            
            # Display info
            cv2.putText(frame, f"Driver Status: {driver_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Gaze Direction: {gaze_status}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Alert overlay
            if alert_active:
                overlay = np.zeros_like(alert_frame)
                overlay[:] = (0, 0, 200)
                
                if alert_type == "yawn":
                    alert_text = "DROWSINESS ALERT! Yawning detected"
                elif alert_type == "sleep":
                    alert_text = "DROWSINESS ALERT! Driver is sleeping"
                else:
                    alert_text = f"ALERT! {gaze_status}"
                
                cv2.putText(overlay, alert_text, 
                           (frame.shape[1]//2 - 200, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                if time.time() - alert_start_time > ALERT_DURATION:
                    alert_active = False
                    yawn_counter = drowsy_counter = head_turn_counter = 0
            
            cv2.imshow('Driver Monitoring System', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()