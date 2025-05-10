# Driver_Drowsiness_Detection

A real-time Driver Monitoring System that uses YOLOv8, dlib, and OpenCV to detect signs of driver fatigue and distraction, such as yawning, drowsiness (eye closure), and looking away from the road. When any such behavior is detected, the system plays a sound alert and saves a screenshot of the event.

🔍 Features
Real-time face and landmark detection using YOLOv8 and dlib

Yawn detection via mouth aspect ratio

Drowsiness detection using Eye Aspect Ratio (EAR)

Distraction detection through head pose estimation

Audio alerts when risky behavior is detected

Automatic screenshots saved during alerts

Live visual feedback with overlay status and debug info

📁 Project Structure
perl
Copy
Edit
driver-monitoring-system/
├── dms_alerts/             # Alert screenshots will be saved here
├── alert_beep.wav          # Sound file for alert (optional)
├── dlib_shape_predictor.dat  # Dlib model file (68 landmarks)
├── main.py                 # Main script (this repo)
└── README.md               # You're here!


🧠 Technologies Used
Python 3.8+

Ultralytics YOLOv8

dlib

OpenCV

pygame

NumPy

HuggingFace Hub

