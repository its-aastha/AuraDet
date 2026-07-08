# AuraDet

AuraDet the AuraDetection ,it is an **AI-powered real-time facial emotion detection system** built using **Python, OpenCV, DeepFace, and Flask**.  
It detects facial expressions through a webcam and displays emotion-based feedback in real time via a web interface.

This project focuses on **student attention, emotional awareness, and intelligent human–computer interaction**, making it suitable for academic projects, hackathons, and AI demonstrations.

---

## Features

-  Real-time webcam video streaming  
-  Facial emotion detection (happy, sad, angry, neutral, fear, surprise, disgust)  
-  Deep learning–based emotion analysis using **DeepFace**  
-  Face detection using **OpenCV DNN (Caffe model)**  
-  Web interface using **Flask + HTML/CSS**  
-  Frame skipping to improve performance  
-  Modular backend & frontend structure  

---

## Problem Statement

Students often lose focus during online study sessions and are unaware of their emotional state.  
Traditional productivity tools **do not adapt to human emotions**.

**AuraDet** solves this by analyzing facial expressions in real time and providing **emotion-aware feedback**, helping users regain focus and self-awareness.

---

## Solution Overview

AuraDet uses:
- **OpenCV DNN** to detect faces from live video  
- **DeepFace** to analyze facial emotions  
- **Flask** to stream live video to a web interface  

The system processes frames efficiently and overlays detected emotions directly on the video feed.

---

## 🗂️ Project Structure

```text
AURADET/
│
├── backend/
│   ├── app.py                                                # Flask server & emotion logic
│   ├── deploy.prototxt                                       # Face detector config
│   ├── res10_300x300_ssd_iter_140000.caffemodel              # Pre-trained model
│   └── test_mediapipe.py                                     # MediaPipe test file
│
├── frontend/
│   ├── index.html                                            # Web UI
│   └── style.css                                             # Styling
│
├── requirements.txt                                          # Dependencies
└── README.md
```

##To run 
cd/backend :- python app.py
