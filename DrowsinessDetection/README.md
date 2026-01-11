# Drowsiness Detection Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Installing `dlib` requires CMake installed on your system.*

2. Download Facial Landmark Model:
   The code relies on `shape_predictor_68_face_landmarks.dat`.
   
   - Download it from: [https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract the `.bz2` file.
   - Place the `.dat` file in this directory (`DrowsinessDetection/`).

3. Run the script:
   ```bash
   python drowsiness_detect.py
   ```
