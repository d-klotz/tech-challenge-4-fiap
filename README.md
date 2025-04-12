# Video Analysis System

This system analyzes a video file to detect faces, emotions, activities, and generate a summary. It addresses four key tasks:

1. **Facial Recognition**: Identifies and marks faces in the video
2. **Emotional Analysis**: Analyzes the emotional expressions of identified faces
3. **Activity Detection**: Detects and categorizes activities performed by people in the video
4. **Summary Generation**: Creates an automatic summary of the main activities and emotions detected

## Prerequisites

The application requires the following Python packages:

```bash
pip install opencv-python
pip install mediapipe
pip install face_recognition
pip install deepface
pip install numpy
pip install tqdm
```

Note: Installing `face_recognition` may require additional system dependencies:
- On macOS: `brew install cmake`
- On Linux: `sudo apt-get install -y cmake`
- Windows users may need to install Visual C++ build tools

## Usage

1. Place the video file to be analyzed as `video.mp4` in the root directory
2. Ensure you have face images for recognition in the `images` folder
3. Run the analysis:

```bash
python tech-challenge.py
```

4. The system will generate:
   - An annotated video file: `output_analysis.mp4`
   - A text summary: `output_analysis_summary.txt`

## How It Works

The system uses:
- **MediaPipe** for pose estimation and activity detection
- **face_recognition** library for facial recognition
- **DeepFace** for emotion analysis

The `VideoAnalyzer` class performs the following steps:
1. Loads face encodings from reference images
2. Processes each frame of the video to detect poses and faces
3. Identifies activities based on pose landmarks
4. Recognizes faces and analyzes emotions
5. Generates a comprehensive summary

## Face Recognition

The system expects images of people to be recognized in the `images` folder. The filename should include the person's name (e.g., `john1.png`, `john2.png`).

## Activity Recognition

The system can detect the following activities:
- Raising arm
- Sitting
- Moving
- Standing

## Output Format

The summary includes statistics about:
- People detected in the video and their screen time
- Main activities detected and their duration
- Emotions displayed by each person 