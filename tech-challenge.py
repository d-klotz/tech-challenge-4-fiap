import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
from deepface import DeepFace
from tqdm import tqdm
from collections import Counter, defaultdict, deque

class VideoAnalyzer:
    def __init__(self, video_path, output_path, face_images_folder="images"):
        """Initialize the video analyzer with paths and settings."""
        self.video_path = video_path
        self.output_path = output_path
        self.face_images_folder = face_images_folder
        
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load face recognition data
        self.known_face_encodings, self.known_face_names = self.load_face_encodings()
        
        # Counters and trackers for analysis
        self.activities = defaultdict(int)  # Track activities
        self.emotions = defaultdict(lambda: defaultdict(int))  # Track emotions per person
        self.frame_count = 0
        self.person_frames = defaultdict(int)  # Count frames person appears in
        
        # Activity recognition parameters
        self.prev_landmarks = None
        self.activity_window_size = 30  # frames to consider for activity detection
        self.activity_window = deque(maxlen=self.activity_window_size)

        # Cache last emotion inference to avoid calling DeepFace every frame
        self.last_emotion_results = []
    
    def load_face_encodings(self):
        """Load face encodings from the images folder."""
        known_face_encodings = []
        known_face_names = []
        
        if not os.path.exists(self.face_images_folder):
            print(f"Warning: Folder '{self.face_images_folder}' doesn't exist.")
            return known_face_encodings, known_face_names
        
        for filename in os.listdir(self.face_images_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(self.face_images_folder, filename)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Extract name from filename (remove numbers and extension)
                    name = ''.join([c for c in os.path.splitext(filename)[0] if not c.isdigit()])
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded face: {name}")
        
        return known_face_encodings, known_face_names
    
    def detect_activity(self, pose_landmarks):
        """Detect the current activity based on pose landmarks."""
        if not pose_landmarks:
            return "No activity"
        
        landmarks = pose_landmarks  # No need to access .landmark attribute
        
        # Check for arm up pose
        if self.is_arm_up(landmarks):
            return "Raising arm"
        
        # Check if person is sitting
        if self.is_sitting(landmarks):
            return "Sitting"
        
        # Check if person is moving significantly
        if self.prev_landmarks and self.is_significant_movement(landmarks):
            return "Moving"
        
        # Default activity if no specific movement detected
        return "Standing"
    
    def is_arm_up(self, landmarks):
        """Detect if person is raising an arm."""
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Check if wrist and elbow are above corresponding shoulder / eye
        left_arm_up = (left_wrist.y < left_eye.y) and (left_elbow.y < left_shoulder.y)
        right_arm_up = (right_wrist.y < right_eye.y) and (right_elbow.y < right_shoulder.y)
        
        return left_arm_up or right_arm_up
    
    def is_sitting(self, landmarks):
        """Detect if person is sitting based on knee position relative to hip."""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Average positions for robustness
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        vertical_upper_leg = knee_y - hip_y
        vertical_torso = shoulder_y - hip_y if shoulder_y - hip_y != 0 else 1e-5
        ratio = vertical_upper_leg / vertical_torso

        # Sitting if knees are close to hips vertically (small ratio)
        return ratio < 0.3
    
    def is_significant_movement(self, current_landmarks):
        """Detect if there's significant movement between frames."""
        if not self.prev_landmarks:
            return False
        
        movement_score = 0
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW
        ]
        
        # Calculate movement of key points
        for point in key_points:
            curr = current_landmarks[point]  # Current landmarks are directly indexable
            prev = self.prev_landmarks[point]
            
            # Calculate euclidean distance in 2D
            dist = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            movement_score += dist
        
        # Normalize by number of points
        movement_score /= len(key_points)
        
        # Threshold for significant movement
        return movement_score > 0.05
    
    def analyze_video(self):
        """Process the video for face recognition, emotion analysis, and pose detection."""
        # Open the video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up the output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Initialize the pose detector
        with self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7) as pose:
            
            # Prepare MediaPipe face detector (used as fallback)
            face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

            for _ in tqdm(range(total_frames), desc="Processing video"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Make a copy of the frame for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with pose detector
                frame.flags.writeable = False
                pose_results = pose.process(frame_rgb)
                frame.flags.writeable = True
                
                # Store current activity based on pose landmarks
                current_activity = "Unknown"
                if pose_results.pose_landmarks:
                    current_activity = self.detect_activity(pose_results.pose_landmarks.landmark)

                    # Temporal smoothing of activity
                    self.activity_window.append(current_activity)
                    smoothed_activity = Counter(self.activity_window).most_common(1)[0][0]

                    current_activity = smoothed_activity
                    self.activities[smoothed_activity] += 1
                    
                    # Update previous landmarks for next frame comparison
                    self.prev_landmarks = pose_results.pose_landmarks.landmark
                    
                    # Draw pose landmarks on the frame
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Primary face detection with CNN model
                face_locations = face_recognition.face_locations(frame_rgb, model="cnn")

                # Fallback to MediaPipe Face Detection if no faces were found
                if len(face_locations) == 0:
                    detections = face_detection.process(frame_rgb)
                    if detections.detections:
                        for det in detections.detections:
                            bbox = det.location_data.relative_bounding_box
                            x1 = int(bbox.xmin * width)
                            y1 = int(bbox.ymin * height)
                            w = int(bbox.width * width)
                            h = int(bbox.height * height)
                            top = max(y1, 0)
                            left = max(x1, 0)
                            bottom = min(y1 + h, height)
                            right = min(x1 + w, width)
                            face_locations.append((top, right, bottom, left))

                # Encode faces
                face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                
                # Try emotion detection with DeepFace (every 3 frames)
                if self.frame_count % 3 == 0:
                    try:
                        emotion_results = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False, detector_backend='mediapipe')
                        if not isinstance(emotion_results, list):
                            emotion_results = [emotion_results]
                        self.last_emotion_results = emotion_results
                    except Exception:
                        emotion_results = self.last_emotion_results
                else:
                    emotion_results = self.last_emotion_results
                
                # Process recognized faces
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    # Try to match with known faces
                    name = "Unknown"
                    if i < len(face_encodings):
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[i])
                        if True in matches:
                            match_index = matches.index(True)
                            name = self.known_face_names[match_index]
                    
                    # Count this person's appearance
                    self.person_frames[name] += 1
                    
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    # Find matching emotion result if available
                    emotion = "Unknown"
                    for emotion_result in emotion_results:
                        # Check if this emotion result corresponds to this face
                        er_left = emotion_result.get('region', {}).get('x', 0)
                        er_top = emotion_result.get('region', {}).get('y', 0)
                        er_right = er_left + emotion_result.get('region', {}).get('w', 0)
                        er_bottom = er_top + emotion_result.get('region', {}).get('h', 0)
                        
                        # Check overlap between face region and emotion region
                        if (left < er_right and right > er_left and 
                            top < er_bottom and bottom > er_top):
                            emotion = emotion_result.get('dominant_emotion', "Unknown")
                            self.emotions[name][emotion] += 1
                            break
                    
                    # Display name and emotion
                    text = f"{name}: {emotion}, {current_activity}"
                    cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Write frame to output video
                out.write(frame)
            
            # Release resources
            cap.release()
            out.release()
    
    def generate_summary(self):
        """Generate a summary of detected activities and emotions."""
        summary = "Video Analysis Summary\n"
        summary += "=====================\n\n"
        
        # People detected
        summary += "People Detected:\n"
        for person, frames in sorted(self.person_frames.items(), key=lambda x: x[1], reverse=True):
            percentage = (frames / self.frame_count) * 100
            summary += f"- {person}: appeared in {percentage:.1f}% of frames\n"
        summary += "\n"
        
        # Main activities
        summary += "Main Activities Detected:\n"
        total_activities = sum(self.activities.values())
        for activity, count in sorted(self.activities.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_activities) * 100
                summary += f"- {activity}: {percentage:.1f}% of activity time\n"
        summary += "\n"
        
        # Emotions per person
        summary += "Emotions Detected per Person:\n"
        for person, emotions in self.emotions.items():
            if sum(emotions.values()) > 0:
                summary += f"- {person}:\n"
                total_emotions = sum(emotions.values())
                for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_emotions) * 100
                    if percentage > 5:  # Only show significant emotions (>5%)
                        summary += f"  * {emotion}: {percentage:.1f}%\n"
        
        # Write summary to file
        summary_path = os.path.splitext(self.output_path)[0] + "_summary.txt"
        with open(summary_path, 'w') as file:
            file.write(summary)
        
        print(f"Summary saved to {summary_path}")
        return summary

def main():
    """Main function to run the video analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'video.mp4')
    output_video_path = os.path.join(script_dir, 'output_analysis.mp4')
    face_images_folder = os.path.join(script_dir, 'images')
    
    # Create the analyzer
    analyzer = VideoAnalyzer(
        video_path=input_video_path,
        output_path=output_video_path,
        face_images_folder=face_images_folder
    )
    
    # Run the analysis
    print("Starting video analysis...")
    analyzer.analyze_video()
    
    # Generate and display summary
    print("\nGenerating summary...")
    summary = analyzer.generate_summary()
    print("\n" + summary)

if __name__ == "__main__":
    main()
