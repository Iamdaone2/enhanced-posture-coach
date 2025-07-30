from openai import OpenAI
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
from matplotlib.animation import FuncAnimation
import openai
import pyttsx3
import random
import cv2
import mediapipe as mp
import numpy as np
import random
import threading


import time
import math
import simpleaudio as sa

roast_fallbacks = [
    "Straighten up before I call a chiropractor!",
    "Is your spine on vacation or something?",
    "Posture so bad, even your shadow is concerned.",
    "You look like a question mark. Fix that back!",
    "Slouching like it’s your new superpower, huh?",
    "Your back’s sending SOS signals — listen up!",
    "Keep this up and you'll invent a new yoga pose: 'The Fallen Slouch'.",
    "Are you trying to hug yourself all day? Stand tall instead!",
    "Your posture called; it’s filing a complaint.",
    "Chairs everywhere are worried about your bad posture."
]

openai.api_key = os.getenv("OPENAI_API_KEY")

engine = pyttsx3.init()
engine.setProperty('rate', 200)       # increase speaking rate for faster speech
engine.setProperty('volume', 1.0) 
speaking_lock = threading.Lock()

def speak_distance_roast():
    roast = generate_roast(is_posture=False)
    print("💬 GPT Distance Roast:", roast)
    speak_async(roast)

def speak_async(text):
    def run():
        with speaking_lock:
            engine.say(text)
            engine.runAndWait()
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

def generate_roast(is_posture=True, score=None):
    """
    Generate a funny roast based on context.
    If OpenAI API fails or quota is exceeded, pick a random fallback roast.
    """


    if is_posture:
        if score is not None:
            prompt = f"Give me a short, funny roast (under 15 words) about someone with posture score {score}."
        else:
            prompt = "Give me a short, funny roast (under 15 words) about someone with bad posture."
    else:
        prompt = "Give me a short, funny roast (under 15 words) about someone sitting too close to their screen."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sarcastic assistant who roasts people for bad posture."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=40,
            temperature=0.9
        )
        roast_text = response['choices'][0]['message']['content'].strip()
        return roast_text

    except Exception as e:
        print(f"[Warning] OpenAI API call failed: {e}")
        # Return a random fallback roast instead of fixed default string
        return random.choice(roast_fallbacks)


def play_insult(is_posture=True):
    if is_posture:
        prompt = "Give me a short, funny roast (under 15 words) about someone with horrible posture."
    else:
        prompt = "Give me a short, funny roast (under 15 words) about someone sitting too close to their screen."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sarcastic assistant who roasts people for bad posture."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=40,
            temperature=0.9
        )
        insult = response['choices'][0]['message']['content'].strip()
        print("AI-generated insult:", insult)
        engine.say(insult)
        engine.runAndWait()
    except Exception as e:
        print(f"[Warning] Error generating insult: {e}")
        # Instead of crashing, pick a random roast from the fallback list:
        fallback_insult = random.choice(roast_fallbacks)
        engine.say(fallback_insult)
        engine.runAndWait()

def play_alert():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")

# Initialize MediaPipe Pose and Face Mesh models
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

posture_scores = []
posture_score = 100
posture_history = []
history_size = 10

# Face distance variables
face_distance_score = 100
face_distance_history = []
face_distance_history_size = 10
ideal_face_size_ratio = 0.15  
face_too_close = False
last_distance_alert_time = 0
distance_alert_cooldown = 0  # seconds

# Performance tracking
prev_time = 0

# Alert timer settings
last_alert_time = 0
alert_cooldown = 0  # seconds

poor_posture_detected = False
snapshot_taken = False

front_view_mode = True


def start_posture_graph():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_title("Posture Score Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")

    def update(frame):
        ydata = posture_scores[-100:] 
        xdata = list(range(len(ydata)))
        line.set_data(xdata, ydata)
        ax.set_xlim(0, max(100, len(ydata)))
        return line,

    ani = animation.FuncAnimation(fig, update, interval=100)
    plt.show()

graph_thread = threading.Thread(target=start_posture_graph, daemon=True)
graph_thread.start()

def save_snapshot(image):
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"snapshots/snapshot_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"[Snapshot Saved] {filename}")

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    angle_radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_degrees = abs(math.degrees(angle_radians))
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees
    return angle_degrees

def calculate_face_distance_score(face_landmarks, image_shape):
    """Calculate a score based on how close the face is to the screen"""
    global face_distance_history
    
    try:
        if not face_landmarks:
            return face_distance_score
            
        h, w, _ = image_shape
        
        # Get face bounding box
        x_coordinates = [landmark.x for landmark in face_landmarks]
        y_coordinates = [landmark.y for landmark in face_landmarks]
        
        # Calculate face width and height relative to frame
        face_width = max(x_coordinates) - min(x_coordinates)
        face_height = max(y_coordinates) - min(y_coordinates)
        
        # Use primarily the width for distance estimation
        face_width_ratio = face_width
        
        # If face is too big (too close), score decreases
        if face_width_ratio < ideal_face_size_ratio: 
            distance_score = min(100, 100 * (face_width_ratio / ideal_face_size_ratio))
        else: 
            distance_score = max(0, 100 - 200 * (face_width_ratio - ideal_face_size_ratio))
        
        face_distance_history.append(distance_score)
        if len(face_distance_history) > face_distance_history_size:
            face_distance_history.pop(0)
            
        return int(sum(face_distance_history) / len(face_distance_history))
        
    except Exception as e:
        print(f"Error calculating face distance: {e}")
        return face_distance_score

def calculate_posture_score_front_view(landmarks):
    """Calculate posture score based on front view landmarks"""
    global posture_history
    try:
        # Get landmark coordinates
        ls = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        rs = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        le = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y])
        re = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y])
        lh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        rh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        # Calculate midpoints
        mid_shoulder = (ls + rs) / 2
        mid_ear = (le + re) / 2
        mid_hip = (lh + rh) / 2

        # 1. Shoulder alignment (horizontal)
        shoulder_alignment = abs(ls[1] - rs[1])
        s_score = max(0, 100 - (shoulder_alignment * 500))
        
        # 2. Vertical alignment (ear above shoulders)
        vertical_alignment = abs(mid_ear[0] - mid_shoulder[0])
        v_score = max(0, 100 - (vertical_alignment * 500))
        
        # 3. Spine straightness (shoulders above hips)
        spine_straightness = abs(mid_shoulder[0] - mid_hip[0])
        sp_score = max(0, 100 - (spine_straightness * 500))

        # Calculate final score with weights
        final_score = int(s_score * 0.3 + v_score * 0.4 + sp_score * 0.3)

        posture_history.append(final_score)
        if len(posture_history) > history_size:
            posture_history.pop(0)
        
        return int(sum(posture_history) / len(posture_history))

    except Exception as e:
        print(f"Error calculating front view posture: {e}")
        return posture_score


def calculate_posture_score_side_view(landmarks):
    """Calculate posture score based on side view landmarks"""
    global posture_history
    try:
        # Determine which side is more visible to the camera
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Use the most visible side (higher visibility score)
        use_left = (left_ear.visibility + left_shoulder.visibility + left_hip.visibility)/3 > \
                  (right_ear.visibility + right_shoulder.visibility + right_hip.visibility)/3
        
        # Extract coordinates based on the most visible side
        if use_left:
            ear = np.array([left_ear.x, left_ear.y])
            shoulder = np.array([left_shoulder.x, left_shoulder.y])
            hip = np.array([left_hip.x, left_hip.y])
            ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        else:
            ear = np.array([right_ear.x, right_ear.y])
            shoulder = np.array([right_shoulder.x, right_shoulder.y])
            hip = np.array([right_hip.x, right_hip.y])
            ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        
        # 1. Ear-Shoulder Alignment (vertical alignment)
        ear_shoulder_alignment = abs(ear[0] - shoulder[0])
        ear_shoulder_score = max(0, 100 - (ear_shoulder_alignment * 500))
        
        # 2. Shoulder-Hip Alignment (vertical alignment)
        shoulder_hip_alignment = abs(shoulder[0] - hip[0])
        shoulder_hip_score = max(0, 100 - (shoulder_hip_alignment * 500))
        
        # 3. Overall vertical alignment (ear, shoulder, hip, ankle)
        ideal_x = (ear[0] + shoulder[0] + hip[0] + ankle[0]) / 4  # Average x position
        ear_deviation = abs(ear[0] - ideal_x)
        shoulder_deviation = abs(shoulder[0] - ideal_x)
        hip_deviation = abs(hip[0] - ideal_x)
        ankle_deviation = abs(ankle[0] - ideal_x)
        
        avg_deviation = (ear_deviation + shoulder_deviation + hip_deviation + ankle_deviation) / 4
        vertical_score = max(0, 100 - (avg_deviation * 400))
        
        final_score = int(ear_shoulder_score * 0.4 + shoulder_hip_score * 0.3 + vertical_score * 0.3)
        
        # Smoothing
        posture_history.append(final_score)
        if len(posture_history) > history_size:
            posture_history.pop(0)
        
        return int(sum(posture_history) / len(posture_history))
    
    except Exception as e:
        print(f"Error calculating side view posture: {e}")
        return posture_score

def get_feedback(score):
    """Get posture feedback text and color based on score"""
    if score >= 90:
        return "Excellent posture!", (0, 255, 0)
    elif score >= 70:
        return "Good posture", (0, 255, 255)
    elif score >= 50:
        return "Adjust your posture", (0, 165, 255)
    else:
        return "Poor posture! Sit up straight", (0, 0, 255)

def get_distance_feedback(score):
    """Get screen distance feedback text and color based on score"""
    if score >= 90:
        return "Perfect distance from screen", (0, 255, 0)
    elif score >= 70:
        return "Good viewing distance", (0, 255, 255)
    elif score >= 50:
        return "Consider adjusting distance", (0, 165, 255)
    else:
        return "Move back from screen!", (0, 0, 255)

def draw_posture_front_view(image, landmarks):
    """Draw helpful indicators for front view posture"""
    if landmarks:
        h, w, c = image.shape
        
        # Get key points
        ls = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        rs = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        le = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
        re = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))
        lh = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        rh = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
              int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
        
        # Draw shoulder line
        cv2.line(image, ls, rs, (0, 255, 0), 3)
        
        # Draw spine line 
        mid_shoulder = ((ls[0] + rs[0])//2, (ls[1] + rs[1])//2)
        mid_hip = ((lh[0] + rh[0])//2, (lh[1] + rh[1])//2)
        cv2.line(image, mid_shoulder, mid_hip, (0, 255, 0), 3)
        
        # Draw ear to shoulder lines
        cv2.line(image, le, ls, (0, 255, 0), 2)
        cv2.line(image, re, rs, (0, 255, 0), 2)
        
        # Draw vertical center line as reference
        center_x = w // 2
        cv2.line(image, (center_x, 0), (center_x, h), (200, 200, 200), 1)
        
        # Draw circles at key points
        cv2.circle(image, ls, 8, (0, 0, 255), -1)
        cv2.circle(image, rs, 8, (0, 0, 255), -1)
        cv2.circle(image, le, 8, (0, 0, 255), -1)
        cv2.circle(image, re, 8, (0, 0, 255), -1)
        cv2.circle(image, lh, 8, (0, 0, 255), -1)
        cv2.circle(image, rh, 8, (0, 0, 255), -1)

def draw_posture_side_view(image, landmarks):
    """Draw helpful indicators for side view posture"""
    if landmarks:
        h, w, c = image.shape
        
        # Determine which side is more visible
        left_vis = (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility + 
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility + 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility) / 3
        
        right_vis = (landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility + 
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility + 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility) / 3
        
        use_left = left_vis > right_vis
        
        # Get landmarks of the most visible side
        if use_left:
            ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
            shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), 
                       int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
            hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
            ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h))
            side_text = "Left Side View"
        else:
            ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))
            shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), 
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
            hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w), 
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
            ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w), 
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h))
            side_text = "Right Side View"
        
        # Calculate ideal vertical line (average x position of all points)
        ideal_x = (ear[0] + shoulder[0] + hip[0] + ankle[0]) // 4
        
        # Draw ideal vertical alignment line
        cv2.line(image, (ideal_x, 0), (ideal_x, h), (200, 200, 200), 1)
        
        # Draw additional vertical reference line specifically for ear-shoulder alignment
        ear_shoulder_line_x = (ear[0] + shoulder[0]) // 2
        cv2.line(image, (ear_shoulder_line_x, ear[1]), (ear_shoulder_line_x, shoulder[1]), 
                (0, 255, 255), 2)  # Yellow for ear-shoulder reference
        
        # Draw postural alignment lines
        cv2.line(image, ear, shoulder, (0, 255, 0), 3)  # Head to shoulder
        cv2.line(image, shoulder, hip, (0, 255, 0), 3)  # Shoulder to hip
        cv2.line(image, hip, ankle, (0, 255, 0), 3)     # Hip to ankle
        
        # Draw circles at key points
        cv2.circle(image, ear, 8, (0, 0, 255), -1)
        cv2.circle(image, shoulder, 8, (0, 0, 255), -1)
        cv2.circle(image, hip, 8, (0, 0, 255), -1)
        cv2.circle(image, ankle, 8, (0, 0, 255), -1)
        
        # Calculate deviations from ideal vertical line
        ear_dev = abs(ear[0] - ideal_x)
        shoulder_dev = abs(shoulder[0] - ideal_x)
        hip_dev = abs(hip[0] - ideal_x)
        ankle_dev = abs(ankle[0] - ideal_x)
        
        # Display ear-shoulder alignment information
        ear_shoulder_alignment = abs(ear[0] - shoulder[0])
        alignment_text = "Ear-Shoulder Aligned" if ear_shoulder_alignment < 20 else "Align Ear with Shoulder"
        alignment_color = (0, 255, 0) if ear_shoulder_alignment < 20 else (0, 165, 255)
        cv2.putText(image, alignment_text, (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, alignment_color, 2)
        
        cv2.putText(image, side_text, (w//2 - 80, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_face_distance_indicators(image, face_landmarks):
    """Draw helpful indicators for face distance"""
    if not face_landmarks:
        return
        
    h, w, c = image.shape
    
    #face bounding box
    x_coordinates = [landmark.x for landmark in face_landmarks]
    y_coordinates = [landmark.y for landmark in face_landmarks]
    
    min_x, max_x = min(x_coordinates), max(x_coordinates)
    min_y, max_y = min(y_coordinates), max(y_coordinates)
    
    # Convert to pixel coordinates
    face_left = int(min_x * w)
    face_right = int(max_x * w)
    face_top = int(min_y * h)
    face_bottom = int(max_y * h)
    
    # Draw face bounding box
    cv2.rectangle(image, (face_left, face_top), (face_right, face_bottom), (0, 255, 255), 2)
    
    # Draw ideal face size reference
    frame_width = w
    ideal_face_width = int(ideal_face_size_ratio * frame_width)
    
    # Show ideal face width with reference lines
    center_x = w // 2
    face_y = (face_top + face_bottom) // 2
    left_ideal = center_x - ideal_face_width // 2
    right_ideal = center_x + ideal_face_width // 2
    
    # Draw ideal face width reference
    cv2.line(image, (left_ideal, face_y), (right_ideal, face_y), (0, 255, 0), 2)
    cv2.line(image, (left_ideal, face_y - 10), (left_ideal, face_y + 10), (0, 255, 0), 2)
    cv2.line(image, (right_ideal, face_y - 10), (right_ideal, face_y + 10), (0, 255, 0), 2)

print("Starting Enhanced Posture Coach. Press 'q' to quit.")
print("Press SPACE to toggle between Front View and Side View modes.")
graph_thread = threading.Thread(target=start_posture_graph, daemon=True)
graph_thread.start()

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read frame.")
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pose_results = pose.process(rgb)
        
        face_results = face_mesh.process(rgb)
        
        current_poor_posture = False
        face_landmarks = None

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if front_view_mode:
                posture_score = calculate_posture_score_front_view(pose_results.pose_landmarks.landmark)
                draw_posture_front_view(image, pose_results.pose_landmarks.landmark)
            else:
                posture_score = calculate_posture_score_side_view(pose_results.pose_landmarks.landmark)
                draw_posture_side_view(image, pose_results.pose_landmarks.landmark)

            posture_scores.append(posture_score)
            if len(posture_scores) > 300:
                posture_scores.pop(0)
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            face_distance_score = calculate_face_distance_score(face_landmarks, image.shape)
            draw_face_distance_indicators(image, face_landmarks)
                
        alert_threshold = 60
        snapshot_threshold = 50
        current_alert_posture = posture_score < alert_threshold 
        current_poor_posture = posture_score < snapshot_threshold

        distance_alert_threshold = 45
        current_face_too_close = face_distance_score < distance_alert_threshold
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        posture_feedback, posture_color = get_feedback(posture_score)
        distance_feedback, distance_color = get_distance_feedback(face_distance_score)
        
        if current_alert_posture:
            if (current_time - last_alert_time) > alert_cooldown:
                # Play alert sound (keep your thread for async sound if you want)
                sound_thread = threading.Thread(target=play_alert)
                sound_thread.daemon = True
                sound_thread.start()

                # Generate and print the roast from GPT
                roast = generate_roast(is_posture=True, score=posture_score)

                print("💬 GPT Roast:", roast)

                # Optionally, speak it out loud using your pyttsx3 engine (if you want)
                speak_async(roast)

                last_alert_time = current_time
        
        if current_face_too_close:
            if (current_time - last_distance_alert_time) > distance_alert_cooldown:
                sound_thread = threading.Thread(target=play_alert)
                sound_thread.daemon = True
                roast_thread = threading.Thread(target=speak_distance_roast)
                roast_thread.daemon = True
                roast_thread.start()
                last_distance_alert_time = current_time
                
        if current_poor_posture:
            if not poor_posture_detected:
                poor_posture_detected = True
                snapshot_taken = False
            
            if not snapshot_taken:
                save_snapshot(image)
                snapshot_taken = True
        else:
            poor_posture_detected = False

        cv2.putText(image, f"Posture: {posture_score}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)
        cv2.putText(image, posture_feedback, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, posture_color, 2)
        
        cv2.putText(image, f"Screen Distance: {face_distance_score}%", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, distance_color, 2)
        cv2.putText(image, distance_feedback, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, distance_color, 2)
                    
        cv2.putText(image, f"FPS: {int(fps)}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        mode_text = "FRONT VIEW MODE" if front_view_mode else "SIDE VIEW MODE"
        cv2.putText(image, mode_text, (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        h, w, _ = image.shape
        cv2.putText(image, "Press SPACE to switch view mode", (w//2 - 150, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Enhanced Posture Coach', image)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  
            front_view_mode = not front_view_mode
            print(f"Switched to {'FRONT' if front_view_mode else 'SIDE'} view mode")
            posture_history = []

finally:
    cap.release()
    pose.close()
    face_mesh.close()
    cv2.destroyAllWindows()
