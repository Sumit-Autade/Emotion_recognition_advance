from flask import Flask, render_template, request, jsonify, Response, render_template_string, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time, json, base64, os
from fer import FER
import face_recognition
from pymongo import MongoClient
from bson.binary import Binary
from bson import ObjectId
import pickle
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# --------------------------
# MongoDB Initialization
# --------------------------
client = MongoClient("mongodb+srv://sumit:sumit@cluster0.gvjdh.mongodb.net/test")
# Although the URI uses "test", the database used is explicitly set below.
db = client["face_recognition_db"]
users_collection = db["users"]
sessions_collection = db["sessions"]
logs_collection = db["logs"]  # Log collection for storing warnings and errors

# --------------------------
# Global Initializations
# --------------------------

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Initialize FER for emotion detection
emotion_detector = FER()

# --------------------------
# Warning Logging Helper
# --------------------------
def log_warning(message):
    logs_collection.insert_one({
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "type": "warning",
        "message": message
    })

# Global dictionaries to track the last time a valid head moment and face presence were detected per session.
last_head_moment = {}    # Key: session_id, Value: timestamp of last valid head moment
last_face_presence = {}  # Key: session_id, Value: timestamp of last face detection

# --------------------------
# Helper Functions for Head Pose Estimation
# --------------------------
LANDMARK_INDICES = {
    'nose_tip': 4,
    'chin': 152,
    'left_eye': 33, 
    'right_eye': 263,
    'left_mouth': 61,
    'right_mouth': 291
}

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float64)

def get_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])

def determine_direction(pitch, yaw, threshold=10):
    if pitch < threshold and yaw < threshold:
        return ["Facing the Camera"]
    directions = []
    if pitch > threshold:
        directions.append("Up")
    elif pitch < -threshold:
        directions.append("Down")
    if yaw > threshold:
        directions.append("Left")
    elif yaw < -threshold:
        directions.append("Right")
    return directions

# --------------------------
# Eye & Iris Tracking and Lip Status Functions
# --------------------------
LEFT_EYE = [33, 133, 160, 144, 158, 153]
RIGHT_EYE = [362, 263, 385, 380, 387, 373]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [472, 473, 474, 475]

def check_eye_open_status(face_landmarks, side, rows):
    if side == "left":
        upper = face_landmarks.landmark[159]
        lower = face_landmarks.landmark[145]
    else:
        upper = face_landmarks.landmark[386]
        lower = face_landmarks.landmark[374]
    vertical_distance = abs(upper.y - lower.y) * rows
    threshold = 10  # Adjust as needed
    return "Closed" if vertical_distance < threshold else "Open"

def get_gaze_direction(face_landmarks, side, image_width, image_height):
    if side == "left":
        iris_indices = LEFT_IRIS
        eye_indices = LEFT_EYE
    else:
        iris_indices = RIGHT_IRIS
        eye_indices = RIGHT_EYE

    iris_x = iris_y = 0
    for idx in iris_indices:
        landmark = face_landmarks.landmark[idx]
        iris_x += landmark.x * image_width
        iris_y += landmark.y * image_height
    iris_x /= len(iris_indices)
    iris_y /= len(iris_indices)
    
    xs = [face_landmarks.landmark[idx].x * image_width for idx in eye_indices]
    if not xs:
        return "Unknown"
    min_x, max_x = min(xs), max(xs)
    
    ratio = (iris_x - min_x) / (max_x - min_x + 1e-6)
    if ratio < 0.4:
        gaze = "Right"
    elif ratio > 0.6:
        gaze = "Left"
    else:
        gaze = "Center"
    return gaze

def get_eye_status_and_gaze(face_landmarks, side, image_width, image_height):
    open_status = check_eye_open_status(face_landmarks, side, image_height)
    if open_status == "Closed":
        return "Closed"
    else:
        gaze = get_gaze_direction(face_landmarks, side, image_width, image_height)
        return f"Open, Gaze: {gaze}"

def get_lip_status(face_landmarks, image_height):
    """
    Determines lip status based on the vertical distance between the upper (landmark 13)
    and lower (landmark 14) lips. Adjust the threshold value if needed.
    """
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    vertical_distance = abs(upper_lip.y - lower_lip.y) * image_height
    threshold = 10  # Adjust this threshold as needed
    return "Open" if vertical_distance > threshold else "Closed"

# --------------------------
# Frame Processing Function
# --------------------------
def process_frame(frame):
    """
    Process a single frame:
      - Extracts facial landmarks via MediaPipe.
      - Estimates head pose (yaw, pitch, roll).
      - Detects emotion via FER.
      - Determines eye status (and gaze direction) and lip status.
      - Performs face recognition using face_recognition.
    Returns a dictionary with the analysis data.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    output = {}
    output['error'] = None
    output['analysis'] = {}
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        # Compute head pose
        img_points = []
        for key in LANDMARK_INDICES:
            idx = LANDMARK_INDICES[key]
            point = face_landmarks.landmark[idx]
            img_points.append([point.x * w, point.y * h])
        img_points = np.array(img_points, dtype=np.float64)
        focal_length_current = w
        cam_matrix = np.array([
            [focal_length_current, 0, w/2],
            [0, focal_length_current, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        success_pnp, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, img_points, cam_matrix, np.zeros((4,1)),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success_pnp:
            angles = get_euler_angles(rvec)
            pitch_val, yaw_val, roll_val = angles
            directions = determine_direction(pitch_val, yaw_val, threshold=10)
            head_state = ", ".join(directions)
        else:
            head_state = "Unknown"
            pitch_val, yaw_val, roll_val = 0, 0, 0

        # Emotion detection using FER
        emotion_text = "None"
        try:
            emotions = emotion_detector.detect_emotions(frame)
            for face in emotions:
                if "box" in face:
                    emotion = max(face["emotions"], key=face["emotions"].get)
                    emotion_text = emotion
                    break
        except Exception as e:
            log_warning("Error detecting emotion: " + str(e))
            emotion_text = "Error detecting emotion"
        
        # Eye tracking
        left_eye_status = get_eye_status_and_gaze(face_landmarks, "left", w, h)
        right_eye_status = get_eye_status_and_gaze(face_landmarks, "right", w, h)
        
        # Lip tracking
        lip_status = get_lip_status(face_landmarks, h)
        
        # Face recognition using face_recognition
        recognition_name = "No Face"
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(face_locations) == 1 and face_encodings:
                face_encoding = face_encodings[0]
                # Compare with users that have stored face encodings.
                users = users_collection.find({"face_encoding": {"$exists": True}})
                known_face_encodings = []
                known_face_names = []
                for user in users:
                    try:
                        encoding = pickle.loads(user["face_encoding"])
                        known_face_encodings.append(encoding)
                        known_face_names.append(user["name"])
                    except Exception as e:
                        log_warning("Error loading face encoding for user {}: {}".format(user.get("username", "unknown"), str(e)))
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.45
                )
                if True in matches:
                    first_match_index = matches.index(True)
                    recognition_name = known_face_names[first_match_index]
                else:
                    recognition_name = "Unknown"
            elif len(face_locations) > 1:
                recognition_name = "Multiple Faces"
            else:
                recognition_name = "No Face"
        except Exception as e:
            log_warning("Error in recognition: " + str(e))
            recognition_name = "Error in recognition"
        
        output['analysis'] = {
            "yaw": float(f"{yaw_val:.2f}"),
            "pitch": float(f"{pitch_val:.2f}"),
            "roll": float(f"{roll_val:.2f}"),
            "head_state": head_state,
            "emotion": emotion_text,
            "left_eye": left_eye_status,
            "right_eye": right_eye_status,
            "lip_status": lip_status,
            "recognition": recognition_name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
    else:
        output['analysis'] = {"error": "No face detected"}
    return output

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def index():
    # Only logged-in users can access the main page.
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user_id=session['user_id'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            # Save user ID temporarily for face verification.
            session['temp_user_id'] = str(user['_id'])
            return redirect(url_for('face_verification'))
        return jsonify({"error": "Invalid credentials"}), 400
    return render_template('login_html.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        password = data.get('password')
        name = data.get('name')
        
        if not username or not password or not name:
            return jsonify({"error": "Username, password and name are required"}), 400
        
        if users_collection.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 400
        
        hashed_password = generate_password_hash(password)
        inserted = users_collection.insert_one({
            "username": username,
            "password": hashed_password,
            "name": name
        })
        # Store the new user's ID temporarily so we can capture their face.
        session['temp_user_id'] = str(inserted.inserted_id)
        return redirect(url_for('capture_face'))
    return render_template('signup_html.html')

@app.route('/capture_face', methods=['GET', 'POST'])
def capture_face():
    """
    This route is used after signup to capture multiple images of the user's face.
    The server computes face encodings from each image, averages them, and stores
    the robust embedding in the database.
    """
    if request.method == 'POST':
        try:
            data = request.get_json()
            images = data.get('images', [])
            if not images:
                return jsonify({"error": "No images provided"}), 400
            
            encodings_list = []
            for img_data in images:
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if len(face_encodings) == 1:
                    encodings_list.append(face_encodings[0])
            
            if not encodings_list:
                return jsonify({"error": "Could not capture valid face data. Please try again."}), 400
            
            avg_encoding = np.mean(np.array(encodings_list), axis=0)
            temp_user_id = session.get('temp_user_id')
            if not temp_user_id:
                return jsonify({"error": "No temporary user found in session"}), 400
            users_collection.update_one(
                {"_id": ObjectId(temp_user_id)},
                {"$set": {"face_encoding": pickle.dumps(avg_encoding)}}
            )
            # Remove the temporary user ID once the embedding is stored.
            session.pop('temp_user_id', None)
            return jsonify({"success": True})
        except Exception as e:
            log_warning("Error during face capture: " + str(e))
            return jsonify({"error": str(e)}), 500
    return render_template('capture_face_html.html')

@app.route('/face_verification', methods=['GET', 'POST'])
def face_verification():
    """
    After entering login credentials, the user must verify via a face scan.
    The captured image is compared with the stored embedding.
    """
    if request.method == 'POST':
        try:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "No image provided"}), 400
            img_data = data['image']
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Could not decode image"}), 400

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(face_encodings) != 1:
                return jsonify({"error": "Please ensure exactly one face is visible"}), 400
            captured_encoding = face_encodings[0]

            temp_user_id = session.get('temp_user_id')
            if not temp_user_id:
                return jsonify({"error": "No temporary user found in session"}), 400
            user = users_collection.find_one({"_id": ObjectId(temp_user_id)})
            if not user or "face_encoding" not in user:
                return jsonify({"error": "User or face encoding not found"}), 400

            stored_encoding = pickle.loads(user["face_encoding"])
            matches = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=0.45)
            if matches[0]:
                session['user_id'] = session.pop('temp_user_id')
                return jsonify({"success": True})
            else:
                return jsonify({"error": "Face verification failed. Try again."}), 400
        except Exception as e:
            log_warning("Error during face verification: " + str(e))
            return jsonify({"error": str(e)}), 500

    return render_template('face_verification_html.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    """
    Receives a captured frame, processes it (head pose, emotion, eye & lip status,
    and face recognition), returns the analysis, and logs it in the current session.
    Additionally, if there is no valid head moment for more than 10 seconds or if no face is detected for more than 10 seconds,
    a warning is generated and logged.
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        img_data = data['image']
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        result = process_frame(frame)
        
        session_id = data.get('session_id')
        current_time = time.time()
        warnings_list = []
        if session_id:
            # Check head moment:
            if "head_state" in result["analysis"]:
                if result["analysis"]["head_state"] != "Unknown":
                    last_head_moment[session_id] = current_time
                else:
                    if session_id not in last_head_moment:
                        last_head_moment[session_id] = current_time
                if current_time - last_head_moment.get(session_id, current_time) > 10:
                    warning_msg = "No head moment for 10 seconds."
                    warnings_list.append(warning_msg)
                    log_warning(warning_msg)
            # Check face presence:
            if "error" in result["analysis"] and result["analysis"]["error"] == "No face detected":
                if session_id not in last_face_presence:
                    last_face_presence[session_id] = 0
                if current_time - last_face_presence.get(session_id, 0) > 10:
                    warning_msg = "User not present on screen for 10 seconds."
                    warnings_list.append(warning_msg)
                    log_warning(warning_msg)
            else:
                last_face_presence[session_id] = current_time

        if warnings_list:
            result["warnings"] = warnings_list

        # Log the analysis in the session document.
        if session_id:
            sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$push": {"data": result['analysis']}}
            )
        return jsonify(result)
    except Exception as e:
        log_warning("Error in process_frame route: " + str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/start_session', methods=['POST'])
def start_session():
    """
    Starts a new logging session for the logged-in user.
    Returns a session_id to be used for data logging.
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', None)
        if not user_id:
            return jsonify({"error": "User ID required"}), 400
        session_doc = {
            "user_id": user_id,
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "end_time": None,
            "data": []
        }
        session_id = sessions_collection.insert_one(session_doc).inserted_id
        return jsonify({"message": "Session started", "session_id": str(session_id)})
    except Exception as e:
        log_warning("Error starting session: " + str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    try:
        data = request.get_json()
        session_id = data.get('session_id', None)
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}}
        )
        return jsonify({"message": "Session ended"})
    except Exception as e:
        log_warning("Error ending session: " + str(e))
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
