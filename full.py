import cv2
import dlib
import pyttsx3
import time
from scipy.spatial import distance
from playsound import playsound  # Ensure this module is installed for alarm sound

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Initialize camera
cap = cv2.VideoCapture(0)

# Face detector and face landmark detector
face_detector = dlib.get_frontal_face_detector()
shape_predictor_path = r"C:\Users\anush\Downloads\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
dlib_facelandmark = dlib.shape_predictor(shape_predictor_path)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for detecting closed eyes
EYE_AR_THRESHOLD = 0.22
ALERT_DURATION = 5  # Time in seconds before triggering drowsiness alert

# Variables to track alert conditions
eyes_closed_start_time = None
eyes_open_status = True
alert_count = 0
alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)

        right_eye_points = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]
        left_eye_points = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]

        # Draw lines around the eyes
        for n in range(42, 48):
            next_point = (n + 1) if n < 47 else 42
            cv2.line(frame, right_eye_points[n - 42], right_eye_points[next_point - 42], (0, 255, 0), 1)
        
        for n in range(36, 42):
            next_point = (n + 1) if n < 41 else 36
            cv2.line(frame, left_eye_points[n - 36], left_eye_points[next_point - 36], (255, 255, 0), 1)

        # Calculate EAR for both eyes and average
        right_ear = calculate_eye_aspect_ratio(right_eye_points)
        left_ear = calculate_eye_aspect_ratio(left_eye_points)
        avg_ear = (right_ear + left_ear) / 2.0

        if avg_ear < EYE_AR_THRESHOLD:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()

            elapsed_time = time.time() - eyes_closed_start_time
            
            if elapsed_time >= ALERT_DURATION:
                cv2.putText(frame, "DROWSINESS DETECTED - WAKE UP!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if alert_count < 2:
                    if eyes_open_status:
                        engine.say("Alert!!!! WAKE UP DUDE")
                        engine.runAndWait()
                        eyes_open_status = False
                        alert_count += 1

                elif alert_count >= 2 and not alarm_playing:
                    # Play the alarm once for at least 10 seconds
                    alarm_playing = True
                    playsound(r"C:\Users\anush\Downloads\mixkit-alert-alarm-1005.wav")  # Make sure this file exists
                    alert_count += 1

            else:
                # Indicate the eyes are closed
                cv2.putText(frame, "Eyes are closed", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            eyes_closed_start_time = None
            alarm_playing = False
            if not eyes_open_status:
                cv2.putText(frame, "Eyes are open", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                eyes_open_status = True

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
