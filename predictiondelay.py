import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque, Counter
import time

# Load the trained model
model = tf.keras.models.load_model("asl_cnn_model_morning.h5")

# Define class labels (A-Z, Space)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

predicted_text = ""
frame_buffer = deque(maxlen=15)  # Store last 15 predictions for smoothing
prediction_delay = 0.5  # Time delay between predictions (seconds)
last_prediction_time = time.time()  # Track last prediction time
consecutive_frames = 0  # To track consecutive frames with confidence > 0.9

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of the hand
            x_min, y_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), \
                           int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max, y_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), \
                           int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Crop the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size == 0:
                continue

            # Preprocess the hand image
            img = cv2.resize(hand_roi, (128, 128))
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img)
            confidence = np.max(predictions)
            predicted_class = class_labels[np.argmax(predictions)]

            # If confidence is high, add to the buffer
            if confidence > 0.8:
                frame_buffer.append(predicted_class)

            # Only process prediction after accumulating sufficient stable predictions
            if len(frame_buffer) == frame_buffer.maxlen:
                most_common_prediction = Counter(frame_buffer).most_common(1)[0][0]

                # If enough time has passed, update the text with the most common prediction
                if time.time() - last_prediction_time > prediction_delay:
                    if most_common_prediction == "space":
                        predicted_text += " "  # Add space
                    elif most_common_prediction == "delete":
                        predicted_text = predicted_text[:-1]  # Remove last character (Backspace)
                    elif most_common_prediction != "nothing":
                        predicted_text += most_common_prediction  # Add detected letter

                    last_prediction_time = time.time()  # Reset timer
                    frame_buffer.clear()  # Clear buffer after using the prediction

            # Display current prediction
            cv2.putText(frame, f"Prediction: {predicted_class} ({confidence:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display accumulated text
    cv2.putText(frame, f"Text: {predicted_text}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
