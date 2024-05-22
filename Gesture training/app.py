from function import *
from keras.utils import to_categorical #type: ignore
from keras.models import model_from_json #type: ignore
from keras.layers import LSTM, Dense #type: ignore
from keras.callbacks import TensorBoard #type: ignore
import pyautogui
import time

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define colors
colors = [(245,117,16) for _ in range(20)]

# New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Open the camera
cap = cv2.VideoCapture(0)

# Set mediapipe model
with mp.solutions.hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Crop frame
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0,40), (300,400), 255, 2)

        # Make detections
        image, results = mediapipe_detection(cropframe, hands)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try: 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                # Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))
                                # Check if a specific gesture is detected, then press a key
                                if actions[np.argmax(res)] == 'Forward':
                                    print("Triggering Key Press...")
                                    pyautogui.press('win')
                                    print("Key Press Triggered")
                                    time.sleep(0.5)
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)]*100)) 

                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            pass

        # Display output
        cv2.rectangle(frame, (0,0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: -"+' '.join(sentence)+''.join(accuracy), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('OpenCV Feed', frame)

        # Break if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()