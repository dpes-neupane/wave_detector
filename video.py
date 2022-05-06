
import re
import cv2 as cv
import numpy as np
import mediapipe as mp
from tensorflow import keras
from keras.models import load_model


def model_loader():
    path = "./best_model.pkl"
    model = load_model(path)
    return model


    
    





model = model_loader()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv.VideoCapture(0)
    print(cap.isOpened())
    input_array = []
    while cap.isOpened():
        # print("yes")
        # now = time.time()
        ret, frame = cap.read()
        if ret:
            results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            if results.pose_landmarks is not None:
                pose_landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
                pose_landmarks = np.around(pose_landmarks, 5).flatten()
                input_array.append(pose_landmarks)
                
                
            if len(input_array) > 36:
                input_array.pop(0)
            if len(input_array) == 36:
                prediction_array = input_array
                prediction_array = np.array(prediction_array)
                value = model.predict(prediction_array.reshape(1, prediction_array.shape[0], prediction_array.shape[1]))
                frame = cv.putText(frame, str(np.around(value[0, 0],5)), (100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                if value[0, 0] > .70:
                    frame = cv.putText(frame, "WAVE", (200, 200), cv.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 1)
                    frame[:, :, :2] = 0
                # print(value.shape, value)
                
            mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
            # print(frame.shape)
            cv.imshow("frame", frame)
        if cv.waitKey(10) & 0xff == ord("q"):
            
            break
    cap.release()
    cv.destroyAllWindows()