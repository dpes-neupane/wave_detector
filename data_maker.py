import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import time


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

wave_path = "./wave"
import os
files = os.listdir(wave_path)





counter = []
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        for file in files:
            cap = cv.VideoCapture(os.path.join(wave_path , file))
            count=0
            # with open("./notCsv/" + file[:-4]+ ".csv", "w") as csv_file:
            #     csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     heading = []
            #     for i in range(33):
            #         heading.append("x" + str(i))
            #         heading.append("y" + str(i))
            #         heading.append("z" + str(i))
            #         heading.append("v" + str(i))
            #     csv_writer.writerow( ["count"] + heading + ["class"])        
            while cap.isOpened():
                    # now = time.time()
                    ret, frame = cap.read()
                    if not ret or count>35 :
                        break
                    count+=1
                    
                    results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    
                    if results.pose_landmarks is not None:
                        pose_landmarks = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark]
                        if len(pose_landmarks) != 33:
                            print("not")
                        # pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()
                        
                        # csv_writer.writerow([count] + pose_landmarks + ["not"])
                        
                        
                    # print(count    , end = "\r")
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    # next_ = time.time()
                    # fps = 1 / (next_ - now)
                    # print(cap.get(cv.CAP_PROP_FPS), fps)
                    # print(fps)
                    cv.imshow("frame", frame)
                    if cv.waitKey(10) & 0xff == ord("q"):
                        break
            counter.append(count)
        cap.release()
        
cv.destroyAllWindows()
        
print(counter)