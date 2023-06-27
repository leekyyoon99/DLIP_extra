import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv.VideoCapture(1)

# Curl counter variables
counter = 0 
stage = None
image_number=1

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
            
        if cv.waitKey(1) & 0xFF == 27:
            break
        # Recolor image to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA
                                )
            
            # Curl counter logic
            if angle > 160:
                stage = "down"
                # cv.imwrite(cv.imwrite(f'C:/Users/LeeKiYoon/Desktop/이기윤/23년도 1학기/DLIP/extra/{counter}.jpg', image))
               
            
            
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                cv.imwrite(f'frame_{image_number}.jpg', frame)
                print(f'Saved frame_{image_number}.jpg')
                image_number += 1
                print(counter)

            # cv.imwrite(cv.imwrite(f'C:/Users/LeeKiYoon/Desktop/이기윤/23년도 1학기/DLIP/extra/{counter}.jpg', image))
            # if flag==1:
               
            #     flag = 0

            

        except:
            pass
        
        # Render curl counter
        # Setup status box
        # cv.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv.putText(image, 'REPS', (15,12), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(image, str(counter), 
                    (10,60), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv.LINE_AA)
        
        # Stage data
        cv.putText(image, 'STAGE', (65,12), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
        cv.putText(image, stage, 
                    (60,60), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv.imshow('Mediapipe Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()