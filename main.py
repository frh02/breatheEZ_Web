#!/usr/bin/env python
# coding: utf-8

# In[29]:


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# In[30]:




# In[31]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[6]:


mp_holistic.POSE_CONNECTIONS


# In[7]:


mp_holistic.HAND_CONNECTIONS


# In[46]:


mp_holistic.FACE_CONNECTIONS


# In[13]:


mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)


# In[14]:





# In[5]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
         # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[25]:


len(landmarks)


# In[27]:


for lndmrk in mp_holistic.PoseLandmark:
    print(lndmrk)


# In[26]:


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


# In[27]:


from math import sqrt
def calculate_distance(upp,low):
    upp = np.array(upp)
    low = np.array(low)
    return sqrt( (low[0] - upp[0])**2 + (low[1] - upp[1])**2 )


# In[21]:


upper_lip = [results.face_landmarks.landmark[13].x,results.face_landmarks.landmark[13].y]
lower_lip = [results.face_landmarks.landmark[14].x,results.face_landmarks.landmark[14].y]


# In[23]:


calculate_distance(upper_lip,lower_lip)


# In[13]:


shoulder_left = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
elbow_left = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
hip_left = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]


# In[38]:


shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
elbow_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
hip_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]


# In[40]:


shoulder_left, elbow_left, hip_left


# In[41]:


shoulder_right, elbow_right, hip_right


# In[43]:


calculate_angle(shoulder_left, elbow_left, hip_left)


# In[44]:


calculate_angle(shoulder_right, elbow_right, hip_right)


# In[16]:




# In[ ]:


cap = cv2.VideoCapture(0)
counter = 0
stage = None
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for the right arm 
            shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            hip_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
            
            # Getting coordinates for the left arm 
            
            shoulder_left = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            hip_left = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            
            #Getting coordinates for the lip coordinates
            
            upper_lip = [results.face_landmarks.landmark[13].x,results.face_landmarks.landmark[13].y]
            lower_lip = [results.face_landmarks.landmark[14].x,results.face_landmarks.landmark[14].y]
            
            # Calculate angle
            
            angle_left = calculate_angle(shoulder_left, elbow_left, hip_left)
            angle_right = calculate_angle(shoulder_right, elbow_right, hip_right)
            
            # Calculate distance 
            
            distance = calculate_distance(upper_lip,lower_lip)
            # Visualize angle
            
            cv2.putText(image, str(angle_right), 
                           tuple(np.multiply(shoulder_right, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(angle_left), 
                           tuple(np.multiply(shoulder_left, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(distance), 
                           tuple(np.multiply(upper_lip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            if distance < 0.007 and angle_left > 50 and angle_right > 50:
                stage = "incorrect"
            
            
            if distance > 0.06 and angle_left < 40 and angle_right < 40:
                stage="correct"
                counter +=1
                print(counter)
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
         # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        # 4. Face Connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
