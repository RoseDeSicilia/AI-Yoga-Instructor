from flask import Flask, render_template, Response

import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import pickle 

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# filename = 'src/yoga_pose_detector.pkl'
# os.makedirs(os.path.dirname(filename), exist_ok=True)

with open('src/yoga_pose_detector.pkl', 'rb') as f:
    model = pickle.load(f)

def gen_video(pose):
    
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        # Make Detections
        results = holistic.process(image)
    
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 2. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 3. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
            
        # Export coordinates
        try:
            # 1. Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Concate rows
            row = np.array(pose_row)

            # Make Detections
            X = row.reshape(1, -1)
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
                        
            # Get status box
            cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
            
            if(pose == body_language_class.split(',')[0] and 
               round(body_language_prob[np.argmax(body_language_prob)],2) > 0.8):
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(',')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Display Probability
#                 cv2.putText(image, 'PROB'
#                             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
#                             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            else:
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'waiting...'
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 # Display Probability
#                 cv2.putText(image, 'PROB'
#                             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(image, 'N/A'
#                             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            
        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)
        
        ret, buffer = cv2.imencode('.jpg', image)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/learn.html')
def learn():
    """learn page"""
    return render_template('learn.html')

@app.route('/about.html')
def about():
    """about page"""
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    """contact page"""
    return render_template('contact.html')

@app.route('/index.html')
def recindex():
    """Return to home page."""
    return render_template('index.html')



@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=2204)
