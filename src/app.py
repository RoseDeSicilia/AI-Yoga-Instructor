from flask import Flask, render_template, Response
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import pandas as pd
import pickle 

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open("src/yoga_pose_detector.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


def gen_video(page_pose):

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
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
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                    )
            cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
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
                
                #print("class: ", body_language_class, ", learning: ", pose)

                print(body_language_class)
                print('\n')
                print(body_language_class)
                print('\n')
                print(page_pose)
                # if body_language_class.split(' ,') == page_pose:
                #     print('matches')
                # Get status box          
                # cv2.putText(image, body_language_class, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   
                # cv2.putText(image, body_language_class, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)   

                if(page_pose == body_language_class and round(body_language_prob[np.argmax(body_language_prob)],2) > 0.5):
                # Display Class
                    cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class, (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                # Display Class
                    cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'waiting...', (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                # cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, body_language_class, (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                # cv2.putText(image, 'failed',
                #     (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                pass

            ret, frame = cv2.imencode('.jpg', image)
            frame = frame.tobytes()
            #yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def gen_video(pose):
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor Feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False        
            
#             # Make Detections
#             results = holistic.process(image)
            
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True   
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # 2. Right hand
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 3. Left Hand
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 4. Pose Detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )
#             cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
#             # Export coordinates
#             try:
#                 # 1. Extract Pose landmarks
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
#                 # Concate rows
#                 row = np.array(pose_row)

#                 # Make Detections
#                 X = row.reshape(1, -1)
#                 body_language_class = model.predict(X)[0]
#                 body_language_prob = model.predict_proba(X)[0]
                
#                 print("class: ", body_language_class, ", learning: ", pose)
#                 # Get status box                
#                 if(round(body_language_prob[np.argmax(body_language_prob)],2) > 0.6):
#                     # Display Class
#                     cv2.putText(image, body_language_class,
#                         (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 else:
#                     # Display Class
#                     cv2.putText(image, 'no detection',
#                         (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
#             except:
#                 cv2.putText(image, 'failed',
#                     (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 pass

#             ret, frame = cv2.imencode('.jpg', image)
#             frame = frame.tobytes()
#             #yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
#             yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_video(pose):
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor Feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False        
            
#             # Make Detections
#             results = holistic.process(image)
            
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True   
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # 2. Right hand
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 3. Left Hand
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 4. Pose Detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )
#             cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
#             # Export coordinates
#             try:
#                 # 1. Extract Pose landmarks
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
#                 # Concate rows
#                 row = np.array(pose_row)

#                 # Make Detections
#                 X = row.reshape(1, -1)
#                 body_language_class = model.predict(X)[0]
#                 body_language_prob = model.predict_proba(X)[0]
                
#                 print("class: ", body_language_class, ", learning: ", pose)
#                 # Get status box                
#                 if(round(body_language_prob[np.argmax(body_language_prob)],2) > 0.6):
#                     # Display Class
#                     cv2.putText(image, body_language_class,
#                         (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 else:
#                     # Display Class
#                     cv2.putText(image, 'no detection',
#                         (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
#             except:
#                 cv2.putText(image, 'failed',
#                     (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 pass

#             ret, frame = cv2.imencode('.jpg', image)
#             frame = frame.tobytes()
#             #yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
#             yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_video(pose):
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor Feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False        
            
#             # Make Detections
#             results = holistic.process(image)
            
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True   
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # 2. Right hand
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 3. Left Hand
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 4. Pose Detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )
#             cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
#             # Export coordinates
#             try:
#                 # 1. Extract Pose landmarks
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
#                 # Concate rows
#                 row = np.array(pose_row)

#                 # Make Detections
#                 X = row.reshape(1, -1)
#                 body_language_class = model.predict(X)[0]
#                 body_language_prob = model.predict_proba(X)[0]
                
#                 print("class: ", body_language_class, ", learning: ", pose)
#                 # Get status box                
#                 if(round(body_language_prob[np.argmax(body_language_prob)],2) > 0.6):
#                     # Display Class
#                     cv2.putText(image, body_language_class,
#                         (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 else:
#                     # Display Class
#                     cv2.putText(image, 'no detection',
#                         (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
#             except:
#                 cv2.putText(image, 'failed',
#                     (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 pass

#             ret, frame = cv2.imencode('.jpg', image)
#             frame = frame.tobytes()
#             #yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
#             yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def gen_video(pose):
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             # Recolor Feed
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False        
            
#             # Make Detections
#             results = holistic.process(image)
            
#             # Recolor image back to BGR for rendering
#             image.flags.writeable = True   
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
#             # 2. Right hand
#             mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 3. Left Hand
#             mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )

#             # 4. Pose Detections
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
#                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                                     )
#             cv2.rectangle(image, (0,0), (350, 60), (245, 117, 16), -1)
#             # Export coordinates
#             try:
#                 # 1. Extract Pose landmarks
#                 pose = results.pose_landmarks.landmark
#                 pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
#                 # Concate rows
#                 row = np.array(pose_row)

#                 # Make Detections
#                 X = row.reshape(1, -1)
#                 body_language_class = model.predict(X)[0]
#                 body_language_prob = model.predict_proba(X)[0]
                
#                 print("class: ", body_language_class, ", learning: ", pose)
#                 # Get status box                
#                 if(round(body_language_prob[np.argmax(body_language_prob)],2) > 0.6):
#                     # Display Class
#                     cv2.putText(image, body_language_class,
#                         (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#                 else:
#                     # Display Class
#                     cv2.putText(image, 'no detection',
#                         (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
#             except:
#                 cv2.putText(image, 'failed',
#                     (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                 pass

#             ret, frame = cv2.imencode('.jpg', image)
#             frame = frame.tobytes()
#             #yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n')
#             yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
@app.route('/index.html')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    """Contact page."""
    return render_template('contact.html')

@app.route('/about.html')
def about():
    """About page."""
    return render_template('about.html')

@app.route('/learn.html')
def learn():
    """learn page."""
    return render_template('learn.html')

@app.route('/forward.html')
def forward():
    """Video streaming home page."""
    return render_template('forward.html')

@app.route('/mountain.html')
def mountain():
    """Video streaming home page."""
    return render_template('mountain.html')

@app.route('/warrior1.html')
def warrior1():
    """Video streaming home page."""
    return render_template('warrior1.html')

@app.route('/warrior2.html')
def warrior2():
    """Video streaming home page."""
    return render_template('warrior2.html')

@app.route('/warrior3.html')
def warrior3():
    """Video streaming home page."""
    return render_template('warrior3.html')

@app.route('/detect_warrior1')
def detect_warrior1():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video("Warrior Pose 1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_warrior2')
def detect_warrior2():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video("Warrior Pose 2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_warrior3')
def detect_warrior3():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video("Warrior Pose 3"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_forward')
def detect_forward():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video("Forward Fold"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_mountain')
def detect_mountain():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_video("Mountain Fold"), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=2204)
