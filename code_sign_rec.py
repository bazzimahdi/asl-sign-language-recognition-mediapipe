
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
import time
from utils_sign_rec import preprocess_landmark, get_landmarks_list, return_alphabet
import string
import pickle


# load the model from disk
clf = pickle.load(open('D:\\Python\\PROM02 Code and data\\mediapipe\\svm_model.pkl', 'rb'))

# map numbers to alphabets
alphabet = list(string.ascii_uppercase)
dict_alphabet={}
for i in range(26):
    dict_alphabet[i]=alphabet[i]

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For static images:
    mp_model = mp_hands.Hands(
        static_image_mode=True, # only static images
        max_num_hands=1, # max 1 hand detection
        min_detection_confidence=0.5, # detection confidence
        min_tracking_confidence=0.7) # confidence for interpolating landmarks during movement


    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        ##############################
        frame=cv2.flip(frame, 1)

        roi=frame.copy()
        cv2.imshow('roi',roi)

        results = mp_model.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks_data = []
            # preprocess landmarks to be compatible with svm clf
            landmarks = preprocess_landmark(get_landmarks_list(roi,results),False)
            landmarks_data.append(landmarks)

            for handLms in results.multi_hand_landmarks:
                cx_data=[]
                cy_data=[]
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = roi.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    cv2.circle(roi, (cx,cy), 3, (255,0,255), cv2.FILLED)
                mp_drawing.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)

            pred = clf.predict(landmarks_data)
            logit = max(clf.predict_proba(landmarks_data)[0])
    #         print(pred)

            threshold = 0.3
            if logit > threshold:
                cv2.putText(roi,dict_alphabet[pred[0]], (50,75), cv2.FONT_HERSHEY_PLAIN, 3, (0,250,200), 2)      
                cv2.imshow('roi',roi)

            else:
                cv2.putText(roi,'no sign detected', (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,250,200), 2)
                cv2.imshow('roi', roi)

        # calculate FPS
        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
            print("FPS:", fps / TIME)
            fps = 0 
            start_time = time.time()

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()




