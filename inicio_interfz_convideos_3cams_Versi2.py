from __future__ import absolute_import, division, print_function
from concurrent.futures import thread

from typing import Dict, List, Any
from collections import OrderedDict
from functools import partial


from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QStyle,
    QGridLayout,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QApplication,
)
from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
    QPoint,
    pyqtSlot,
    QSize,
    Qt,
    QTimer,
    QTime,
    QDate,
    QObject,
    QEvent,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPalette,
    QResizeEvent,
    QMouseEvent,
    QFont,
    QIcon,
)

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from decouple import config
from dataclasses import dataclass
from typing import Any, Dict, Optional, Text, Literal
from datetime import datetime

import tkinter as tk
import cv2
import imutils
import os
import tkinter.messagebox as MessageBox
import mysql.connector as mysql
import pymysql
import numpy as np
import sys
import shutil
import time
import smtplib
import pyautogui as pg
import time
import webbrowser as web

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score # Accuracy metrics 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import webbrowser as web
import pyautogui as pg
import time

import requests
import json
import threading


#Se tienen que crear las funciones def funcion x para cada algoritmo
#Definir la interfaz tkinter para que este en funcion con la api request y api rest 


#camara_web=1 #central
#camara_web2=3 #izquierda
#camara_web3=4 #derecha

#------------------------------------------------------Crear funcion para las interfaces y apis request-------------------------------------------------------
#-----------------------------------------------------------------Funcion para captura de gestos---------------------------------------------------------------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions4



apiURL = "http://192.168.100.29:105/"

datos=[]

#camaraweb=1



@dataclass
class KnownFace:
    label: Text
    position: Optional[Dict[Literal["x", "y", "w", "h"], int]] = None
    was_found: bool = False
    was_notified: bool = False
    timestamp: Optional[datetime] = None


def terminada_camara2():
    
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    #MessageBox.showinfo("STATUS", "DI")
    print("DICCIONARIO GENERADO")




def terminado_captura2():
    
    t3 = threading.Timer(10, terminada_camara2)
    t3.start()



def inici_gestos():
    global cap
    global cap2
    t5=threading.Thread(target=terminado_captura2)
                
    t5.start()

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions4
    class_name = audiotexto.get()

    #1 central
    #2 derecha
    #3 izquierda

    camara_web=1 #central
    camara_web2=3 #izquierda
    camara_web3=4 #derecha

    classcam=class_name+'.mp4'


    cap = cv2.VideoCapture('D:/videos presentacion/hola 1.mp4')
    cap2=cv2.VideoCapture('D:/videos presentacion/hola 2.mp4')
    cap3=cv2.VideoCapture('D:/videos presentacion/hola 3.mp4')
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False, 
    model_complexity=1, 
    smooth_landmarks=True ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            ret2, frame2=cap2.read()
            ret2, frame3=cap3.read()

            
            # Recolor Feed
            #-------------------------------------------camara 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)

            #--------------------------------------------camra 2
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image2.flags.writeable = False        
            
            # Make Detections
            results2 = holistic.process(image2)

            #-----------------------------------------------camara 3

            image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            image3.flags.writeable = False        
            
            # Make Detections
            results3 = holistic.process(image3)




            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            #-------------------------------------------------CONFIGURACIONES DETECCION CAMARA 1---------------------------------
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
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
            
            if results.right_hand_landmarks:
                righthand=results.right_hand_landmarks.landmark

            if results.left_hand_landmarks:
                lefthand=results.left_hand_landmarks.landmark

            
            #--------------------------------------CONFIGURACIONES DETECCION CAMARA 2-----------------------
            image2.flags.writeable = True   
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            
            mp_drawing.draw_landmarks(image2, results2.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
                                    
                                    
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            
            if (results2.right_hand_landmarks or results2.left_hand_landmarks) is not None :
                righthand2=results2.right_hand_landmarks.landmark
                lefthand2=results2.left_hand_landmarks.landmark
            
            #----------------------------------------------------------configuraciones camara 3--------------

            image3.flags.writeable = True   
            image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            
            mp_drawing.draw_landmarks(image3, results3.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
                                    
                                    
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image3, results3.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image3, results3.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image3, results3.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            
            if (results3.right_hand_landmarks or results3.left_hand_landmarks) is not None :
                righthand3=results3.right_hand_landmarks.landmark
                lefthand3=results3.left_hand_landmarks.landmark
            """

            if results2.left_hand_landmarks:
                lefthand2=results2.left_hand_landmarks.landmark
                """
                            
            cv2.imshow('Raw Webcam Feed', image)
            cv2.imshow('camara dos', image2)
            cv2.imshow('camara 3', image3)

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    #valo=results.left_hand_landmarks.landmark[0].visibility
    #print(valo)

    #num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
    #num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)+ len(results.left_hand_landmarks)+len(results.right_hand_landmarks)
    num_coords = (len(results.pose_landmarks.landmark)+ len(results2.pose_landmarks.landmark ) + len(results3.pose_landmarks.landmark)
    )+(len(results.face_landmarks.landmark)+len(results2.face_landmarks.landmark )+ len(results3.face_landmarks.landmark )
    )+ (len(righthand) +len( righthand2)+ len( righthand3))  + (len(lefthand) +len( lefthand2)+ len(lefthand3))
    
    print(num_coords)
    

    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        

    print("pasando a csv")


    with open('primera_prueba_abcnum11.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
    

def terminada_camara():
    
    print("CAPTURA TERMINADA")
    cap.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()
    
    
    dataPath = 'C:/Users/zero_/OneDrive/Documentos/Manos_training/ActionDetectionforSignLanguage/para_prueba_3camsvids'
    gestospath=os.listdir(dataPath)
    print(gestospath)
    audiotexto.delete(0, 'end')
    MessageBox.showinfo("STATUS", "GESTO GUARDADO")

def terminado_captura():
    
    t = threading.Timer(20, terminada_camara)
    t.start()
                        



def captura_gestos():
    global cap
    global cap2
    global cap3
    
    
    #t1=threading.Thread(target=terminado_captura)
                
    #t1.start()
    

    
        




    """

    with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    """
    
    


    
    print("CAPTURANDO GESTO")

    

    for x in range(3):

        #print(x)
        class_name = audiotexto.get()
        dataPath = 'C:/Users/zero_/OneDrive/Documentos/Manos_training/ActionDetectionforSignLanguage/para_prueba_3camsvids'
        personPath = dataPath + '/' + class_name
        if not os.path.exists(personPath):
            print('Carpeta creada: ',personPath)
            os.makedirs(personPath)
        class_name = audiotexto.get()
    
        cap = cv2.VideoCapture(class_name+' 1.mp4') #central
        cap2=cv2.VideoCapture(class_name+' 2.mp4') #derecha
        cap3=cv2.VideoCapture(class_name+' 3.mp4') #izquierda
        # Initiate holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        
            while (cap.isOpened()) or (cap2.isOpened()) or (cap3.isOpened()):
                ret, frame = cap.read()
                ret2, frame2=cap2.read()
                ret3,frame3=cap3.read()
                if (ret  == True  ):
            
                    # Recolor Feed
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False        
                
                    # Make Detections
                    results = holistic.process(image)

                    #camara2------------------------------

                    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    image2.flags.writeable = False        
                
                    # Make Detections
                    results2 = holistic.process(image2)

                    #camara3--------------------------------------------

                    image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                    image3.flags.writeable = False        
                
                    # Make Detections
                    results3 = holistic.process(image3)


                    # print(results.face_landmarks)
                
                    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                    # Recolor image back to BGR for rendering

                    #configuraciones camara 1----------------------------------------------------------------
                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    # 1. Draw face landmarks
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
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

                    #configuraciones camara 2----------------------------------------------------------------
                    image2.flags.writeable = True   
                    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
                
                    # 1. Draw face landmarks
                    mp_drawing.draw_landmarks(image2, results2.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                    # 2. Right hand
                    mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                    # 3. Left Hand
                    mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )

                    #configuraciones camara 3----------------------------------------------------------------
                    image3.flags.writeable = True   
                    image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)
                
                    # 1. Draw face landmarks
                    mp_drawing.draw_landmarks(image3, results3.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                    # 2. Right hand
                    mp_drawing.draw_landmarks(image3, results3.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                    # 3. Left Hand
                    mp_drawing.draw_landmarks(image3, results3.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(image3, results3.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                    # Export coordinates
                    try:
                        #poses camara 1
                        # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                        # Extract Face landmarks
                        face = results.face_landmarks.landmark
                        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                        if results.right_hand_landmarks is not None:
                            righthand=results.right_hand_landmarks.landmark
                            hand_right=results.right_hand_landmarks.landmark
                            right_row=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right]).flatten())

                        if results.left_hand_landmarks is not None:
                            lefthand=results.left_hand_landmarks.landmark
                            hand_left=results.left_hand_landmarks.landmark
                            left_row=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left]).flatten())
                        
                        #poses camara 2-------------------------------------
                        pose2 = results2.pose_landmarks.landmark
                        pose_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose2]).flatten())
                    
                        # Extract Face landmarks
                        face2 = results2.face_landmarks.landmark
                        face_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face2]).flatten())

                        if results2.right_hand_landmarks is not None:
                            righthand2=results2.right_hand_landmarks.landmark
                            hand_right2=results2.right_hand_landmarks.landmark
                            right_row2=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right2]).flatten())

                        if results2.left_hand_landmarks is not None:
                            lefthand2=results2.left_hand_landmarks.landmark
                            hand_left2=results2.left_hand_landmarks.landmark
                            left_row2=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left2]).flatten())
                        

                        #poses camara 3---------------------------------

                        pose3 = results3.pose_landmarks.landmark
                        pose_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose3]).flatten())
                    
                        # Extract Face landmarks
                        face3 = results3.face_landmarks.landmark
                        face_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face3]).flatten())

                        if results3.right_hand_landmarks is not None:
                            righthand3=results3.right_hand_landmarks.landmark
                            hand_right3=results3.right_hand_landmarks.landmark
                            right_row3=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right3]).flatten())

                        if results3.left_hand_landmarks is not None:
                            lefthand3=results3.left_hand_landmarks.landmark
                            hand_left3=results3.left_hand_landmarks.landmark
                            left_row3=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left3]).flatten())
                                
                        # Concate rows
                        row = pose_row+ face_row + right_row + left_row +pose_row2+ face_row2 + right_row2 + left_row2 + pose_row3+ face_row3 + right_row3 + left_row3
                    
                        # Append class name 
                        row.insert(0, class_name)
                    
                        # Export to CSV
                        with open('D:/csv_3cams/primera_prueba_abcnum13.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row) 
                    
                    except:
                        pass
                                
                    cv2.imshow('Raw Webcam Feed', image)
                    cv2.imshow('camara 2 ', image2)
                    cv2.imshow('camra 3', image3)
                    



                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                else: break

    
    terminada_camara()
        


def gestos_trainning():
    
   
    print("Comenzando Entrenamiento")
    df = pd.read_csv('primera_prueba_abcnum10.csv',encoding = "ISO-8859-1")

    df.head()

    df.tail()

    df[df['class']=='bien']

    X = df.drop('class', axis=1) # features
    y = df['class'] # target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    pipelines = {
        'lr':make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='ovr', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier(random_state=42)),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=100,random_state=42,)),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, min_samples_split=2,
                                      max_depth=1,
                                      learning_rate=0.1, subsample=0.5
                                      ,random_state=42)),
    }


    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    fit_models['rc'].predict(X_test)

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))


    fit_models['rf'].predict(X_test)

    with open('primera_prueba_abcnum11.pkl', 'wb') as f:

        pickle.dump(fit_models['rf'], f)
    """
    url = apiURL+'/modelo/señas'
    ficheros = {'file1': ('body_language.pkl', open('body_language.pkl', 'rb'), 'application/xml')}
    r = requests.post(url, files=ficheros)
    print("ARCHIVO ENVIADO CON EXITO")

    

    """
    with open('primera_prueba_abcnum6.pkl', 'rb') as f:
        model = pickle.load(f)
        
        


    print("Modelo terminado ")
    MessageBox.showinfo("STATUS", "MODELO ENTRENADO ")




def acquire_frame(capture_device, capture_device2, capture_device3):
    # Aquí se hace la captura del frame y envío con cv2
    ret, frame = capture_device.read()
    ret2, frame2=capture_device2.read()
    ret3, frame3= capture_device3.read()
    return frame, frame2, frame3


def prediccion_señas():
    MessageBox.showinfo("STATUS", "INICIANDO PREDICCION")
    class_name = audiotexto.get()


    camara_web=2
    camara_web3=1
    camara_web2=4

    dataPath = 'C:/Users/zero_/OneDrive/Documentos/Manos_training/ActionDetectionforSignLanguage/para_prueba_3camsvids'
    gestospath=os.listdir(dataPath)
    print(gestospath)
    def lenguaje_señas(frame, known_faces: Dict[Text, KnownFace], frame2, frame3):
        with open('modelo_presentacion_3_1camaras.pkl', 'rb') as f:
            model = pickle.load(f)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
            #camara 1 configuraciones---------------------------
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
        
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
        
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #camara 2 configuraciones------------------------------------

            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image2.flags.writeable = False        
        
            # Make Detections
            results2 = holistic.process(image2)
            # print(results.face_landmarks)
        
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
            # Recolor image back to BGR for rendering
            image2.flags.writeable = True   
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)


            #camara 3 configuraciones----------------------------------

            image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            image3.flags.writeable = False        
        
            # Make Detections
            results3 = holistic.process(image3)
            # print(results.face_landmarks)
        
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
            # Recolor image back to BGR for rendering
            image3.flags.writeable = True   
            image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)

            #--------------------------------------------------------------

            #configuraciones camara 1----------------------------------------------------------
            # 1. Draw face landmarks
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
        
            # 2. Right hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

            #configuraciones camara 2----------------------------------------------------------
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image2, results2.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
        
            # 2. Right hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            

            #configuraciones camara 3----------------------------------------------------------

            k_faces_labels = set()
            mp_drawing.draw_landmarks(image3, results3.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
        
            # 2. Right hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image3, results3.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image3, results3.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            k_faces_labels = set()
            mp_drawing.draw_landmarks(image3, results3.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            


            
            # Export coordinates
            try:

                #camara 1-----------------------------------------------

                
                        # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                if results.right_hand_landmarks is not None:
                    righthand=results.right_hand_landmarks.landmark
                    hand_right=results.right_hand_landmarks.landmark
                    right_row=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right]).flatten())

                if results.left_hand_landmarks is not None:
                    lefthand=results.left_hand_landmarks.landmark
                    hand_left=results.left_hand_landmarks.landmark
                    left_row=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left]).flatten())
                    
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in righthand]).flatten())

                left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lefthand]).flatten())

                #camara 2----------------------------------------------------------------------

                pose2 = results2.pose_landmarks.landmark
                pose_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose2]).flatten())
            
                # Extract Face landmarks
                face2 = results2.face_landmarks.landmark
                face_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face2]).flatten())

                if results2.right_hand_landmarks is not None:
                    righthand2=results2.right_hand_landmarks.landmark
                    hand_right2=results2.right_hand_landmarks.landmark
                    right_row2=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right2]).flatten())

                if results2.left_hand_landmarks is not None:
                    lefthand2=results2.left_hand_landmarks.landmark
                    hand_left2=results2.left_hand_landmarks.landmark
                    left_row2=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left2]).flatten())
                    
                # Extract Pose landmarks
                pose2 = results2.pose_landmarks.landmark
                pose_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose2]).flatten())
            
                # Extract Face landmarks
                face2 = results2.face_landmarks.landmark
                face_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face2]).flatten())

                right_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in righthand2]).flatten())

                left_row2 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lefthand2]).flatten())

                #camara 3-------------------------------------------------------------

                pose3 = results3.pose_landmarks.landmark
                pose_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose3]).flatten())
            
                # Extract Face landmarks
                face3 = results3.face_landmarks.landmark
                face_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face3]).flatten())

                if results3.right_hand_landmarks is not None:
                    righthand3=results3.right_hand_landmarks.landmark
                    hand_right3=results3.right_hand_landmarks.landmark
                    right_row3=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_right3]).flatten())

                if results3.left_hand_landmarks is not None:
                    lefthan3=results3.left_hand_landmarks.landmark
                    hand_left3=results3.left_hand_landmarks.landmark
                    left_row3=list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand_left3]).flatten())
                    
                # Extract Pose landmarks
                pose3 = results3.pose_landmarks.landmark
                pose_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose3]).flatten())
            
                # Extract Face landmarks
                face3 = results3.face_landmarks.landmark
                face_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face3]).flatten())

                right_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in righthand3]).flatten())

                left_row3 = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lefthan3]).flatten())


            
                # Concate rows
                row = pose_row+face_row + right_row+ left_row + pose_row2 +face_row2  + right_row2 + left_row2 + pose_row3 +face_row3  + right_row3 + left_row3
            
    #             # Append class name 
    #             row.insert(0, class_name)
            
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 
            
                # Make Detections
                X = pd.DataFrame([row])
            
                body_language_class = model.predict(X.values)[0]
                body_language_prob = model.predict_proba(X.values)[0]
                aproximado=round(body_language_prob[np.argmax(body_language_prob)],2)
                proximado=int(aproximado)
                #print(aproximado)
                #print(body_language_class)

                if aproximado >=0.0:

            
                #print(body_language_class)

                    label = str(body_language_class)
                    known = known_faces.get(label)
            

                

                    k_faces_labels.add(label)
                    known_faces = {
                        **known_faces,
                        label: KnownFace(
                            label=label,
                        
                            was_found=True,
                            was_notified=known.was_found,
                            timestamp=datetime.utcnow(),
                        ),
                    }
                    #se guarda de forma consecutiva
                    
                    #time.sleep(8)
                    #pg.write(body_language_class)
                    #pg.press('enter')
                    #print("mensaje de whatsapp enviado con exito")
                
                    # Grab ear coords

                    #camara 1
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    #print(coords)
                
                    cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                    # Get status box 
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    #camara 2--------------------------------------------

                    coords2 = tuple(np.multiply(
                                    np.array(
                                        (results2.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results2.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    #print(coords)
                
                    cv2.rectangle(image2, 
                                    (coords2[0], coords2[1]+5), 
                                    (coords2[0]+len(body_language_class)*20, coords2[1]-30), 
                                    (245, 117, 16), -1)
                    cv2.putText(image2, body_language_class, coords2, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                    # Get status box 
                    cv2.rectangle(image2, (0,0), (250, 60), (245, 117, 16), -1)
                
                    # Display Class
                    cv2.putText(image2, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image2, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    # Display Probability
                    cv2.putText(image2, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image2, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    #camara 3-----------------------------------------------

                    coords3 = tuple(np.multiply(
                                    np.array(
                                        (results3.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results3.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    #print(coords)
                
                    cv2.rectangle(image3, 
                                    (coords3[0], coords3[1]+5), 
                                    (coords3[0]+len(body_language_class)*20, coords3[1]-30), 
                                    (245, 117, 16), -1)
                    cv2.putText(image3, body_language_class, coords3, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                    # Get status box 
                    cv2.rectangle(image3, (0,0), (250, 60), (245, 117, 16), -1)
                
                    # Display Class
                    cv2.putText(image3, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image3, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    # Display Probability
                    cv2.putText(image3, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image3, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
            
            except:
                pass

            unrecognized_faces = set(known_faces.keys()) - k_faces_labels
            #un solo dato

            known_faces = {
                label: KnownFace(
                    label=face.label,
                    position=face.position if label not in unrecognized_faces else None,
                    was_found=face.was_found if label not in unrecognized_faces else False,
                    was_notified=face.was_notified
                    if label not in unrecognized_faces
                    else False,
                    timestamp=face.timestamp if label not in unrecognized_faces else None,
                )
                for label, face in known_faces.items()
            }


                        
            cv2.imshow('Lenguaje de señas', image)
            cv2.imshow('camara 2 ', image2)
            cv2.imshow('camara 3', image3)
        
        return known_faces
        
        



    def send_notification(face: KnownFace):
    
        #Proceso para enviar notificación a donde quieras
        face.was_notified = True
        mano=face.label
        handsign=str(mano)
        print(handsign  )
        
        #message= (nombrecara + "  DETECTADO")
        #time.sleep(8)
        datos.append(handsign)
        daatos2=" "


        datos1=daatos2.join(datos)
        #print(datos1)

        tup=[]

        print("El numero de datos en la lista son: ", len(datos))

        if len(datos)== 5:
            print(datos1)
        
        else:
            if len(datos)>=6:
                
                datos.clear()
                
        #pg.write(handsign)
        #pg.press('enter')
        #print("mensaje de whatsapp enviado con exito")

        #url1 = apiURL+'/notificacion/señas/Whatsapp'

        #paramss1 = {'señas': handsign}

        #dato1 = requests.post(url1, data = paramss1)

        #print(dato1.status_code)

        return face

    

    def main(known_faces):

    
        classee=audiotexto.get()
    
        
        capture_device = cv2.VideoCapture(camara_web)
        capture_device2 = cv2.VideoCapture(camara_web2)
        capture_device3 = cv2.VideoCapture(camara_web3)

        """
        capture_device = cv2.VideoCapture(classee+ ' 1.mp4')
        capture_device2 = cv2.VideoCapture(classee+ ' 2.mp4')
        capture_device3 = cv2.VideoCapture(classee+ ' 3.mp4')"""

        while True:
            frame , frame2 , frame3= acquire_frame(capture_device, capture_device2, capture_device3)
            #frame2=acquire_frame(capture_device2)
            #frame3= acquire_frame(capture_device3)
            if cv2.waitKey(1) == ord("q"):
                break
            known_faces = lenguaje_señas(frame, known_faces, frame2, frame3)
        
            known_faces = {
                label: send_notification(face)
                if face.was_found and not face.was_notified

                else face
                for label, face in known_faces.items()
            }

            #show_frame("frame", frame)
        capture_device.release()
        cv2.destroyAllWindows()


    if __name__ == "__main__":
        #gestos=['por favor', 'si', 'Hola que tal', 'gracias', 'familia','mucho gusto', 'buenas tardes', 'mi nombre es']
        
        dicio={v:KnownFace(label=v)
    



        for v in gestospath}

    
    
    
        # Aquí podrías cargar configuraciones como credenciales
        # desde un archivo o algo.
        main(dicio)



root= Tk()
root.iconbitmap("rost.ico")

root.title("RECONOCIMIENTO FACIAL")
root.config(bg="yellow")

root.geometry("550x600+700+10")

root.resizable(width=False,height=False)


miImagen= PhotoImage(file="Deep-Learning-4.gif")
my= Label(root, image = miImagen).place(x=0, y=0, relwidth=1, relheight=1)
    
    
boton= Button(root, text="LENGUAJE VIDEO", command= prediccion_señas)
boton.place(x=40, y=320)


boton1= Button(root, text="ENTRENAMIENTO", command=  gestos_trainning )
boton1.place(x=40, y=250)

boton2= Button(root, text="Iniciar diccionario nuevo csv", command= inici_gestos)
boton2.place(x=40, y=10)

boton3= Button(root, text="Captura gestos", command= captura_gestos)
boton3.place(x=40, y=400)

audiotext= StringVar()
audiotexto=Entry(root,textvariable=audiotext)
audiotexto.place(x=200, y=400)




root.mainloop()
