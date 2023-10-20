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

from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   
from numba import njit

import timeit

#3 CAMARAS

def gestos_trainning():
    
   
    print("Comenzando Entrenamiento")
    df = pd.read_csv('D:/csv_3camss_abecedario/primera_prueba_abcnum11.csv',encoding = "ISO-8859-1")

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

    print("INICIANDO...")
    print('Para 3 camaras con abecedario')
    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    fit_models['rc'].predict(X_test)

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))


    fit_models['rf'].predict(X_test)

    with open('modelo_abecedario_camaras.pkl', 'wb') as f:

        pickle.dump(fit_models['rf'], f)
    """
    url = apiURL+'/modelo/señas'
    ficheros = {'file1': ('body_language.pkl', open('body_language.pkl', 'rb'), 'application/xml')}
    r = requests.post(url, files=ficheros)
    print("ARCHIVO ENVIADO CON EXITO")

    

    """
    


if __name__ == "__main__":

    print("INICIANDO TIEMPO")
    
    start = timer()
    gestos_trainning()
    print("with GPU:", timer()-start)
    
    