from lib2to3.pgen2.token import OP
from operator import mod
from typing import Optional
from fastapi import FastAPI

from facelib import FaceDetector, AgeGenderEstimator
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Model():

    def __init__(self):
        self.face_detector = FaceDetector()
        self.age_gender_detector = AgeGenderEstimator()

    def predict(self, img):
        faces, boxes, scores, landmarks = self.face_detector.detect_align(img_test2)
        print("Face Score: {}".format(scores))
        genders, ages = self.age_gender_detector.detect(faces)
        print(genders, ages)

        return {'genders':genders, "ages":ages}

app = FastAPI()
model = Model()

@app.get("/predict/{img}")
def predict(img, hash: Optional[str] = None):
    global model

    return model.predict(img)



    