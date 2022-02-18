from facelib import FaceDetector, AgeGenderEstimator
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

if __name__ == "__main__":
    face_detector = FaceDetector()
    age_gender_detector = AgeGenderEstimator()

    faces, boxes, scores, landmarks = face_detector.detect_align(img_test2)
    print("Face Score: {}".format(scores))
    genders, ages = age_gender_detector.detect(faces)
    print(genders, ages)