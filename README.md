# FaceLib : Face Detection and Facial Expression Recognition and Age & Gender Classification and Face Recognition with pytorch

## Face Detection: RetinaFace

A PyTorch implementation of RetinaFace: Single-stage Dense Face Localisation in the Wild. Model size only 1.7M, 
when Retinaface The official code in Mxnet can be found here.

 you can use this backbone networks:
    Resnet50, mobilenet, slim, RFB
 
 The following example illustrates the ease of use of this package:

  ```python
   from Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cpu')
   boxes, scores, landmarks = detector.detect_faces(your_image)
  ```
  
## WiderFace Validation Performance in single scale When using Mobilenet
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Pytorch (same parameter with Mxnet) | 88.67% | 87.09% | 80.99% |
| Pytorch (original image scale) | 90.70% | 88.16% | 73.82% |
| Mxnet | 88.72% | 86.97% | 79.19% |
| Mxnet(original image scale) | 89.58% | 87.11% | 69.12% |


## Image Alignment: similar transformation

  ```python
   from Retinaface import FaceDetector
   detector = FaceDetector(name='mobilenet', weight_path='mobilenet.pth', device='cuda')
   faces, boxes, scores, landmarks = detector.detect_align(frame)
  ```
- let us see a few examples

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res1.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/input2.jpg)|![image](https://github.com/sajjjadayobi/FaceRec/blob/master/imgs/res2.jpg)|

## Age & Gender Classification:
- very soon will be completed

## Facial Expression Recognition:

  ```python
   from FaceExpression import EmotionDetector
   from Retinaface import FaceDetector
   
   face_detector = FaceDetector(name='mobilenet', weight_path='weights/mobilenet.pth', device='cuda')
   emotion_detector = EmotionDetector(name='resnet34', weight_path='weights/resnet34.pth', device='cuda')
   
   faces, boxes, scores, landmarks = face_detector.detect_align(frame)
   list_of_emotions, probab = emotion_detector.detect_emotion(faces)
  ```

## Face Recognition
- very soon will be completed


## Citation
- Author = Sajjad Ayoubi
- Title = FaceLib
- url = https://github.com/sajjjadayobi/FaceLib/
- Year = 2020
