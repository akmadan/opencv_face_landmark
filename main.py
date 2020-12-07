#pip install opencv-contrib-python
# pip install opencv-python
import cv2

img = cv2.imread('input.jpg')

img = cv2.resize(img,(512,512))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

harcascade = r'C:\Users\Admin\Desktop\PythonProject\OpenCVTutorial\haarcascade_frontalface_default.xml'

detector = cv2.CascadeClassifier(harcascade)
faces = detector.detectMultiScale(img_gray)

LBFmodel = r'C:\Users\Admin\Desktop\PythonProject\OpenCVTutorial\lbfmodel.yaml'
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)
_,landmarks = landmark_detector.fit(img_gray,faces)
for landmark in landmarks:
    for x,y in landmark[0]:
        cv2.circle(img,(x,y),2,(0,255,0),2)

cv2.imshow('original', img)
cv2.waitKey(0)