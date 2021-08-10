# Import numpy and OpenCV
import numpy as np
import cv2
from cv2 import dnn_superres
sr = cv2.dnn_superres.DnnSuperResImpl_create()



cap = cv2.VideoCapture("/home/long/Desktop/manual/tanks.mov")
# image = cv2.imread("/home/long/darknet/data/dog.jpg")
# path = "/home/long/Desktop/manual/video_stabilty/edsr_x4.pb"
# sr.readModel(path)
# sr.setModel("edsr",4)
# result = sr.upsample(image)


# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# im = cv2.filter2D(image, -1, kernel) 
# while(1):
#     resized = cv2.resize(im, (960,720), interpolation = cv2.INTER_CUBIC)
#     resized1= cv2.resize(image, (960,720), interpolation = cv2.INTER_AREA)
       
#     cv2.imshow('inter', resized1)
#     cv2.imshow('cubic', result)
#     cv2.waitKey(30)


while (cap.isOpened()):
    _ , frame =  cap.read()
    frame = cv2.resize(frame, (640,480) )
    height, width = frame.shape[:2]
    center = (width/2, height/2)

    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1.1)
    rotated_image = cv2.warpAffine(src=frame, M=rotate_matrix, dsize=(width, height))
    # kernel = np.array([[-1,-1,-1], [-1,8.9,-1], [-1,-1,-1]])
    kernel = np.array([[0,-1,0], [-1,-5,-1], [0,-1,0]])
    im = cv2.filter2D(rotated_image, -1, kernel) 
 

    # resized = cv2.resize(frame, (960,720), interpolation = cv2.INTER_LANCZOS4)
    # resized1= cv2.resize(frame, (960,720), interpolation = cv2.INTER_AREA)
    # unsharp_image = cv2.addWeighted(frame, 1.5, frame, -0.5, 0, frame)
    cv2.imshow('Original image', frame)
    # cv2.imshow('Inter', resized1)
    cv2.imshow('cubic', im)
    # cv2.imshow('Rotated image', rotated_image)
    cv2.waitKey(30)

