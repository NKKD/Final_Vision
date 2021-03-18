import cv2

cam = 0
cap = cv2.VideoCapture(cam)

ret,frame = cap.read()

cv2.imshow('cccc',frame)


