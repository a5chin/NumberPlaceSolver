from calendar import c
import cv2

gray = cv2.imread('../../data/problem/example2.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(
    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

img = cv2.drawContours(gray, contours, -1, (0,255,0), 3)

cv2.imshow('sample', img)
cv2.waitKey(0)
