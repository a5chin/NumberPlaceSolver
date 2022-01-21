import numpy as np
import cv2


gray = cv2.imread('data/problem/problem.png', cv2.IMREAD_GRAYSCALE)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
inv = cv2.bitwise_not(th)
contours, hierarchy = cv2.findContours(
    inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

max_area = 0

for cnt in contours:
    arclen = cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon=0.001 * arclen, closed=True)
    if len(approx_cnt) == 4:
        area = cv2.contourArea(approx_cnt)
        if area > max_area:
            max_area = max(area, max_area)
            contour = approx_cnt

print(np.array(contour).reshape(4, -1))
img = cv2.drawContours(inv, contour, -1, (255,255,255), 16)

pts1 = np.array([[581, 79], [66, 192], [198, 733], [759, 571]], dtype=np.float32)
pts2 = np.array([[800, 0], [0, 0], [0, 800], [800, 800]], dtype=np.float32)
mat = cv2.getPerspectiveTransform(pts1, pts2)
perspective_img = cv2.warpPerspective(img, mat, (800, 800))

cv2.imshow('sample', perspective_img)
cv2.waitKey(0)
