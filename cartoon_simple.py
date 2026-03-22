import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread("./image.png")

# 1. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 블러 (노이즈 제거)
gray_blur = cv2.medianBlur(gray, 5)

# 3. 엣지 검출 (카툰 외곽선)
edges = cv2.adaptiveThreshold(
    gray_blur,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    9,
    9
)

# 4. 색상 부드럽게 (카툰 느낌 핵심)
color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)

# 5. 엣지 + 컬러 합성
cartoon = cv2.bitwise_and(color, color, mask=edges)

# 결과 저장
cv2.imwrite("cartoon_output2.jpg", cartoon)

print("Saved: cartoon_output2.jpg")