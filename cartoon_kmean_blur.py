import cv2
import numpy as np
import sys

# 이미지 불러오기
input_path = sys.argv[1] if len(sys.argv) > 1 else "./image.png"
output_path = sys.argv[2] if len(sys.argv) > 2 else "cartoon_improved.jpg"

img = cv2.imread(input_path)
if img is None:
    print(f"Error: cannot read {input_path}")
    sys.exit(1)

# === 1. 색상 평탄화 (Bilateral Filter 반복 적용) ===
# 반복 적용으로 실사 그라데이션/텍스처를 강하게 평탄화
color = img.copy()
for _ in range(7):
    color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

# === 2. 엣지 검출 (Canny + 형태학적 처리) ===
# Adaptive Threshold 대신 Canny를 사용하여 주요 윤곽선만 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 강한 블러로 세밀한 텍스처(거미줄 패턴 등)를 사전 제거
gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
gray_blur = cv2.medianBlur(gray_blur, 7)

# Canny 엣지: 높은 threshold로 주요 윤곽만 추출
edges = cv2.Canny(gray_blur, 80, 150)

# 팽창(dilate)으로 엣지선을 두껍게 → 만화 외곽선 느낌
edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

# 엣지 반전 (배경=흰, 엣지=검정 → 마스크용)
edges_inv = cv2.bitwise_not(edges)

# === 3. 어두운 영역 보정 ===
# CLAHE로 어두운 영역의 대비를 개선하여 엣지 노이즈 억제
lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
color = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# === 4. K-Means 색상 양자화 ===
# 색상 수를 제한하여 만화 특유의 단색 영역 표현
data = color.reshape((-1, 3)).astype(np.float32)
k = 12  # 색상 수
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
centers = np.uint8(centers)
color_quantized = centers[labels.flatten()].reshape(img.shape)

# === 5. 엣지 + 컬러 합성 ===
edges_3ch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
cartoon = cv2.bitwise_and(color_quantized, edges_3ch)

# 결과 저장
cv2.imwrite(output_path, cartoon)
print(f"Saved: {output_path}")
