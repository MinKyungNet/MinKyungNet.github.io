---
layout: post
title: "Harris Conner Detection"
tags: [Harris]
categories: [Computer Vision]
---

오늘은 이 그림으로 Harris Conner Detection을 적용해보고자한다.
![image](https://user-images.githubusercontent.com/50114210/66966622-d7ff6b00-f0b8-11e9-8acc-ffc7b6a64ed1.png)      


# 코드
```
import cv2
import numpy as np

## 이미지 읽어오기
img = cv2.imread('chessboard.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Harris corner 검출
## https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html#cornerharris
dst = cv2.cornerHarris(src = gray,      # 8-bit or floating-point, 1-channel 이미지
                       blockSize = 2,   # 코너 검출 시 고려할 이웃 픽셀의 거리
                       ksize = 3,       # Sobel 마스크의 크기
                       k = 0.04)        # 해리스 코너 검출 조절 파라미터

# (optional) 특징점을 크게 보기 위하여 팽창 모폴로지 연산 수행
dst = cv2.dilate(dst, None)

## 원하는 특징을 검출하기 위한 임계값 설정
# img[dst > 0.01*dst.max()] = [0,0,255]

특징점 검출
# img[dst > 0.01] = [0,0,255]

## 결과 그리기
cv2.imshow('harris corner', img)
cv2.waitKey()

cv2.destroyAllWindows()
```
임계값을 어떻게 주냐에 따라서 결과가 달라진다.

# img[dst > 0.01]
![image](https://user-images.githubusercontent.com/50114210/66966687-1432cb80-f0b9-11e9-99ac-21b06b67326c.png)
특징점일 확률이 큰 점을 찾는다.

# img[dst == 0]
![image](https://user-images.githubusercontent.com/50114210/66966760-5825d080-f0b9-11e9-8c9a-849b9586beca.png)
특징점일 확률이 0 근처인 부분은 저주파 영역이다.

# img[dst < 0]
![image](https://user-images.githubusercontent.com/50114210/66966889-ca96b080-f0b9-11e9-889d-b7d9ba523934.png)     
