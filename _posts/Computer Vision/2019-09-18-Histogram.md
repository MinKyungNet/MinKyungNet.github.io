---
layout: post
title: "Histogram"
tags: [Histogram, Min-Max strectching, Percentile strectching, equalization]
categories: [Computer Vision]
---
# 사용하는 이미지
![image](https://user-images.githubusercontent.com/50114210/65142177-17508280-da4d-11e9-83ff-3d0efc073c06.png)        
이 이미지로 히스토그램그리기, min-max stretching, Percentile strectching, equalization을 할거다!

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 이미지 파일 읽어오기
img = cv2.imread('landscape.jpg') # cv2.IMREAD_COLOR (default) / cv2.IMREAD_GRAYSCALE / cv2.IMREAD_UNCHANGED
# img = cv2.imread('lung.jpg')

# 히스토그램 생성하기
# 두번째 인자는 몇 단계로 쪼갤래?
# 세번째 인자는 범위는 얼마만큼이니?
hist, bins = np.histogram(img, 256, [0,256])
# bins x축이 나온다.

# 정규화된 누적 히스토그램 생성하기
# hist값의 최대가 8만임
# y값이 8만정도의 범위를 가지게 되겠지
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max() # 정규화 해주는 이유 => 히스토그램과 같은 스케일로 표현하기 위함

print(img.flatten())
print(bins)
# 히스토그램 시각화하기
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0,256], color='r')
plt.xlim([0,256])
plt.legend(['cdf', 'histogram'], loc='upper left')
plt.show()

# 이미지 창 띄우기
cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()
```
![image](https://user-images.githubusercontent.com/50114210/65142142-00aa2b80-da4d-11e9-868a-8091f604a01d.png)

# Min-max stretching
![image](https://user-images.githubusercontent.com/50114210/65142360-7f06cd80-da4d-11e9-91a5-d1131765e1bd.png)        
이 공식을 사용한다!       
### 공식 설명
1. 픽셀 값 - 가장 작은 픽셀 값 : 히스토그램 그래프를 0으로(왼쪽으로) 붙인다.
2. 가장 큰 픽셀 값 - 가장 작은 픽셀 값 : 픽셀들이 분포하고 있는 '범위'    
3. 픽셀 값 - 가장 작은 픽셀 값 / 가장 큰 픽셀 값 - 가장 작은 픽셀 값 : 픽셀의 범위를 0부터 1로 만든다.   
4. 곱하기 255 : 0부터 1로 만든 픽셀 값의 범위를 0부터 255로 만들어준다.

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 이미지 파일 읽어오기
img = cv2.imread('landscape.jpg')

# min-max stretching 수행하기
img_minmax = (img - img.min()) / (img.max() - img.min()) * 255.0
img_minmax = img_minmax.astype(np.uint8)

# 히스토그램 생성하기
hist, bins = np.histogram(img_minmax, 256, [256])

# 정규화된 누적 히스토그램 생성하기
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# 히스토그램 시각화하기
plt.plot(cdf_normalized, color='b')
plt.hist(img_minmax.flatten(), 256, [0,256], color='r')
plt.xlim([0,256])
plt.legend(['cdf', 'histogram'], loc='upper left')
plt.show()

# 이미지 창 띄우기
cv2.imshow('image', img_minmax)
cv2.waitKey()

cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/50114210/65142818-867aa680-da4e-11e9-897e-faab6a8e12ea.png)        

# Percentile stretching
민맥스 스트레칭의 단점은 outlier가 있다면 스트레칭이 제대로 되지 않는다는 것이다.    
예를 들어, 가장 작은 픽셀이 0하나, 가장 큰 픽셀이 255하나가 있고    
나머지 픽셀들이 100과 200사이에 몰려있다면    
민맥스 스트레칭을 한다고해도 영상은 변화하지 않는다.
그래서 일정 비율의 화소를 0과 255로 만들어서 무시하고 나머지 범위에서 스트레칭하는 것을
percentile strectching이라고 한다.

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def percentile_stretching(img, p_low, p_high):
    '''
    percentile stretching 함수
    
    Parameters
    ----------
    p_low: int (0~100)
        percentile stretching을 수행하기 위한 하위 백분율
    p_high: int (0~100)
        percentile stretching을 수행하기 위한 상위 백분율
    
    Returns
    -------
    img_ptile: array
        percentile stretching 처리된 이미지 배열
    
    '''
    
    
    return img_ptile
    

# 이미지 파일 읽어오기
img = cv2.imread('landscape.jpg')

# percentile stretching 수행하기
img_percentile = percentile_stretching(img, 3, 97)

# 히스토그램 생성하기
hist, bins = np.histogram(img_percentile, 256, [0,256])

# 정규화된 누적 히스토그램 생성하기
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# 히스토그램 시각화하기
plt.plot(cdf_normalized, color='b')
plt.hist(img_percentile.flatten(), 256, [0,256], color='r')
plt.xlim([0,256])
plt.legend(['cdf', 'histogram'], loc='upper left')
plt.show()

# 이미지 창 띄우기
cv2.imshow('image', img_percentile)
cv2.waitKey()

cv2.destroyAllWindows()
```
![image](https://user-images.githubusercontent.com/50114210/65142858-9d20fd80-da4e-11e9-8dfe-44375b7d73fb.png)         

# Histogram equalization
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 이미지 파일 읽어오기
img = cv2.imread('landscape.jpg')

# 히스토그램 평활화 수행하기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
img_equalized = cv2.merge([equalized, equalized, equalized])

# 히스토그램 생성하기
hist, bins = np.histogram(img_equalized, 256, [0,256])

# 정규화된 누적 히스토그램 생성하기
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# 히스토그램 시각화하기
plt.plot(cdf_normalized, color='b')
plt.hist(img_equalized.flatten(), 256, [0,256], color='r')
plt.xlim([0,256])
plt.legend(['cdf', 'histogram'], loc='upper left')
plt.show()

# 이미지 창 띄우기
cv2.imshow('image', img_equalized)
cv2.waitKey()

cv2.destroyAllWindows()
```
![image](https://user-images.githubusercontent.com/50114210/65142921-b629ae80-da4e-11e9-8034-6d2df38e99a1.png)        







