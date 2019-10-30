---
layout: post:
title: "Audio Classification using CNN"
tags: [CNN, Classification]
categories: [Code Review]
---

https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e          
이 블로그 글보고 리뷰한다.



# 1. 연구배경
Our eyes are best suited for looking. Our ears for hearing.         
If hypothetically, eyes were far more intelligent and quicker as compared to ears,             
wouldn’t it be more useful to send sound signals to our eyes for processing?   

우리의 눈은 보는 것을, 귀는 듣는 것을 잘한다.          
눈은 귀보다 훨씬 빠르고 지능적이다라는 가정을 한다면,          
**소리 신호를 눈으로 처리하는 것이 훨씬 유용하지 않을까?**

---

If sound frequencies could be turned into images in some manner and sent to eyes to differentiate,           
we might be able to understand a larger range of frequencies.                 
We might start understanding what dogs and dolphins say.           
We might hear the much discussed cosmic hum!            

**소리 주파수를 어떠한 방식으로 이미지로 변환하고 눈으로 보내서 구별**할 수 있다면,       
우리는 더 넓은 범위의 주파수를 이해하는 것이 가능할 것입니다.         
우리는 개나 돌고래의 말을 이해할 수 있게 될지도 모릅니다.         
우리는 많이 논의된 우주의 흥얼거리는 소리를 들을 수 있게 될지도 모릅니다!

---

Its capabilities for performing Machine Learning on images are well known and explored.             
Everyday some new research comes up to show an improvement in the algorithm or some new use case for it.

이것에 대한 머신러닝의 수행 능력은 많이 연구되고 잘 알려져있습니다.          
매일 새로운 연구들이 알고리즘의 향상 혹은 **새로운 케이스의 적용**을 보여주기위해 나타나고 있습니다.

---

It was with this backdrop that we decided to test something other than vision on CNN.            
We decided to test how CNN works for speech data.         
**With applications ranging from speech controls for online games to issuing commands to IoT devices, classifying speech data has a lot of charm.**

CNN에서 비전 말고 다른 것을 시험해 보기로 한 것도 이런 배경에서였습니다.        
우리는 **음성 데이터에서 CNN이 얼마나 잘 작동하는지 테스트**해보기로 결정했습니다.          
온라인 게임의 음성 컨트롤부터 IoT장비까지, 음성 데이터를 분류하는 것은 많은 매력을 가지고 있습니다.

# 2. 실험 과정
So we decided to try a little experiment of our own.         

1. We borrowed a Spoken Digit Dataset by Zohar Jackson
2. Converted each audio file to an image
3. Trained a CNN on these images
4. Tested the model using a laptop’s microphone

그래서 작은 실험을 해보기로 결정했습니다.

1. Zohar Jackson의 Spoken Digit Dataset을 가져오고
2. 각각의 파일들을 이미지로 변환하고
3. CNN을 이미지들로 학습시키고
4. 랩탑의 마이크로폰을 사용해서 모델을 테스트했습니다.

# 3. 데이터 셋

Dataset Details     

A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz.        
1. 4 male speakers with American accent


2. 2,000 recordings (50 of each digit per speaker)


3. English pronunciations

데이터셋의 세부사항
8kHz의 wav files파일에서 0부터 9까지를 말한 단순한 오디오 / 음성 데이터 셋들
1. 미국 악센트를 가진 4명의 남성이 말함


2. 2,000 개의 데이터 (사람당 각 숫자를 50번씩 말함 // 4 * 10 * 50)


3. 영어 발음

# 4. Plotting the audio signal amplitude

The first conversion that can be applied on any sound signal is to plot its samples’ amplitudes over time.            
Since the dataset contains digits from 0 to 9 we have 10 different types of sounds. 9 of those are shown below.

모든 음성 신호에서 적용될 수 있는 첫번째 변환은 **샘플의 amplitude을 시간에 따라 그래프**를 그린 것입니다.           
데이터 셋은 0부터 9까지의 숫자를 포함하고 있기 때문에, 우리는 10개의 sounds타입을 가지고 있습니다.             
그 중 2개는 아래에 있습니다.

![iamge](https://miro.medium.com/max/640/1*mhHHc8lKYFxviPJCbnbFDg.png)

![image](https://miro.medium.com/max/640/1*7UwVWJJSbKebL8M9RE5hKw.png)

Even though the signal for one example of number 4 (or any other digit) would be visually different from other examples for the same number, yet there would be similarities.      

예제의 숫자 4(혹은 다른 숫자)는 시각적으로 다른 샘플들과 다를 수 있지만, 거기에는 **유사한 점**이 있을 것입니다.

---

e.g. Notice the plot for 7 above has two major blobs corresponding to two syllables in **seven**

1. /ˈsɛv
2. (ə)n/

One can also plot the same digit spoken by different people.          
Seven when spoken by three different people looks different but has similar features — one initial bump followed by a narrow tail.

예를 들어, 숫자 7은 두 음절에 해당하는 두 가지 분포를 가지고 있습니다.          
다른 사람이 말한 같은 숫자를 그래프로 그려보면,           
**모양이 다르기는 하지만 유사한 특징**을 지니고 있습니다.          
첫 번째 큰 분포 뒤에 작은 분포가 뒤따르고 있습니다.

![image](https://miro.medium.com/max/640/1*Yiuw82HLHO4ZeFS0VCWDDw.png)
![image](https://miro.medium.com/max/640/1*7I4QeVcYpBC9jCUnMUn1fw.png)
![image](https://miro.medium.com/max/640/1*bc2phmVNXcm8r656ES1WDA.png)

Since CNNs are hungry for images, we want to transform the sound into an image.         
The audio signal can also be represented in yet another way.           
Instead of plotting the audio signal amplitude with respect to time, we can also plot it with respect to frequency.              
The plot we will make is called a spectrogram.

CNN은 이미지를 원하므로, 우리는 소리를 이미지로 변환하고 싶습니다.        
음성신호는 다른 방법으로도 나타낼 수 있습니다.           
음성 신호를 시간에 따른 amplitude에 대한 그래프로 그래는 대신,            
우리는 **빈도수**에 대해서 나타낼 수 있습니다.            
우리가 만들 이 그림은 **스펙트로그램**이라고 부립니다.       

# 5. Plotting the spectrogram
To plot the spectrogram, we break the audio signal into millisecond chunks and compute Short-Time Fourier Transform (STFT) for each chunk.         
We then plot this time chunk as a colored vertical line in the spectrogram.

스펙트로그램을 그리기위해, 오디오 신호를 millisecond chunk로 자르고, 각 chunk마다 STFT를 적용했습니다.          
그리고 이 chunk를 수직선으로 스펙토그램에 표시합니다.

---

**What is a spectrogram?**          
Spectrograms represent the frequency content in the audio as colors in an image.            
Frequency content of milliseconds chunks is stringed together as colored vertical bars.           
Spectrograms are basically two-dimensional graphs, with a third dimension represented by colors.

**스펙트로그램이란?**       
스펙트로그램은 오디오의 빈도수를 이미지로 나타냅니다.         
밀리초의 chunk의 빈도수 내용은 수직 색상 막대기로 표현됩니다.         
스펙트로그램은 기본적으로 3차원 색상으로 표현되는 2차원 그래프입니다.           
(RGB, time, frequency) > (3, time, frequency)

---

1. **Time** runs from left (oldest) to right (youngest) along the horizontal axis.         


2. The vertical axis represents **frequency**, with the lowest frequencies at the bottom and the highest frequencies at the top.        


3. The amplitude (or energy or “loudness”) of a particular frequency at a particular time is represented by the third dimension, **color**, with dark blues corresponding to low amplitudes and brighter colors up through red corresponding to progressively stronger (or louder) amplitudes.

1. 시간은 왼쪽에서 오른쪽으로 가로축입니다.


2. 세로축은 빈도수를 나타냅니다. 낮은 빈도수는 낮은 곳에 있고 높은 빈도수는 높은 곳에 있습니다.


3. 특정시간의 특정 빈도수의 진폭(혹은 에너지 혹은 시끄러움)은 3차원의 색으로 표현됩니다. 낮은 진폭에 해당하는 남색, 높은 진폭에 해당하는 빨간색이 점진적으로 표현됩니다.

![image](https://miro.medium.com/max/541/1*HaBKU1yJWyB9E_GaSPUkKg.jpeg)

For a running sound signal the spectrogram would be like this.       
Notice the blueness to the right of the 2nd image below.           
That is resulting from the low amplitude signals appearing later in time.

진행되는 소리 신호의 스펙토그램은 이와 같을 것입니다.          
두번째 이미지의 오른쪽에 나타나는 파란색에 주목해봅시다.       
그것은 나중에 나타나는 낮은 진폭의 신호에서 비롯되는 결과입니다.    

![gif](https://miro.medium.com/max/600/1*J4vtpmGBFz-eO8dE5W6pQA.gif)

Sending all of our 2000 sound signals through Python’s spectrogram function (in the pyplot library) we get 2000 sepctrograms.        
Now we may not be able to see many patterns in the images below but we hope the CNN will.          
If you look closely there are certain differences between each digit’s spectrogram.

우리의 2000개의 소리 신호를 파이썬의 스펙토그램 펑션에 보내서, 우리는 2000개의 스펙토그램을 얻었다.        
이제 우리는 아래의 이미지에서 많은 패턴을 보지 못하지만, CNN은 할 수 있기를 바란다.    
자세히보면 각 자리수의 스펙트럼에는 분명한 차이가 있다.

![image](https://miro.medium.com/max/1920/1*AUe0w9TasoElym-BLmqjiQ.png)
![image](https://miro.medium.com/max/1920/1*BK4nlxe1UB7gsdA83qDeTQ.png)
![image](https://miro.medium.com/max/1920/1*iCg-70HzN1bRlSLLRQGkYQ.png)

These spectrograms now become an image representation of our spoken digits.              
Every digit audio corresponds to a spectrogram.            
The hope is that spectrograms of 0’s sound would be similar across different speakers and genders.          
We also hope that despite the difference in volume, pitch, timbre etc. 0 spoken by anyone should have similarities with other 0 sounds.          
Same goes for the digits 1–9.

If there exist certain similarities between the same digit sounds across all these variables then we are sure CNN will catch them in the spectrogram.

위의 스펙토그램들은 우리의 숫자를 말한 데이터가 이미지로 나타난 것이다.           
모든 숫자 신호는 스펙토그램에 해당한다.    
0의 소리가 다른 사람이나 성별에 대해서도 비슷한 스펙토그램을 가지고 있기를 바란다.         
또한 다른 volume, pitch, timbre에서도 누군가 말한 0이 다른 0의 소리와 비슷하기를 바란다.        


1부터 9까지에도 그러하다.         


만약 모든 변수에 거쳐서 같은 숫자에 대해 확실한 유사성들이 존재한다면, 우리는 **CNN이 스펙토그램에서 그 차이를 잡아낼 것**이라고 확신한다.

# 6. Define the model layers
As seen above 1 audio has two kinds of images associated with it.        
위에서 보았듯이 한 개의 소리는 연관된 두 종류의 이미지를 가지고있다.

1. Audio signal : Amplitude v/s Time


2. Spectrogram : Freqeuncy Content v/s Time

---

Logically both of them can be used to train our CNN.        
We tried doing that and observed that pure audio signal yields a test-accuracy of 94% as compared to the spectrograms that yield a test-accuracy of 97%.

논리적으로 두 종류의 이미지 모두 CNN에서 학습될 수 있다.          
Audio signal의 이미지로 학습했을 때 94%의 정확도가 나왔고,         
Spectrogram의 이미지로 학습했을 때는 97%의 정확도가 나왔다.

---

The CNN we use has the following layers

1. Convolution layer with kernel size : 3x3


2. Convolution layer with kernel size : 3x3


3. Max Pooling layer with pool size : 2x2


4. Dropout layer


5. Flattening layer


6. 2 Dense layered Neural Network at the end

```python
#Define Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#Compile
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
print(model.summary())
#Train and Test The Model
model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
```

Printing the model summary Keras tells us about the various layers of our model

![image](https://miro.medium.com/max/1482/1*2K__-d1_0s35EmYCtk8d6Q.png)

Size of training and testing data.          

* Training Data : 1800 Images of Spectrograms. Each has a size of **34x50** pixels


* Training Data has 1800 labels corresponding to each of the images


* Test Data : 200 Images of Spectrograms. Each has a size of 34x50 pixels


* Test Data has 200 labels corresponding to each of the images

# 7. Training and Testing the Network

We issue the Keras command model.fit() and tell Keras to accept are test set as the validation set.           
(Look at the code snippet above).           
The model iterates over 10 epochs and improves its parameters until it gets its highest val_acc i.e. the test accuracy.            
We get 97% in this case

We make the network go through 10 epochs.           
Ideally one should have a larger number of epochs and should stop the network when test accuracy stops increasing.

![image](https://miro.medium.com/max/1600/1*wudvYfaZo_sACTOT5USMdQ.png)

As Explained in Keras FAQ the terms training loss and testing loss correspond to the metrics that show improvements in the model.        
Higher the loss worse the model accuracy.

![image](https://miro.medium.com/max/1542/1*tPmgFiRIk3CvAbo-7v8ZUA.png)

트레이닝 시에는 로스가 계속 줄겠지만              
test에서는 트레이닝했을 때 가장 좋은 로스로 loss를 산출할 것이기 때문에 트레이닝이 더 높은 경우도 있다.        
트레이닝 (10, 9, 8, 7, 5)            
test (5, 5, 5, 5, 5)

# 8. Final Testing
Finally once the model is created we save it locally.         
We don’t want to create model every time we want to test something.  
Whenever we want to perform some test we load the model and record an audio.

마지막으로 모델이 생성되는 우리는 로컬에 저장했다.         
우리는 우리가 모델을 테스트할 때마다 모델을 만다는 것을 원하지 않았다.       
우리가 테스트를 수행하고 싶을 때마다 우리는 모델을 불러와서 오디오를 녹음할 것이다.

---

As an example a test subject was asked to say out the number one.        
Its captured waveform is plotted.        
Then an algorithm runs that tries to locate the most important sound signal and crops the signal removing the leading and trailing spaces.         
**The red lines** indicate the final signal that is sent for spectrogram creation.

한 테스트의 예시로 우리는 실험자가 1을 말해달라고 요청받았다.          
이것의 캡쳐된 파형이 나타날 것이다.          
알고리즘이 가장 중요한 소리 신호를 찾아내고, 나머지 앞과 뒤 신호를 삭제할 것이다.         
빨간 선은 스펙토그램을 생성하기 위한 마지막신호를 나타낸다.

![image](https://miro.medium.com/max/640/1*tfxtM6C1sw6emkD_FNuHhA.png)

The spectrogram of the above capture is sent to the model to be evaluated.          
The model responds by giving its observation of which class it thinks the signal belongs to.         
Notice below that it says that the audio is one with a probability of 97% and 9 with a probability of 2%.

위의 캡쳐된 스펙토그램은 모델에 평가되어지기위해 전송한다.        
모델은 이 신호가 어떤 클래스에 속할지 판단한 것을 보낸다.     

![image](https://miro.medium.com/max/1120/1*Pk5Iw4aqsncUfc74zWqhGg.png)

# 9. Practical Observations

1. 9s and 1s sound similar and thus the model is sometimes confused between the two.


2. We trained the model on 4 different **male** speakers all with an **American accent.**


3. Even when tested on **female Indian** speaker the model work pretty well.


4. Code performs very poorly in the presence of background noise

1. 9와 1이 유사하여 모델이 종종 둘을 혼동한다.


2. 우리는 미국 남자 4명의 데이터로 모델을 학습했다.


3. 인도 여자의 데이터로 테스트해도 모델은 잘 작동했다.


4. 주변 소음에 굉장히 민감하게 반응한다.

![image](https://miro.medium.com/max/1680/1*5IuaOoALBJiHMKl0WO5lvA.png)

# 1. 임포트

```python
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import random
from skimage.measure import block_reduce

# To find the duration of wave file in seconds
import wave
import contextlib

# Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json

import time
import datetime
```

# 2. 변수 설정
```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
imwidth = 50
imheight = 34
total_examples = 2000
speakers = 4
examples_per_speaker = 50
tt_split = 0.1
num_classes = 10
test_rec_folder = "./testrecs"
log_image_folder = "./logims"
recording_directory = "../SoundCNN/recordings/"
num_test_files = 1

THRESHOLD = 1000
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
RATE = 8000  # 44100
WINDOW_SIZE = 50
CHECK_THRESH = 3
SLEEP_TIME = 0.5  # (seconds)
IS_PLOT = 1F
```
# 3. 조용한지 체크
```python
# Check for silence
def is_silent(snd_data):
    """
    snd_data : (array), 녹음된 음성
    
    Returns 'True' if below the 'silent' threshold
    """
    # 가장 큰 값이 임계점보다 작으면 True
    # THRESHOLD = 1000
    return max(snd_data) < THRESHOLD


"""
Record a word or words from the microphone and 
return the data as an array of signed shorts.
"""
```
# 4. 녹음

```python
def record():
    p = pyaudio.PyAudio()
    # CHUNK_SIZE = 512, FORMAT = pyaudio.paInt16, RATE = 8000  # 44100
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        # CHUNK_SIZE = 512
        snd_data = array('h', stream.read(CHUNK_SIZE))
        # 빅 엔디언이면 리틀 엔디언 방식으로 변경(byteswap())해줌
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)
        
        # 임계치 넘는지 확인, boolean
        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            break
    
    # FORMAT = pyaudio.paInt16
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # r : 녹음된 배열
    return sample_width, r
```

# 5. 녹음 데이터의 범위 정하기
```python
# Extract relevant signal from the captured audio
def get_bounds(ds):
    np.array(ds)
    lds = len(ds)
    count = 0
    ll = -1 # Lower Limit
    ul = -1 # Upper Limit

    # Lower Limit
    # WINDOW_SIZE = 50
    # 임계점을 세번 넘으면 세번의 Window전부터 라고 li에 기록
    for i in range(0, lds, WINDOW_SIZE):
        sum = 0
        for k in range(i, (i + WINDOW_SIZE) % lds):
            sum = sum + np.absolute(ds[k])
            
        # THRESHOLD = 1000
        if (sum > THRESHOLD):
            count += 1
            
        # CHECK_THRESH = 3
        if (count > CHECK_THRESH):
            ll = i - WINDOW_SIZE * CHECK_THRESH # i - 150
            break

    # Upper Limit
    # 임계점보다 세번 작아지면 세번의 Window전부터라고 ui에 기록
    count = 0
    for j in range(i, lds, WINDOW_SIZE):
        sum = 0
        for k in range(j, (j + WINDOW_SIZE) % lds):
            sum = sum + np.absolute(ds[k])
        if (sum < THRESHOLD):
            count += 1
        if (count > CHECK_THRESH):
            ul = j - WINDOW_SIZE * CHECK_THRESH # j - 150

        if (ul > 0 and ll > 0):
            break
    return ll, ul
```

# 6. 파일로 저장
```python
# Records from the microphone and outputs the resulting data to 'path'
def record_to_file(path):
    """
    path : 저장할 경로
    return : 0 or 1
    """
    
    sample_width, data = record()
    ll, ul = get_bounds(data)
    print(ll, ul)
    if (ul - ll < 100):
        return 0
    # nonz  = np.nonzero(data)
    # 관심있는 영역만 자르기
    ds = data[ll:ul]
    # IS_PLOT = 1
    if (IS_PLOT):
        plt.plot(data)
        plt.axvline(x=ll)
        # plt.axvline(x=ll+5000)
        plt.axvline(x=ul)
        plt.show()

    # data = pack('<' + ('h'*len(data)), *data)
    fname = "0.wav"
    # 경로 없으면 경로 만들고
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 파일 생성
    wf = wave.open(os.path.join(path, fname), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(ds)
    wf.close()
    return 1
```

# 7. 파일 길이 확인
```python
# Function to find the duration of the wave file in seconds
def findDuration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw = f.getsampwidth()
        chan = f.getnchannels()
        # 재생시간 = 파일의 크기 / 비트 레이트
        duration = frames / float(rate)

        return duration
```        
**파일명: stop.mp3**

재생시간: 4분 59초 (299초)   
비트레이트: 56Kbps (CBR)        
채널수: 2 ch             
비트: 16 bit           
샘플레이트:22050 Hz         
크기: 2,094,939byte           
        
**MP3 파일 크기 계산**
     
먼저 MP3 파일 크기를 계산해 보자. 이때 필요한 값이 재생시간과 비트레이트이다.         

파일크기 = 재생시간 * 비트레이트 / 8           

8를 나눠준 이유는 비트에서 바이트로 변환하기 위한 것이다.          

     
299 * 56000 / 8 = 2,093,000 (byte)           

실제 값과 약간의 오차가 있는 이는 재생 시간이 1초 이하의 시간을 고려하지 않았고,           
MP3의 테그와 헤더 정보를 고려하지 않았기 때문이다.             
비트 레이트는 실제로 처리되는 데이터 크기이므로 실제 재생시간과 곱함으로서 실제 데이터 크기를 알아낼 수 있다.

https://ospace.tistory.com/101

# 8. 스펙트럼 이미지로 만들기
```python
# Plot Spectrogram
def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    # 길이 찾기
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    # 스펙토그램으로 만들기
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75, 0.5]

    fig.canvas.draw()
    size_inches = fig.get_size_inches()
    dpi = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    # 3차원의 이미지 배열
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray
```

# 9. 그레이 스케일로 변환
```python
# Convert color image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
```    
사람의 눈은 초록색, 빨간색, 파란색 순으로 색에 민감하다.         
따라서 그레이 스케일로 변환할때         
(R + G + B) / 3을 사용할 수도 있지만         
(0.3*R + 0.6*G + 0.1*B) / 3을 사용하면             
인간의 시각에 적합한 그레이 스케일로 변환할 수 있다.

# 10. 정규화
```python
# Normalize Gray colored image
# array의 값들을 0부터 1까지로 만들어준다.
def normalize_gray(array):
    return (array - array.min()) / (array.max() - array.min())
```
![image](https://user-images.githubusercontent.com/50114210/67631261-373e5600-f8d7-11e9-86e6-a58ca795df55.png)

# 11. 데이터셋 만들기
```python
# Split the dataset into test and train sets randomly
def create_train_test(audio_dir):
    # .wav 파일 리스트 받아오기
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    # 파일 정렬
    file_names.sort()
    test_list = []
    train_list = []
    
    # total_examples = 2000, examples_per_speaker = 50, tt_split = 0.1
    # for i in range(40):
    for i in range(int(total_examples / examples_per_speaker)):
        [i*51 : (i + 1) * 50]
        test_list.extend(random.sample(file_names[(i * examples_per_speaker + 1):(i + 1) * examples_per_speaker],
                                       int(examples_per_speaker * tt_split)))
    
    # test에 없는걸 train으로
    train_list = [x for x in file_names if x not in test_list]

    y_test = np.zeros(len(test_list))
    y_train = np.zeros(len(train_list))
    x_train = np.zeros((len(train_list), imheight, imwidth))
    x_test = np.zeros((len(test_list), imheight, imwidth))
    
    # test 데이터 셋
    for i, f in enumerate(test_list):
        y_test[i] = int(f[0])
        spectrogram = graph_spectrogram(audio_dir + f)
        graygram = rgb2gray(spectrogram)
        normgram = normalize_gray(graygram)
        norm_shape = normgram.shape
        if (norm_shape[0] > 150):
            continue
        redgram = block_reduce(normgram, block_size=(3, 3), func=np.mean)
        x_test[i, :, :] = redgram
        print("Progress Test Data: {:2.1%}".format(float(i) / len(test_list)), end="\r")

    # train 데이터 셋
    for i, f in enumerate(train_list):
        y_train[i] = int(f[0])
        spectrogram = graph_spectrogram(audio_dir + f)
        graygram = rgb2gray(spectrogram)
        normgram = normalize_gray(graygram)
        norm_shape = normgram.shape
        if (norm_shape[0] > 150):
            continue
        redgram = block_reduce(normgram, block_size=(3, 3), func=np.mean)
        x_train[i, :, :] = redgram
        print("Progress Training Data: {:2.1%}".format(float(i) / len(train_list)), end="\r")

    return x_train, y_train, x_test, y_test
```

# 12. CNN 모델

```python
# Create Keras Model
def create_model(path):
    x_train, y_train, x_test, y_test = create_train_test(path)

    print("Size of Training Data:", np.shape(x_train))
    print("Size of Training Labels:", np.shape(y_train))
    print("Size of Test Data:", np.shape(x_test))
    print("Size of Test Labels:", np.shape(y_test))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    input_shape = (imheight, imwidth, 1)
    batch_size = 4
    epochs = 1

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # classification 모델 답게 cross-entropy
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    
    # 모델 요약한거 출력해줌
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
    return model
```   
# 13. 녹음된 오디오 파일에서 wave 데이터 추출
```python
# Extract wave data from recorded audio
def get_wav_data(path):
    # 오디오 > 스펙토그램 > 그레이 스케일 > 정규화
    input_wav = path
    spectrogram = graph_spectrogram(input_wav)
    graygram = rgb2gray(spectrogram)
    normgram = normalize_gray(graygram)
    norm_shape = normgram.shape
    # print("Spec Shape->", norm_shape)
    if (norm_shape[0] > 100):
        redgram = block_reduce(normgram, block_size=(26, 26), func=np.mean)
    else:
        redgram = block_reduce(normgram, block_size=(3, 3), func=np.mean)
    redgram = redgram[0:imheight, 0:imwidth]
    red_data = redgram.reshape(imheight, imwidth, 1)
    empty_data = np.empty((1, imheight, imwidth, 1))
    empty_data[0, :, :, :] = red_data
    new_data = empty_data
    return new_data
```
**block_reduce**      
skimage.measure.block_reduce(image, block_size, func=<function sum>, cval=0)          
#로컬 블록에 함수를 적용하여 이미지를 다운 샘플링합니다.           
```python
>>> from skimage.measure import block_reduce
>>> image = np.arange(3*3*4).reshape(3, 3, 4)
>>> image 
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]],
       [[24, 25, 26, 27],
        [28, 29, 30, 31],
        [32, 33, 34, 35]]])
>>> block_reduce(image, block_size=(3, 3, 1), func=np.mean)
array([[[ 16.,  17.,  18.,  19.]]])
>>> image_max1 = block_reduce(image, block_size=(1, 3, 4), func=np.max)
>>> image_max1 
array([[[11]],
       [[23]],
       [[35]]])
>>> image_max2 = block_reduce(image, block_size=(3, 1, 4), func=np.max)
>>> image_max2 
array([[[27],
        [31],
        [35]]])
```
    
https://code-examples.net/ko/docs/scikit_image/api/skimage.measure#skimage.measure.block_reduce

# 14. 모델 저장하기
```python
# Save created model
def save_model_to_disk(model):
    # serialize model to JSON
    model_json = model.to_json()
    
    # with as로 열어서 따로 close안 해준다.
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
# 15. 모델 불러오기
# Load saved model
def load_model_from_disk():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

# 모델이랑 가중치는 따로 처리해야하는구나
```

# 16. 메인
```python
# 학습시킬 때의 main이 아니라 test하는 용도임
if __name__ == '__main__':
    while (1):
        # 말한거 녹음해서 파일만들고
        # SLEEP_TIME = 0.5  # (seconds)
        time.sleep(SLEEP_TIME)
        if (os.path.isfile('model.json')):
            print("please speak a word into the microphone")
            success = record_to_file(test_rec_folder)
            if (not success):
                print(" Speak Again Clearly")
                continue
        else:
            print("********************\n\nTraining The Model\n")
        
        # 모델 불러오고
        if (os.path.isfile('model.json')):
            model = load_model_from_disk()
        else:
            model = create_model(recording_directory)
            save_model_to_disk(model)
        
        # 돌리기
        # fname = 'r4.wav'
        # new_data = get_wav_data(fname)
        for i in range(num_test_files):
            # for i in range(1):
            fname = str(i) + ".wav"

            new_data = get_wav_data(os.path.join(test_rec_folder, fname)) # 녹음된 파일 스펙토그램으로 만들고
            predictions = np.array(model.predict(new_data)) # 모델한테 예측시키고
            maxpred = predictions.argmax()  # 가장 확률이 높은거
            normpred = normalize_gray(predictions) * 100 # 예측했던 애들
            predarr = np.array(predictions[0])
            sumx = predarr.sum()
            print("TestFile Name: ", fname, " The Model Predicts:", maxpred)
            for nc in range(num_classes):
                confidence = np.round(100 * (predarr[nc] / sumx))
                print("Class ", nc, " Confidence: ", confidence)
            # print("TestFile Name: ",fname, " Values:", predictions)
            print("_____________________________\n")
```            
![image](https://miro.medium.com/max/1120/1*Pk5Iw4aqsncUfc74zWqhGg.png)
  
