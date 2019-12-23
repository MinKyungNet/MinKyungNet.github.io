---
layout: post
title: "Real-world Anomaly Detection in Surveillance Videos"
tags: [Surveillance, Video, anomaly]
categories: [Paper Review]
---

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vR_gaenQVHSntwf14wctDneObjtcZIU7dXd4jX7ARTFEPzHQMvNqnA3s1eKXewkJQ/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

https://arxiv.org/abs/1801.04264

Abstract
to avoid annotationg the anomalous segments or clips in training videos, which is very time comsuming, we propose to learn anomaly through the deep multiple instance ranking framework by leveraging weakly labeled training videos, i.e. the training labels (anomalous or normal) are at video-level instead of clip-level                
라벨링하는 노가다 줄이려고 비디오 단위로 라벨링

exploiting : 착취하는, (부당하게)이용하는, 최대한 잘 활용하다
instance : 사례
leveraging : 활용하기

@ multiple instance learning (MIL)?

1.	Introduction
glaring : 확연한
deficiency : 결점

The goal of a practical anomaly detection system is to timely signal an activity that deviates normal patterns and identify the time window of the occurring anomaly             
실용적인 이상탐지 시스템은 정상에서 벗어난 패턴을 적시에 탐지하고 발생하는 이상 징후의 시간 창을 식별하는 것이다.

coarse : 거친

Therefore, anomaly detection can be considered as coarse level video understanding, which filters out anomalies from normal patterns. Once an anomaly is detected, it can further be categorized into one of the specific activities using classification techniques         
정상 비정상으로 분류, 비정상으로 분류된 애들을 다시 클래시피케이션

Real-world anomalous events are complicated and diverse. It is difficult to list all of the possible anomalous events         
실제 문제는 복잡하고 다양해서 가능한 모든 비정상을 나열하는 것은 어렵다.

Therefore, it is desirable that the anomaly detection algorithm does not rely on any prior information about the events. In other words, anomaly detection should be done with minimum supervision.         
그래서 이전 정보에 기대면 안 된다 즉, 감독이 덜 된 상태로 가능해야한다.

deviates from the learned normal patterns would be considered as an anomaly.         
이전의 모델들은 정상에서 벗어난 애들을 비정상이라고 봤음

it is very difficult or impossible to define a normal event which takes all possible normal patterns/behaviors into account           
근데 모든 상황에서 그런게 가능한게 아니쟈너 ㄴㄴ

More importantly, the boundary between normal and anomalous behaviors is often ambiguous.           
정상이랑 비정상이 헷갈릴 수도 있지    

In addition, under realistic conditions, the same behavior could be a normal or an anomalous behavior under different conditions.                   
그리고 같은 행동이라도 상황에 따라 다르지

we propose an anomaly detection algorithm using weakly labeled training videos.                       
트레이닝 데이터에 약간 라벨링해서 이상 탐지 알고리즘을 맨드러씀

That is we only know the video-level labels,           
우리가 아는 한에선 유일한 비디오 레벨 라벨링임

i.e. a video is normal or contains anomaly somewhere, but we do not know where.            
비정상인건 아는데 어디서 비정상인줄은 모르지

intriguing : 흥미로운

@ multiple instance learning (MIL) ??

we propose to learn anomaly through a deep MIL framework by treating normal and anomalous surveillance videos as bags and short segments/clips of each video as instances in a bag.              
MIL에 넣어서 정상과 비정상 비디오들을 작은 단위로 나눔

Based on training videos, we automatically learn an anomaly ranking model that predicts high anomaly scores for anomalous segments in a video.                 
그래서 이상 포인트가 높은 애들을 예측해서 만들 수 있음

컨트리뷰션 정리해줌 ㅠ 감덩
1.	We propose a MIL solution to anomaly detection by leveraging only weakly labeld training videos. We propose a MIL ranking loss with sparsity and smoothness constraints for a deep learning network to learn anomaly scores for video segments. To the best of our knowledge, we are the first to formulate the video anomaly detection problem in the context of MIL
2.	We introduce a large-scale video anomaly detection dataset consisting of 1900 real-world surveillance videos of 13 different anomalous events and normal activities captured by surveillance cameras. It is by far the largest dataset with more than 15 times videos than existing anomaly datasets and has a total of 128 hours of videos.
3.	Experimental results on our new dataset show that our proposed method achieves superior performance as compared to the state-of-the-art anomaly detection approaches.
4.	Our dataset also serves a challenging benchmark for activity recognition on untrimmed videos due to the complexity of activities and large intra-class variations. We provide results of baseline methods C3D, and TCNN on recognizing 13 different anomalous activities.

2.	Related Work
Anomaly detection.
이전엔
1.	팔다리와 모션으로 봄
2.	오디오까지 넣음
3.	폭력과 안 폭력 클래스피케이션함
4.	노멀한 행동을 트래킹해서 안 노멀한애 디텍트

Our approach not only considers normal behaviors but also anomalous behaviors for anomaly detection, using only weakly labeled training data.         
근데 우리는 대충 라벨링해서 둘다 볼거야

Ranking
retrieval : 회수하다
이전엔
1.	SVM
2.	linear programming
3.	딥러닝

In contrast to the existing methods, we formulate anomaly detection as a regression problem in the ranking framework by utilizing normal and anomalous data.          
우리는 정상적이고 비정상적인 데이터를 활용하여 순위 프레임워크의 회귀 문제로서 이상 징후 탐지를 공식화한다.

alleviate : 완화시키다.

To alleviate the difficulty of obtaining precise segment-level labels to learn the anomaly model and detect video segment level anomaly during testing.             
네 위크 라벨링했다고 합니다; 즉당히 하세연;


3.	Proposed Anomaly Detection Method
The proposed approach (summarized in Figure 1) begins with dividing surveillance videos into a fixed number of segments during training.        
제안된 접근법에서는 cctv train 영상을 고정된 수의 세그먼트로 나누는 것부터 시작한다.

These segments make instances in a bag.         
세그먼트들은 instances in a bag을 생선한다.

Using both positive (anomalous) and negative (normal) bags, we train the anomaly detection model using the proposed deep MIL ranking loss.          
두개의 가방을 사용해서 우리는 anomaly detection model을 deep MIL ranking loss를 이용하여 학습한다.

3.1	Multiple Instance Learning
In standard supervised classification problems using support vector machine, the labels of all positive and negative examples are available and the classifier is learned using the following optimization function         
SVM에선 긍정 부정 예시 둘다 사용하고 분류기는 다음과 같은 최적화 함수를 따른다.
그리고 잘 하려면 라벨링이 잘 되야할텐데 쌉노가다 에바참치;
MIL relaxes the assumption of having these accurate temporal annotations.          
MUL은 이런 정확한 라벨링이 필요하다는 가정을 완화한다.

In MIL, precise temporal locations of anomalous events in videos are unknown. Instead, only video-level labels indicating the presence of an anomaly in the whole video is needed.          
그냥 비디오하나 통에다가 anormaly이 존재한다 팡! 박아두면 된다

B(a) : 비정상
B(n) : 정상
instance : 한 세그먼트 영상

Then, we represent a positive video as a positive bag Ba, where different temporal segments make individual instances in the bag, (p 1 , p2 , . . . , pm), where m is the number of instances in the bag.        
B(a)는 p1, p2들로 구성되어있고 m은 bag의 인스턴스 개수

We assume that at least one of these instances contains the anomaly            
최소한 한 세그먼트는 비정상일거다

정상 영상도 마찬가지고 거기 안엔 비정상 인스턴스는 없겠지

Since the exact information (i.e. instance-level label) of the positive instances is unknown, one can optimize the objective function with respect to the maximum scored instance in each bag [4]:         
비정상 인스턴스의 정확한 정보를 알 수 없으므로 각 가방에서 최대 점수화된 인스턴스에 대한 객관적 기능을 최적화할 수 있다.

YBj : bag-lebel label
z : total number of bags
yi : label of each example
공집합(x) : feature representation of an image patch or a video segment
b : bias
k : total number of training examples
w : classifier to be learned

w*x – b

영상 안에 있는 인스턴스 중에 가장 큰 값
근데 그게 0보다 작으면 걍 0사용
아니면 ybj값 사용
그걸 개수 z로 나눔
그리고 그걸 최소화하지
L2 사용했네요

3.2	Deep MIL Ranking Model
이전엔

subjective : 주관적인
Anomalous behavior is difficult to define accurately [9], since it is quite subjective and can vary largely from person to person        
비정상이란게 사람마다 너무 다르니까 정확하게 정의하기가 어려움
 
Further, it is not obvious how to assign 1/0 labels to anomalies         
점수로 매기기도 명확하지가 않어;

Moreover, due to the unavailability of sufficient examples of anomaly, anomaly detection is usually treated as low likelihood pattern detection instead of classification problem         
게다가 비정상 데이터의 양도 적어서 분류 문제 대신에 로그라이크리후드 패턴으로 대체됐다

우리는
In our proposed approach, we pose anomaly detection as a regression problem.           
우리는 회귀 문제로 풀었다.

anomalous : 변칙

We want the anomalous video segments to have higher anomaly scores than the normal segments.             
우리는 변칙저인 비디오 세그먼트가 일반 세그먼트보다 더 높은 이상 점수를 갖기를 원한다.

encourages : 장려하다

The straightforward approach would be to use a ranking loss which encourages high scores for anomalous video segments as compared to normal segments          
간단한 접근 방식은 정상 세그먼트에 비해 비정상적인 비디오 세그먼트에 높은 점수를 부여하는 순위 손실을 사용하는 것이다.

f(Va) > f(Vn),

Va : 비정상
Vn : 정상
f(Va) : 비정상 predicted scores
f(Vn) : 정상 predicted scores

absence : 결석

However, in the absence of video segment level annotations, it is not possible to use Eq. 3. Instead, we propose the following multiple instance ranking objective function:         
인스턴스 빠진게 있어서 아래 수식으로 대체

max i∈Ba f(V i a ) > max i∈Bn f(V i n ),

max is taken over all video segments in each bag           
최대값은 가방에 있는 모든 비디오 세그먼트에 의해서 구해짐  

Instead of enforcing ranking on every instance of the bag, we enforce ranking only on the two instances having the highest anomaly score respectively in the positive and negative bags.                   
가방의 모든 인스턴스의 랭킹을 올리는 것이 아니라, 높은 스코어 받은 정상 비정상 하나씩 올린다.

이상한 가방에서 나온 애는 트루 파지티브
정상 가방에서 나온애는 가장 이상해보이지만 실제로는 정상

This negative instance is considered as a hard instance which may generate a false alarm in anomaly detection.             
정상가방에서 나오 ㄴ가장 이상한애가 아마 false alarm으로 디텍트 되겠지

we want to push the positive instances and negative instances far apart in terms of anomaly score.           
우리는 긍정과 부정을 최대한 떨어트려 놓고싶다.


Our ranking loss in the hinge-loss formulation is therefore given as follows:                 
힌지 로스 공식에서 우리의 순위 로스는 다음과 같다.

l(Ba, Bn) = max(0, 1 − max i∈Ba f(V i a ) + max i∈Bn f(V i n ))

문제
One limitation of the above loss is that it ignores the underlying temporal structure of the anomalous video             
위 손실의 한 가지 제한은 변칙적인 비디오의 기본 시간 구조를 무시하는 것이다.

First, in real-world scenarios, anomaly often occurs only for a short time.                  
비정상적인 장면은 자주 한 프레임에 나타난다.

In this case, the scores of the instances (segments) in the anomalous bag should be sparse, indicating only a few segments may contain the anomaly.               
이 경우에 비정상 가방에서 인스턴스의 점수는 희박해야 하며, 일부 세그먼트만 이상 징후를 포함할 수 있음을 나타낸다.

Second, since the video is a sequence of segments, the anomaly score should vary smoothly between video segments.           
두번째로 비정상 장면은 비디오 내에서 스무스하게 존재한다.

해결
Therefore, we enforce temporal smoothness between anomaly scores of temporally adjacent video segments by minimizing the difference of scores for adjacent video segments         
따라서 인접한 비디오 세그먼트에 대한 점수 차이를 최소화하여 시간적으로 인접한 비디오 세그먼트의 이상 점수 간에 시간적 매끄러움을 적용하고 있다.

incorporating : 통합의

By incorporating the sparsity and smoothness constraints on the instance scores, the loss function becomes            
가끔 나타난거랑 스무스하게 나타난거랑 합치기

l(Ba, Bn) = max(0, 1 − max i∈Ba f(V i a ) + max i∈Bn f(V i n )) 
+λ1 1 z }| { (nX−1) i (f(V i a ) − f(V i+1 a ))2 + λ2 2 z }| { Xn i f(V i a ),

앞의 텀은 일시적 스무딩
뒤어 텀은 부분적으로 드러남

In this MIL ranking loss, the error is back-propagated from the maximum scored video segments in both positive and negative bags.          
에러는 양쪽 가방의 가장 큰 세그먼트 스코어로 백프로파게이션된다.

By training on a large number of positive and negative bags, we expect that the network will learn a generalized model to predict high scores for anomalous segments in positive bags     
트레이닝하면서 이상한 애들이 높은 스코어를 가지면 좋겠음

L(W) = l(Ba, Bn) + kWkF ,
로스 함수…
	
Bags Formations.
We divide each video into the equal number of non-overlapping temporal segments and use these video segments as bag instances       
우리는 비디오를 같은 수지로 겹치지 않게 일시 세그먼트로 자르고 그 비디오 조각들을 가방은 인스턴스로 사용했다.

Given each video segment, we extract the 3D convolution features           
각 비디오 세그먼트들에 대해 우리는 3d convolution features로 특징을 추출했다.

We use this feature representation due to its computational efficiency, the evident capability of capturing appearance and motion dynamics in video action recognition.           
우리는 이런 특징 대표방법을 계산 효율 성 때문에 사용했고, 비디오 액션 인식에서 외관과 움직임의 역동성을 포착하는 명백한 능력이 있다.

4.	Dataset
4.1	Previous datasets
4.2.	Our dataset
It consists of long untrimmed surveillance videos which cover 13 realworld anomalies,           
다듬어지지 않은 리얼 월드 이상 13개 종류 포함

These anomalies are selected because they have a significant impact on public safety                   
쟤네들이 공공안전에서 중요해서 사용 됨

Video collection
To ensure the quality of our dataset, we train ten annotators (having different levels of computer vision expertise) to collect the dataset.            
우리의 데이터 셋의 질을 입증하기 위해 교육받은 컴퓨터비전 전문가 10명한테 라벨링 시킴; 미친놈들 아냐

French, Russian, Chinese, etc

following conditions: manually edited, prank videos, not captured by CCTV cameras, taking from news, captured using a hand-held camera, and containing compilation the anomaly is not clear         

Annotation
in order to evaluate its performance on testing videos, we need to know the temporal annotations,                
트레이닝 셋엔 비디오 레벨로 했지만 테스트 셋은 프레임 단위로 해야했다.

To this end, we assign the same videos to multiple annotators to label the temporal extent of each anomaly           
한 비디오를 많은 사람이 라벨링하게함

Training and testing sets
트레이닝 / 정상 800 비정상 810
테스트   / 정상 150 비정상 140
13개의 종류

5.	Experiments
5.1. Implementation Details

4096
512
35
1

더 깊은 애들로도 해봤는데 딱히 의미 없었다.

Evaluation Metric

5.2.	Comparison with the State-of-the-art
our method achieves much higher true positive rates than other methods under low false positive rates            
트루 파지티브가 높음 펄스 파지티브 레이트보다

indispensable: 필수적인	

Our method provides successful and timely detection of those anomalies by generating high anomaly scores for the anomalous frames.         
우리의 방법은 이상 프레임에 대해 높은 이상점수를 생성함으로써 그러한 이상징후를 성공적으로 적시에 감지할 수 있도록한다.

5.3 Analysis of the Proposed Method
The underlying assumption of the proposed approach is that given a lot of positive and negative videos with video-level labels, the network can automatically learn to predict the location of the anomaly in the video.         
제안된 방법의 가정중 하나는 비디오 레벨로 라벨링하면 모델이 알아서 이상한 비디오의 로케이션을 찾아서 예측할 수 있을 것이다였음

To achieve this goal, the network should learn to produce high scores for anomalous video segments during training iterations.        
이 목표를 달성하기 위해서 네트워크는 트레이닝 반복동안에 높은 점수를 이상한 비디오 세그멘테이션에 생성하는 것을 배워야했다.

fig 8번이 잘 보여줌
Note that although we do not use any segment level annotations, the network is able to predict the temporal location of an anomaly in terms of anomaly scores.        
우리가 세그먼트 수준 주석을 사용하지 않지만, 네트워크는 이상 점수의 관점에서 이상 징후의 일시적 위치를 예측할 수 있다.

False alarm rate
In real-world setting, a major part of a surveillance video is normal. A robust anomaly detection method should have low false alarm rates on normal videos. Therefore, we evaluate the performance of our approach and other methods on normal videos only          
현실에선 노멀 비율이 훨씬 많고
false alarm줄이는게 중요하니까 우리는 normal한 영상만 테스트했다.

5.4.	Anomalous Activity Recognition Experiments
The second baseline is the Tube Convolutional Neural Network (TCNN)
it is an end-to-end deep learning based video recognition approach         
end to end 방법임 1번 접근과 다르게

Therefore, our dataset is a unique and challenging dataset for anomalous activity recognition.        

6.	Conclusions
We propose a deep learning approach to detect realworld anomalies in surveillance videos          
우리는 딥러닝을 사용한 현실의 cctv의 이상한 점을 디텍트하는 접근법을 제안했다. 

Due to the complexity of these realistic anomalies, using only normal data alone may not be optimal for anomaly detection.               
현실 비정상의 복잡함 때문에, 정상데이터만을 사용하는 것은 비정상 탐지에 최적이 아니었다.

We attempt to exploit both normal and anomalous surveillance videos.          
우리는 정상과 비정상 이미지를 모두 사용하였고

To avoid labor-intensive temporal annotations of anomalous segments in training videos, we learn a general model of anomaly detection using deep multiple instance ranking framework with weakly labeled data             
노동 집약적인 프레임 단위의 비디오 라벨링을 피하기 위해 우리는 라벨이 약한 깊은 다중 인스턴스 순위 모델을 사용하여 일반적인 비정상 징후 감지 모델을 학습했다.

To validate the proposed approach, a new large-scale anomaly dataset consisting of a variety of real-world anomalies is introduced.        
제안된 접근법을 검증하기 위해 다양한 실제 이상 징후로 구성된 새로운 대규모 이상 징후 데이터 세트를 도입했다.

The experimental results on this dataset show that our proposed anomaly detection approach performs significantly better than baseline methods.        
이 데이터 세트에 대한 실험 결과는 이전의 비정상 신호 탐지 접근법 보다 훨씬 더 잘 수행된다는 것을 보였다.

Furthermore, we demonstrate the usefulness of our dataset for the second task of anomalous activity recognition.           
또한 비정상적인 활동 인식이라는 두번째 작업에 대한 데이터 세트의 유용성을 인증했다.


