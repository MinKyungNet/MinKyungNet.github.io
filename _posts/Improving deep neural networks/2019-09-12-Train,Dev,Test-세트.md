---
layout: post
title: "Train / Deb / Test 세트"
tags: [train, dev, test]
categories: [Improving deep neural networks]
---

# 학습 목표
훈련, 개발, 테스트 세트를 설정할 수 있다.

# 핵심 키워드
훈련 세트(train set)
개발 세트(dev set)
테스트 세트(test set)

# 학습 내용
* 신경망이 몇 개의 층을 가지는지, 각 층의 몇 개의 은닉 유닛을 가지는지, 학습률과 활성화 함수는 무엇인지 등을 결정해 신경망을 훈련시켜야합니다.
* 좋은 하이퍼 파라미터 값을 찾기 위해 사이클을 여러번 반복하며 최적이 값을 선택합니다.
* 이때 훈련, 개발, 테스트 세트를 잘 설정해 과정을 효율적으로 만들 수 있습니다.
* 훈련 세트 : 훈련을 위해 사용되는 데이터
* 개발 세트 : 다양한 모델 줄 어떤 모델이 좋은 성능을 나타내는지 확인
* 테스트 세트 : 모델이 얼마나 잘 작동하는지 확인

# 인트로
딥러닝의 실제적인 면을 다뤄보는 이번 강의에 오신 것을 환영합니다.    
아마 여러분은 신경망을 어떻게 구현하는지 배웠을 것입니다.    
이번 주에는 여러분의 신경망이 잘 작동하기 위한 실질적인 측면을 배워보겠습니다.    
데이터를 설정하기 위한 하이퍼파라미터 튜닝부터      
최적화 알고리즘의 속도를 높여 적당한 시간 안에 학습 알고리즘이 학습할 수 있도록 하는 방법까지 다뤄보도록 하겠습니다.
첫 번째 주에는 머신러닝 문제를 어떻게 해결하는지에 관해 이야기하겠습니다.    
정규화에 대해 다루고 신경망 구현이 맞게 되었는지 확인하는 몇 가지 기술을 살펴보도록 하겠습니다.

# 인공지능을 개발하는 것은 굉장히 반복적인 작업이다.
훈련, 개발, 테스트 세트를 어떻게 설정할지에 관한 좋은 선택을 내리는 것은    
좋은 성능을 내는 네트워크를 빠르게 찾는데 큰 영향을 줍니다.    

### 다양한 하이퍼파라미터
![image](https://user-images.githubusercontent.com/50114210/64781155-2c389c00-d59d-11e9-8643-ec4bdad3a432.png)      
신경망을 훈련시킬 때는 많은 결정을 내려야합니다.    
신경망이 몇개의 층을 가지는지 각각의 층이 몇개의 은닉유닛을 가지는지,    
학습률은 무엇인지 서로 다른 층에 사용하는 활성화 함수는 무엇인지    
새로운 애플리케이션을 시작할 때는 이 모든 것에 대한 올바른 값을 추측하는 것이 거의 불가능합니다.     
다른 하이퍼파라미터에 대해서도 마찬가지입니다.    
따라서 실질적으로 머신러닝을 적용하는 것은 매우 반복적인 과정입니다.

### 반복적인 작업
![image](https://user-images.githubusercontent.com/50114210/64781268-6efa7400-d59d-11e9-9be6-8176bf9952e5.png)    
주로 처음에는 아이디어로 시작합니다.    
특정 개수의 층과 유닛을 가지고 특정 데이터 세트에 맞는 신경망을 만듭니다.    
그럼 이것을 코드로 작성하고 실행하고 실험을 진행합니다.    
그 결과 특정 네트워크 혹은 설정이 얼마나 잘 작동하는지를 알게 됩니다.    
결과에 기반해 아이디어를 개선하고 몇가지 선택을 바꾸게 됩니다.    
그리고 더 나은 신경망을 찾기 위해 이 과정을 반복하게 됩니다.    

### 딥러닝이 활용되는 다양한 분야
![image](https://user-images.githubusercontent.com/50114210/64781404-b97bf080-d59d-11e9-8596-4050772ddd20.png)     
오늘날 딥러닝은 다양한 분야에서 많은 성공을 거두었습니다.    
자연어 처리 과정부터 컴퓨터 비전, 음성 인식, 그리고 구조화된 데이터데 적용된 다양한 애플리케이션    
구조화된 데이터에 포함되는 것은 광고, 웹 검색, 컴퓨터 보안, 물건 배송에도 적용됩니다.    
예를 들면 물건을 가져오고 가져다 놓기 위해 운전사를 어디로 보내는지 알아내는 것에 딥러닝을 적용할 수 있습니다.    
가끔 NLP에 많은 겸험을 가진 연구원이 컴퓨터 비전에서 무언가를 시도하거나,    
음성 인식에 많은 경험을 가진 연구원이 광고에서 무언가를 시도하거나,    
보안 분야의 경험자가 배송 분야로 옮기고 싶어하는 경우를 보았습니다.    
제가 이 과정에서 느낀 것은 어떤 분야나 애플리케이션의 직관이 다른 애플리케이션 영역에 거의 적용되지 않는다는 것입니다.    
그리고 최고의 선택은 가지고 있는 데이터의 양 입력 특성의 개수,    
GPU나 CPU처럼 훈련을 진행하는 컴퓨터 설정, 정확히 어떤 설정을 했는지 등 다양한 요인에 의해 결정됩니다.    
따라서 많은 애플리케이션에서 딥러닝에 아주 경험이 많은 사람일지라도 첫 시도에 하이퍼파라미터에 대한    
최고의 선택을 올바르게 추측하는 것은 거의 불가능하다고 생각합니다.    
따라서 오늘날 딥러닝을 적용하는 것은 애플리케이션의 네트워크에 대한 좋은 선택을 찾기 위해 이 사이클을 여러번 돌아야하는 매우 반복적인 과정입니다.    

# Train, Dev, Test Sets
![image](https://user-images.githubusercontent.com/50114210/64781624-4757db80-d59e-11e9-9000-ea30753be4a6.png)    
따라서 빠른 진전을 이루는데 영향을 미치는 것들은 이 사이클을 얼마나 효율적으로 돌 수 있는지와 데이터 세트를 잘 설정하는 것입니다.
훈련, 개발, 테스트 세트를 잘 설정하는 것은 과정을 더 효율적으로 만듭니다.     
여기 훈련 데이터를 큰 박스로 나타내봅시다.    
전통적인 방법은 모든 데이터를 가져와서 일부를 잘라서 훈련 세트로 만들고      
다른 일부는 교차 검증 세트(가끔 개발 세트라고 부르기도 합니다.) 간단하게 말하면 dev세트이죠.      
마지막 부분은 테스트 세트로 만들게 됩니다.    
작업의 흐름은 훈련 세트에 대해 게속 훈련 알고리즘을 적용시키면서     
개발 세트 혹은 교차 검증 세트에 대해 다양한 모델 중 어떤 모델이 가장 좋은 성능을 내는지 확입합니다.    
이 과정을 충분히 거치고 더 발전시키고 싶은 최종 모델이 나오면      
테스트 세트에 그 모델을 적용시켜 알고리즘이 얼마나 잘 작동하는지 편향 없이 측정하게 됩니다.    

### 머신러닝 이전의 데이터 나누기
#### 명시적인 개발 세트가 없을 때
![image](https://user-images.githubusercontent.com/50114210/64781958-1035fa00-d59f-11e9-8b9a-dcfa61c082ba.png)      
머신러닝 이전 시대에서는 모든 데이터를 가져와서 70 / 30 퍼센트      
70은 훈련 세트, 30은 테스트 세트로 나누는 것이 일반적인 관행이었습니다.    
명시적인 개발 세트가 없을 경우에 말이죠.    

#### 명시적인 개발 세트가 있을 때
![image](https://user-images.githubusercontent.com/50114210/64782029-3bb8e480-d59f-11e9-8344-ee58f91a30f8.png)      
혹은 60은 테스트 세트, 20은 개발 세트, 20은 테스트 세트로 나눴습니다.    

### 현대 빅데이터 시대의 데이터 나누기
![image](https://user-images.githubusercontent.com/50114210/64782125-876b8e00-d59f-11e9-8fe1-a8bf24b2d483.png)       
몇 년 전까지만 해도 이것은 머신러닝에서 최적의 관행으로 여겨졌습니다.    
총 100개, 1,000개, 10,000개의 샘플의 경우에 이 비율은 경험에서 나온 가장 합당한 비율이었습니다.    
그러나 총 100만개 이상의 샘플이 있는 현대 빅데이터 시대에는 개발 세트와 테스트 세트가 훨씬 더 작은 비율이 되는 것이 트렌드가 되었습니다.     
왜냐하면 개발 세트와 테스트 세트의 목표는 서로 다른 알고리즘을 시험하고 어떤 알고리즘이 더 잘 작동하는지 확인하는 것이기 때문에     
개발 세트는 평가할 수 있을 정도로만 크면 됩니다.     
2개, 10개의 알고리즘 선택 중 어느 것이 더 나은지 빠르게 선택할 수 있도록 말입니다.    
이를 위해 전체 데이터의 20퍼센트나 필요하지는 않습니다.     
따라서 백만개의 훈련 샘플이 있는 경우에는 만 개의 샘플을 개발 세트로 설정하면      
두 개의 알고리즘 중 어느 것이 더 좋은지 평가하는데 충분합니다.     
같은 방식으로 테스트 세트의 주요 목표는 최종 분류기가 어느 정도 성능인지 신뢰있는 추정치를 제공하는 것이므로    
예를 들어 백만 개의 샘플이 있으면 만 개의 샘플을 설정하도 충분합니다.     
단일 분류기의 성능을 평가하고 안정적인 추정치를 제공하기 위해서 말입니다.    
따라서 백만 개의 샘플을 가지고 있는 이런 경우에는 개발 세트로 만 개, 테스트 세트로 만 개의 샘플을 설정하면 충분합니다.       
다시 말하자면, 머신 러닝 문제를 설정할 때는 훈련, 개발, 테스트 세트를 설정하게 되는데    
상대적으로 적은 데이터 세트일 경우에는 전통적인 비율로 설정하는 것도 괜찮습니다.    
그러나 훨씬 큰 데이터 세트라면 개발과 테스트 세트를 전체 데이터의 20% 혹은 10%보다 더 작게 설정하는 것도 괜찮은 방법입니다.     

# train set과 test셋 분포의 불일치      
현대 딥러닝의 또 다른 트렌트는 더 많은 사람들이 일치하지 않는 훈련 / 테스트 분포에서 훈련시킨다는 것입니다.    
사용자가 모든 사진을 업로드하는 앱을 만든다고 가정해봅시다.     

### 사용자의 고양이 사진을 찾아보자
여러분의 목표는 사용자에게 그 중 고양이 사진을 찾아 보여주는 것입니다.     
아마 여러분의 훈련 세트는 인터넷에서 다운 받은 고양이 사진에서 온 것일 겁니다.    
그러나 개발과 테스트 세트는 앱을 사용하는 사용자들에 의해 구성된 것일 겁니다.    
따라서 훈련 세트는 인터넷에서 긁어온 많은 사진이고, 개발과 테스트 세트는 사용자에 의해 업로드된 사진들입니다.    
많은 웹퍼이지는 아주 전문가스럽고 잘 정돈된 고양이 사진 결과를 가지고 있으나,    
사용자들은 흐릿한 저해상도의 사진을 올리게 될 것입니다.     
일상적인 상황에서 핸드폰 카메라로 찍은 사진들이겠죠.      

#### 데이터 셋의 분포
![image](https://user-images.githubusercontent.com/50114210/64782774-090feb80-d5a1-11e9-9095-8e7cc7fc2ba4.png)     
따라서 이 두 가지 데이터의 분포는 달라질 수 있습니다.    
경험상 이런 경우에는 개발과 테스트 세트가 같은 분포에서 와야합니다.     
개발 세트를 사용해 다양한 모델을 평가하고 성능을 개선하기 위해 열심히 노력할 것이므로     
개발 세트가 테스트 세트와 같은 분포에서 오는 것이 더 좋습니다.    

# 테스트 셋은 없어도 된다.
마지막으로 테스트 세트를 갖지 않아도 괜찮습니다.    
테스트 세트의 목표는 최종 네트워크의 성능에 대한 비편향 추정을 제공하는 것입니다.    
여러분이 선택한 네트워크에 대해서요.    
그러나 비편향 추정이 필요 없는 경우에 테스트 세트를 갖지 않아도 괜찮습니다.     
따라서 개발 세트만 있는 경우에 모든 테스트 세트를 훈련 세트에서 훈련시키고     
다른 모델 아키텍트를 시도하고 이것을 개발 세트에서 평가합니다.     
그리고 이 과정을 반복해 좋은 모델을 찾습니다.      
개발 세트에 데이터를 맞추기 떄문에 성능에 대한 비편향 추정을 주지 않습니다.      
그러나 그 추정이 필요하지 않다면 테스트 세트가 없어도 괜찮습니다.    
머신러닝에서 별도의 테스트 세트 없이 훈련 세트와 개발 세트만 있는 경우 대부분의 사람들은 개발 세트를 테스트 세트라고 부릅니다.     
그러나 실제로 하는 것은 테스트 세트를 교차 검증 세트로 사용하는 것입니다.     
완벽히 좋은 용어는 아닙니다. 테스트 세트에 과적합하기 때문입니다.    
따라서 팀에서 훈련 세트와 테스트 세트만 있다고 말하는 경우    
그들이 훈련과 개발 세트를 진짜 갖고 있는지 주의 깊게 볼 것입니다.     
왜냐면 테스트 세트에 과적합이 일어나기 때문입니다.    
알고리즘의 성능에 완전한 비편향 추정이 필요 없다면 이런 방식도 괜찮습니다.    
훈련, 개발, 테스트 세트를 설정하는 것은 반복을 더 빠르게 할 수 있도록 합니다.     
또한 알고리즘의 편황과 분산을 더 효율적으로 측정할 수 있도록합니다.




