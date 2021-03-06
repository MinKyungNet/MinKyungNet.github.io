---
layout: post
title: "Network 속의 Network"
tags: [convolutional neural network, filter]
categories: [Convolutional Neural Networks]
---

# 학습 목표
- 1 * 1 합성곱의 역할을 알아본다.

# 핵심 키워드
- 합성곱 신경망
- 필터

# 학습 내용
- 합성곱 신경망을 구축할 때 1 * 1 합성곱은 매우 유용합니다.
![image](https://user-images.githubusercontent.com/50114210/71483017-0764ed80-2849-11ea-8be1-e91e9b795caf.png)     
- 위의 예시 처럼, 192개의 입력숫자가 32개의 1 * 1 필터와 합성곱을 하여 32개의 출력 숫자가 됩니다. 즉, 이는 입력 채널의 수 만큼 유닛을 입력으로 받아서, 이들을 하나로 묶는 연산과정을 통해, 출력채널의 수만큼 출력을 하는 작은 신경망 네트워크로 간주할 수 있습니다. 따라서 네트워크 안의 네트워크라고도 합니다.
- 이처럼 1 * 1 합성곱 연산을 통해 비선형성을 하나 더 추가해 복잡한 함수를 학습 시킬 수도 있고, 채널수를 조절해 줄 수도 있습니다.

# 인트로
합성곱 신경망을 구축할 때 1 x 1 합성곱은 매우 유용합니다
어쩌면 1 x 1 합성곱이 무슨 일을 할 지 질문을 가질 수 있을텐데
그냥 숫자 하나를 곱하는 것이라 큰 의미없어 보이지만 실제로는 그렇지 않습니다
한 번 살펴봅시다

# 1 * 1 필터
![image](https://user-images.githubusercontent.com/50114210/71483200-10a28a00-284a-11ea-8d3d-a3b5c446b5db.png)                     
여기 숫자 2 의 1 x 1 필터가 있습니다            
여기 6 x 6 이미지를 1 x 1 필터와 합성곱 연산을 하게 되면 이미지에 2 만큼 곱해주는 셈입니다            
그래서 1 2 3 이 2 4 6 이 되는 식이죠              
이렇듯 1 x 1 필터와의 합성곱은 그다지 유용해보이지 않습니다 숫자 하나 곱하는 것이기 때문이죠            
그러나 이것은 6 x 6 의 크기에 1 개의 채널만 있을 때이고            
6 x 6 x 32 라고 한다면 1 x 1 필터와의 합성곱을 하는 것이 훨씬 의미 있게 됩니다            
1 x 1 합성곱이 하는 일은 36 개의 위치를 각각 살펴보고             
그 안의 32 개의 숫자를 필터의 32 개의 숫자와 곱해줍니다            
그리고 ReLU 비선형성을 적용해준 뒤에 36 개 중 하나의 위치에 이렇게 한 조각이 생기고            
여기 32 개의 숫자를 이 한 조각의 숫자와 곱해주면             
하나의 숫자만 남게 됩니다 이렇게 한 지점에 출력되겠죠            
이 1 x 1 x 32 필터 안에 있는 32 개의 숫자에 대해서는 마치 하나의 뉴런이 32 개의 숫자를 입력받고            
32 개의 숫자를 각각 같은 높이와 너비에 해당하는 채널 각각에 곱해주고            
ReLU 비선형성을 적용해주면 해당하는 값이 여기 출력될 것입니다            
그리고 만약 하나가 아닌 다수의 필터가 있다면            
여러 개의 유닛을 입력으로 받아서 한 조각으로 묶는 셈이 되고             
출력은 6 x 6 가 필터의 수만큼 있게 됩니다            
그래서 1 x 1 필터에 대해 이해하는 한 가지 방법은             
완전 연결 신경망을 36 개의 위치에 각각 적용해서             
32 개의 숫자를 입력값으로 받고 필터의 수 만큼 출력하는 것이죠            
그래서 이 부분은 사실 nC[l + 1] 입니다            
이것을 36 개의 위치에 각각 실행하게 되면 6 x 6 x (필터의 수) 를 결과로 얻고            
입력값에 대한 자명하지 않은 계산을 해야합니다            
종종 1 x 1 합성곱이라고 불리곤 하는데 네트워크 안의 네트워크 라고도 합니다            
이 논문에서 그렇게 표현하고 있죠 비록 이 논문에서 사용한 구조는 자주 사용되지는 않지만            
1 x 1 합성곱이라는 개념 또는 네트워크 안의 네트워크라는 개념은             
다른 신경망 구조에 많은 영감을 주었습니다            

# 채널의 수를 조절
![image](https://user-images.githubusercontent.com/50114210/71483217-231cc380-284a-11ea-974d-adc27b8df4f2.png)                 
1 x 1 합성곱이 유용한 경우를 실제 경우를 한 번 살펴보면           
28 x 28 x 192 의 입력이 있다고 해봅시다 만약 높이와 너비를 줄이려면 풀링 층을 사용하면 됩니다          
그럼 만약 채널의 수가 너무 많아서 줄이려면 어떻게 해야 할까요          
어떻게 28 x 28 x 32 로 줄일 수 있을까요          
바로 32 개의 1 x 1 필터를 사용하면 됩니다          
실제로는 각 필터가 1 x 1 x 192 의 크기를 가지겠죠          
필터와 입력의 채널 수가 일치해야 하기 때문입니다          
32 개의 필터를 사용하면 출력은 28 x 28 x 32 의 크기를 가질 것입니다          
또한 이것은 nC 를 줄이는 방법이기도 합니다          
풀링 층은 높이와 너비인 nH 와 nW 만 줄일 수 있는 반면에 말이죠          
이후에 어떻게 1 x 1 합성곱이 채널 수를 줄여서           
네트워크의 계산을 용이하게 하는지 살펴볼 것입니다          

# 네트워크의 비선형성을 더해줌
물론 192 개의 채널 수를 유지해도 괜찮습니다          
1 x 1 합성곱의 효과는 비선형성을 더해주고           
하나의 층을 더해줌으로써 더 복잡한 함수를 학습할 수 있습니다          
28 x 28 x 192 를 입력 받아 28 x 28 x 192 를 출력하는 층 말이죠          
이것이 1 x 1 합성곱 층이 하게 되는 중요한 역할입니다          
네트워크에 비선형성을 더해주고 채널의 수를 조절할 수 있게 되죠          
이것이 인셉션 신경망 구축에 매우 유용하게 사용되는데 다음 영상에서 살펴볼 것입니다          
이렇게 1 x 1 합성곱이 꽤나 중요한 역할을 하는 것을 보았습니다          
채널의 수를 원하는대로 줄이거나 늘릴 수 있게 만들어주죠          
