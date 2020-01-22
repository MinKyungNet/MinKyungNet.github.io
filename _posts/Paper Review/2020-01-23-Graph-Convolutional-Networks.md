---
layout: post
title: "Graph Convolutional Networks"
tags: [GCNN]
categories: [Paper Review]
---
![image](https://user-images.githubusercontent.com/50114210/72908465-d8269b00-3d78-11ea-8c37-afad920b4083.png)
Graph Neural Networks(GNN)는 drug discovery 인공지능의 수준을 올리는데 많은 기여를 하고 있습니다. 그 동안, 약을 인공지능이 이해할 수 있게 feature로 표현하는 방법 중에 [Extended-Connectivity Fingerprints(ECFPs)](https://pubs.acs.org/doi/abs/10.1021/ci100050t)가 많이 이용되어 왔습니다. 하지만 [Steven Kearnes et al., 2016](https://arxiv.org/pdf/1603.00856.pdf)에 의하면, 약을 그래프로 표현하는 것이 ECFPs보다 반드시는 아니지만 대체로 좋은 성능을 보이고 있습니다.       
     
![image](https://user-images.githubusercontent.com/50114210/72908527-ef658880-3d78-11ea-9c0a-24c5fed482f0.png)           
   
<center>Fig 1. CVPR 2019 Top 25 keywords</center>
    
위 사진에서 볼 수 있듯이 "graph"는 CVPR 2019 상위 키워드 15위로 등극하였습니다(CVPR 2018은 55위). 그 만큼 GNN에 많은 연구가 이루어지고 있고, 이를 drug discovery 분야에도 적용한다면 많은 발전이 있을 것이라 예상합니다. 그 중, GNN을 대표하는 모델 중 하나인 **Graph Convolutional Networks(GCN)**에 대해 알아보도록 하겠습니다.       
      
<br>
# What Is Graph?
![image](https://user-images.githubusercontent.com/50114210/72910610-1f625b00-3d7c-11ea-809a-3b377deed8c0.png)            
    
<center>Fig 2. Graph(left), Adjacency matrix(right)</center>
    
GNN에서 말하는 그래프에 대해 간략하게 설명하겠습니다. 그래프는 우선 다음 두가지로 이루어져 있습니다.      

1. Node(Vertex) : Fig 2. 왼쪽 그림에서 원으로 표시된 a, b, c, d, e, f를 node라 합니다.          
     
2. Edge : 두 vertices를 연결한 선을 의미합니다.          
         
약에서 nodes는 원소들을, edges는 결합 방법(single, double, triple, aromatic 등)을 의미합니다.         
     
또한, 그래프는 Fig 2. 우측 그림과 같은 인접 행렬(Adjacency matrix)를 이용한다면 비교적 컴퓨터가 이해하기 쉽게 그래프를 표현할 수 있습니다. 각 column과 row에 순서대로 node set을 정의하고, edge로 연결이 되어 있으면 1, 그렇지 않으면 0으로 채워주어 간단하게 인접 행렬을 구할 수 있습니다. 보통, 인접 행렬은 자기 자신으로 가는 edge가 없기 때문에 대각 원소(diagonal elements)를 0으로 채웁니다. **하지만 GCN에서는 자기 자신의 정보를 이용하기 위하여 1로 채워 줍니다.**         
    
이 외에도 그래프는 다음과 같은 상황에서 응용될 수 있습니다.       
       

1. SNS에서 관계 네트워크      
       
2. 학술 연구에서 인용 네트워크     
       
3. 3D Mesh   

    
<br>
# Graph Convolutional Networks
![image](https://user-images.githubusercontent.com/50114210/72911676-cbf10c80-3d7d-11ea-8135-883d3af809de.png)        
<center>Fig 3. An example of Graph Convolutional Networks. <U>Image taken from Thomax Kipf's blog post</U></center>
   
Convolutional Neural Networks(CNN)에서 픽셀 대상으로 하던 합성곱(convolution) 연산을 Graph Convolutional Networks(GCN)에서는 그래프에 적용하자는 것이 가장 기본적인 아이디어입니다.      
   
<br>
## Input
![image](https://user-images.githubusercontent.com/50114210/72912164-94369480-3d7e-11ea-8dd1-e2c93f51644a.png)     
<center>Fig 4. Input matrices of Graph Convolutional Networks</center>
   
GCN에서는 다음의 두 행렬을 입력으로 받습니다.      
   
* A : 그래프의 인접 행렬      
      
* X : N × D feature matrix (N = nodes의 수, D = vertex feature의 차원)    
    
예를 들어, 그래프 구조가 SNS에서 친구들의 관계를 나타내는 네트워크라면 node는 사람이 될 것이고, edge는 사람들 간의 friendship의 정도가 될 것입니다. 이 때, 특징 행렬 X는 각 node의 feature(나이, 신장, 몸무게, 결혼 유무, 흡연 유무 등)로 만들어진 행렬을 의미합니다.     
      
<br> 
## Output
GCN은 node-level output 혹은 graph-level output이 모두 가능합니다. 이는 우리가 해결해야할 task가 어떤 형태인지에 따라 달라지게 됩니다. 예를 들어, SNS관계 네트워크에서 사람 단위로 분류하고 싶은 경우에는 node-level output이, 약을 분류하고 싶은 경우에는 graph-level output이 적절할 것입니다.      
   
* Node-level output Z : N x F feature matrix(N = nodes의 수, F = node feature의 차원)      
     
* Graph-level output은 <U>pooling 연산</U>을 이용     
  
<br>   
## How to update node feature
![image](https://user-images.githubusercontent.com/50114210/72912885-a36a1200-3d7f-11ea-987c-a405177f754c.png)      
<center>Fig 5. Information needed to update feature of node b(left), node a(right)</center>
    
Node feature를 업데이트 하기 위하여 자기 자신의 정보와 인접한 노드들의 정보를 함께 이용합니다. 예를 들어, 노드 b를 업데이트 하기 위해서 노트 a, b, c, d의 정보를 이용하고(Fig 5.의 좌측 그림), 노드 a를 업데이트 하기 위해서는 노드 a, b의 정보만을 이용하면 됩니다.(Fig 5.의 우측 그림).       
     
이를 수식으로 다음과 같이 나타낼 수 있습니다.      
<br>   
![image](https://user-images.githubusercontent.com/50114210/72913129-0491e580-3d80-11ea-8396-991da95b8ead.png)
<br>    
where
<br>     
![image](https://user-images.githubusercontent.com/50114210/72913193-212e1d80-3d80-11ea-89c3-35bb3d469fbc.png)     
<br>   
다시 말해, l+1 레이어에서 node i의 feature를 업데이트 하는 방법은 nodes(node i와 인접한 노드들)의 weight를 곱해주고 bias를 더한 형태에 활성화 함수를 입힌 형태입니다.      
       
이를 모든 노드에 대하여 다음과 같이 하나의 행렬식으로 표현할 수 있습니다.     
      
![image](https://user-images.githubusercontent.com/50114210/72913277-4589fa00-3d80-11ea-806a-9bb7c24160ea.png)   
      
여기서 주의할 점은 인접 행렬 A의 대각원소를 모두 1로 하여야 자기 자신의 정보를 이용한다는 것을 쉽게 보일 수 있습니다.        
        
하지만, 위의 식을 바로 이용하게 되면 A를 정규화(normalization)하지 않기 때문에 연산 과정에서 feature vector의 scale이 완전히 바뀐다는 문제가 생기게 됩니다. 따라서 우리는 인접행렬 A를 다음과 같이 정규화하여 사용합니다.      
<br>    
![image](https://user-images.githubusercontent.com/50114210/72913407-723e1180-3d80-11ea-9681-e0144df3f860.png)
<br>      
where D is the diagonal node degree matrix,     
<br>
![image](https://user-images.githubusercontent.com/50114210/72913466-8550e180-3d80-11ea-8005-99c1cefb4bb0.png)         
D 행렬은 자신을 포함하여 몇 개의 노드와 연결이 되어있는지를 나타내는 행렬이고, 인접 행렬 A의 각 row의 원소들을 더하여 쉽게 얻을 수 있습니다. 이렇게 얻은 D 행렬의 역함수를 구하고 루트를 씌어주어 인접 행렬 A의 앞 뒤에 곱해주면 우리는 정규화된 인접 행렬을 구할 수 있습니다.   
       
실제 사용에서는 graph convolution layer를 세번 정도 거쳐 각 노드의 feature를 업데이트하고 해당하는 task에 따라 classification 혹은 regression을 진행하면 됩니다.   
     
<br>
# Conclusion
<U>Graph Neural Networks는 강력합니다.</U> 더군다나, 그래프 구조로 표현 되는 drug discovery 분야에서는 더욱 강력합니다. 그 중 GNN을 대표하는 Graph Convolutional Networks에 대해 알아봤습니다. 이를 시작으로 SOTA graph model을 공부하여 drug discovery에 적용한다면 빠른 시일내에 인공지능으로 만든 약을 시중에서 볼 수 있을 것이라 예상합니다.
     
<br>
# References    
[1] [<U>Thomax Kipf's blog post</U>](https://tkipf.github.io/graph-convolutional-networks/)   
         
[2] [<U>SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS</U>](https://arxiv.org/pdf/1609.02907.pdf) Thomas N. Kipf and Max Welling, ICLR 2017        
         
[3] [<U>Slide by DonghyeonKim</U>](https://www.slideshare.net/DonghyeonKim7/graph-convolutional-network-gcn)      
      
