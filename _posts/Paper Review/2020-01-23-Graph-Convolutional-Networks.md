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
  
# Graph Convolutional Networks
![image](https://user-images.githubusercontent.com/50114210/72911676-cbf10c80-3d7d-11ea-8135-883d3af809de.png)       
<center>Fig 3. An example of Graph Convolutional Networks. <U>Image taken from Thomax Kipf's blog post</U></center>
