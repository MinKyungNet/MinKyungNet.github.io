---
layout: post
title: "Graph Convolutional Networks"
tags: [GCNN]
categories: [Paper Review]
---
![image](https://user-images.githubusercontent.com/50114210/72908465-d8269b00-3d78-11ea-8c37-afad920b4083.png)
Graph Neural Networks(GNN)는 drug discovery 인공지능의 수준을 올리는데 많은 기여를 하고 있습니다. 그 동안, 약을 인공지능이 이해할 수 있게 feature로 표현하는 방법 중에 [Extended-Connectivity Fingerprints(ECFPs)](https://pubs.acs.org/doi/abs/10.1021/ci100050t)가 많이 이용되어 왔습니다. 하지만 [Steven Kearnes et al., 2016](https://arxiv.org/pdf/1603.00856.pdf)에 의하면, 약을 그래프로 표현하는 것이 ECFPs보다 반드시는 아니지만 대체로 좋은 성능을 보이고 있습니다.
<center>
![image](https://user-images.githubusercontent.com/50114210/72908527-ef658880-3d78-11ea-9c0a-24c5fed482f0.png)         

Fig 1. CVPR 2019 Top 25 keywords</center>

위 사진에서 볼 수 있듯이 "graph"는 CVPR 2019 상위 키워드 15위로 등극하였습니다(CVPR 2018은 55위). 그 만큼 GNN에 많은 연구가 이루어지고 있고, 이를 drug discovery 분야에도 적용한다면 많은 발전이 있을 것이라 예상합니다. 그 중, GNN을 대표하는 모델 중 하나인 Graph Convolutional Networks(GCN)에 대해 알아보도록 하겠습니다.

# What Is Graph?
![image](https://user-images.githubusercontent.com/50114210/72910610-1f625b00-3d7c-11ea-809a-3b377deed8c0.png)          


