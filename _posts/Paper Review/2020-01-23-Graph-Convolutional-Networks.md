---
layout: post
title: "Graph Convolutional Networks"
tags: [GCNN]
categories: [Paper Review]
---
![image](https://user-images.githubusercontent.com/50114210/72908465-d8269b00-3d78-11ea-8c37-afad920b4083.png)
Graph Neural Networks(GNN)는 drug discovery 인공지능의 수준을 올리는데 많은 기여를 하고 있습니다. 그 동안, 약을 인공지능이 이해할 수 있게 feature로 표현하는 방법 중에 [Extended-Connectivity Fingerprints(ECFPs)](https://pubs.acs.org/doi/abs/10.1021/ci100050t)가 많이 이용되어 왔습니다. 하지만 [Steven Kearnes et al., 2016](https://arxiv.org/pdf/1603.00856.pdf)에 의하면, 약을 그래프로 표현하는 것이 ECFPs보다 반드시는 아니지만 대체로 좋은 성능을 보이고 있습니다.

<center><img src=https://user-images.githubusercontent.com/50114210/72908527-ef658880-3d78-11ea-9c0a-24c5fed482f0.png></center>
