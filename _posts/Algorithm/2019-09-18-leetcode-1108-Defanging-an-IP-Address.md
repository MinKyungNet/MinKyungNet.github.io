---
layout: post
title: "LeetCode 1108 Defanging an IP Address"
tags: "String"
categoires: [Algorithm]
---

전에 추천받은 레코드를 처음 써봤다.     
백준이나 프로그래머스보다는 훨씬 좋은 것같다는 생각이 든다.    
얼마나 빨리 풀었는지도 나오고 정답도 바로바로 보여줘서 공부하기 편할 것 같다!    

```python
class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")
```

replace라는 함수를 써서 .을 [.]으로 치환해줬다.    
처음에는 아예 다른 변수를 사용하기도 하고    
같은 변수에 받아서 리턴을 다른줄에 하기도 했는데    
생각해보면 위의 코드가 가장 빠를것 같아서 이렇게 제출했다   

![image](https://user-images.githubusercontent.com/50114210/65088422-41c22180-d9f4-11e9-97eb-cff50a5325d3.png)
