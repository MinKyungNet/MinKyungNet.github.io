---
layout: post
title: "LeetCode 101 Symmetric Tree"
tags: [BFS]
categoires: [Algorithm]
---

# 내가 푼 코드
만약 트리가 존재하지 않았다면 대칭이라고 봐도 무방함으로 True을 리턴     
트리는 존재하지만 val만 존재하고 left, right가 둘다 없으면 True을 리턴    
나머지 경우는 is_decal에서 처리한다.    
왼쪽 트리와 오른쪽트리가 둘다 존재하지 않는다면 대칭이니까 True    
그렇지 않고 둘 중에 하나만 없으면 False     
왼쪽과 오른쪽의 값이 같고, is_decal이 맨 아래가 전부 True가 나오면 True 아니라면 False를 리턴한다.

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def is_decal(p, q):
            if not p and not q: return True
            if not (p and q): return False
            if p.val == q.val and is_decal(p.left, q.right) and is_decal(p.right, q.left): return True
            return False
        
        if not root: return True
        if not root.left and not root.right: return True
        return is_decal(root.left, root.right)
```

# 다른 사람 코드

트리가 존재하지 않는다면 True를 리턴   
아니라면 왼쪽 노드와 오른쪽 노드를 넣어서 대칭인지 확인한다.    
아래는 내가 짠 코드와 비슷하게    
왼쪽과 오른쪽 둘다 존재하지 않는다면 True    
하나만 존재하지 않는다면 false    
값이 같고 아래단의 Symmetric이 True를 리턴한다면 True   
값이 같지 않다면 False를 리턴한다.

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True 
        else:
            return self.Symmetric(root.left, root.right)
                
    def Symmetric(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val == right.val:
            return self.Symmetric(left.left, right.right) and self.Symmetric(left.right, right.left)
        else:
            return False
```

![image](https://user-images.githubusercontent.com/50114210/65399789-aa950980-ddf9-11e9-815a-10f3135f18ea.png)
