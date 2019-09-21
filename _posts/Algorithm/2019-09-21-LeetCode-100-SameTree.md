---
layout: post
title: "LeetCode 100 SameTree"
tags: [dfs, binary tree]
categories: [Algorithm]
---

# 내가 푼 코드
문제 입력을 보면서 이게 무슨 입력이야? 하고 당황했지만,   
저렇게 생긴 트리가 들어온다고 말하는 거였다.    

dfs를 시작하면 트리에 있는 값을 받고    
전위순위로 왼쪽노드를 우선탐색, 그 후에 오른쪽 노드를 탐색한다.    
왼쪽이나 오른쪽에 값이 있는 것을 확인할 때마다    
재귀하여 문제를 풀이했다.   
마지막에는 탐색한 리스트를 반환한다.   
그리고 두 트리가 같은지 최종적으로 비교하며 프로그램 종료

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        plist = []
        qlist = []

        def dfs(tree, search):
            if tree:
                search.append(tree.val)
                if tree.left:
                    dfs(tree.left, val)
                else:
                    search.append(None)

                if tree.right:
                    dfs(tree.right, val)
                else:
                    search.append(None)
            return search

        plist = dfs(p, plist)
        qlist = dfs(q, qlist)

        return plist == qlist
```

# 다른 사람이 푼 코드
p와 q 둘다 없다 = True       
p와 q 둘중에 하나만 없다 = False       
p.val과 q.val, 왼쪽, 오른쪽 모양이 같다 = True      
맨 처음 호출된 함수가 세번째 if를 통과하지 못한다면 = False

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q: return True
        if not (p and q): return False
        if p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right,q.right): return True
        return False
```

![image](https://user-images.githubusercontent.com/50114210/65373471-ebd1d000-dcb8-11e9-970a-e043215560e3.png)
