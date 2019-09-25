---
layout: post
title: "Stored Procedure, Trigger, Modeling"
tags: [Stored Procedure, Trigger, Modeling]
categories: [Web DB Programing
---

# 테이블을 만드는 방법
1. 코드로 만들기
```MYSQL
CREATE TABLE deletedMemberTBL(
	memberID CHAR(8), 
	memberName CHAR(5), 
	memberAddress CHAR(20),
    deletedData DATE);    
```
2. Schemas에서 우클릭으로 만들기
3. 모델링 화면에서 만들기 

# Stored Procedure
![image](https://user-images.githubusercontent.com/50114210/65594055-e3d1a300-dfcc-11e9-88a7-f63b710272ac.png)       
```MYSQL
DELIMITER //
CREATE PROCEDURE myProc()
BEGIN
	SELECT * FROM membertbl;
	SELECT * FROM membertbl WHERE memberName = '당탕이';
	SELECT * FROM producttbl;
	SELECT * FROM producttbl WHERE productName = '냉장고';
END //
DELIMITER ;
CALL myProc();
```
-코딩해둔 절차를 따로 지정해둔다.      
-한번에 뭉텅이로 호출할 수 있게 한다.  
-자주 쓰는데 귀찮아서 프로시저로 지정한다.

# Trigger
![image](https://user-images.githubusercontent.com/50114210/65594073-f8ae3680-dfcc-11e9-810e-eb95aeb8992f.png)        
```MYSQL
DELIMITER //
CREATE TRIGGER trg_deletedMemberTBL
AFTER DELETE
ON memberTBL
FOR EACH ROW
BEGIN
	INSERT INTO deletedMemberTBL VALUES(OLD.memberID, OLD.memberName, OLD.memberAddress, CURDATE());
END //
DELIMITER ;
```

-특정 조건이 되면 작동한다. 위의 코드는 memberTBL의 요소가 삭제되면 trg_deleteMemberTBL에 요소의 정보와 삭제된 시간가지 입력한다.         
-릴레이티드 데이터베이스의 특징을 잘 드러내고 있다.


# Modeling
교수님 생각으로는 가장 중요한 파트라고 생각하신다.     
mysql을 잘 다루는 사람도 중요하지만 설계를 잘하는 사람이 정말 중요하다.    

### 1. 방문내역 + 구매내역 데이터
![image](https://user-images.githubusercontent.com/50114210/65594511-eb457c00-dfcd-11e9-9a96-8ac8b8a12fac.png)       
### 2. 기록된 내용에서 물건 구매 내역이 없는 고객 위로 정렬 (L자형 테이블이 된다.)      
![image](https://user-images.githubusercontent.com/50114210/65594589-18922a00-dfce-11e9-8cc6-43674535734b.png)      
### 3. L자형 테이블을 빈칸이 있는 곳과 없는 곳으로 분류
![image](https://user-images.githubusercontent.com/50114210/65594945-abcb5f80-dfce-11e9-9d0d-9ae644b1723b.png)    
### 4. 테이블 간의 업무적 연관성 정의
![image](https://user-images.githubusercontent.com/50114210/65594724-53945d80-dfce-11e9-89a4-3e86d147e739.png)   
### 5. 테이블 구조 정의
![image](https://user-images.githubusercontent.com/50114210/65594793-69098780-dfce-11e9-93e2-bdb61fc88e31.png)    
### 6. 모델링
![image](https://user-images.githubusercontent.com/50114210/65594398-a7527700-dfcd-11e9-84fc-23c430442c0e.png)            
