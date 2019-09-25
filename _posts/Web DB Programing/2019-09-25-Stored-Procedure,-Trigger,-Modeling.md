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

# Modeling
### 1. 방문내역 + 구매내역 데이터
![image](https://user-images.githubusercontent.com/50114210/65594511-eb457c00-dfcd-11e9-9a96-8ac8b8a12fac.png)       
### 2. 기록된 내용에서 물건 구매 내역이 없는 고객 위로 정렬 (L자형 테이블이 된다.)      
![image](https://user-images.githubusercontent.com/50114210/65594589-18922a00-dfce-11e9-8cc6-43674535734b.png)      
### 3. L자형 테이블을 빈칸이 있는 곳과 없는 곳으로 분류
![image](https://user-images.githubusercontent.com/50114210/65594398-a7527700-dfcd-11e9-84fc-23c430442c0e.png)        
### 4. 테이블 간의 업무적 연관성 정의
![image](https://user-images.githubusercontent.com/50114210/65594724-53945d80-dfce-11e9-89a4-3e86d147e739.png)   
### 5. 테이블 구조 정의
![image](https://user-images.githubusercontent.com/50114210/65594793-69098780-dfce-11e9-93e2-bdb61fc88e31.png)    
