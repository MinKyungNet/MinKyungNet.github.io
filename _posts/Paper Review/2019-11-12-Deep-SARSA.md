---
layout: post
title: "Deep SARSA"
tags: [Deep SARSA, Reinforce Learning]
categories: [Paper Review]
---

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vR6oviM9nzmiG8laRUuNugLgf4JBdFB3EcsK2dBGDtMxqKTFSBQ9NrFx7HSfv6wPQ/embed?start=false&loop=false&delayms=3000" frameborder="0" width="640" height="389" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# 참고한 자료
https://teamsaida.github.io/SAIDA_RL/installation/          
여기에서 설치할 수 있다. 사이다 팀의 코드와 환경설정까지 금방 끝낼 수 있을 것이다.      

https://teamsaida.github.io/SAIDA_RL/AvoidReavers/             
이 글은 AvoidReavers를 DeepSARSA로 돌린 코드를 리뷰한 것이다.


# Deep SARSA란?
2016년 ieee 논문

Deep = Q 함수로 딥 뉴럴 네트워크 알고리즘을 사용        
S = 상태(state)          
A = 행동(action)           
R = 보상(reward)           
S = 다음 상태(state')          
A = 다음 행동(action')        

# On-Policy 방법
on-policy : 학습하는 policy와 행동하는 policy가 반드시 같아야만 학습이 가능한 강화학습 알고리즘         
episode의 실행 중에도 언제든지 학습이 가능            
코드에서도 네트워크 하나만 사용한다.

# Uniform Sampling
논문에서는 데이터 간의 연관성을 심하게 높이지 않는 선에서 상태를 어느정도 파악할 수 있는 4프레임을 concatenate한다고 했지만, SAIDA의 Deep SARSA에서는 1프레임만 사용하고 있다. 1프레임에 게임에서 알아야할 모든 정보가 들어있기 때문이다.
```python
# 한 프레임의 상태 만들기
# preprocess for observation
class ReaverProcessor(Processor):
    def process_observation(self, observation, **kwargs):
    """ Pre-process observation

    # Argument
        # 현재 환경의 상황
        observation (object): The current observation from the environment.

    # Returns
        # 상황의 필요한 정보
        processed observation

    """

    """
    observation
    my_unit {
          unit_type: "Terran_Dropship"
          hp: 150
          pos_x: 78
          pos_y: 42
          velocity_x: 1.26953125
          velocity_y: 0.71875
          angle: 0.6381360077604268
          accelerating: true
    }
    """
    if len(observation.my_unit) > 0:
        # 드랍쉽 5개, 리버 6개 * 3마리
        # 5 features of a Dropship  + 6 features of 3 Reavers
        STATE_SIZE = 5 + 3 * 6

        s = np.zeros(STATE_SIZE)
        me = observation.my_unit[0]
        # 드랍쉽의 상태 파악
        # Observation for Dropship
        s[0] = scale_pos(me.pos_x)  # X of coordinates
        s[1] = scale_pos(me.pos_y)  # Y of coordinates
        s[2] = scale_velocity(me.velocity_x)  # X of velocity
        s[3] = scale_velocity(me.velocity_y)  # y of coordinates
        s[4] = scale_angle(me.angle)  # Angle of head of dropship



        # 리버의 상태 파악
        # Observation for Reavers
        for ind, ob in enumerate(observation.en_unit):
            """
            process ind 0
            process ob unit_type: "Protoss_Reaver"
            hp: 100
            shield: 80
            cooldown: 9
            pos_x: 127
            pos_y: 124
            velocity_x: -0.484375
            velocity_y: -1.9375
            angle: 4.466952054322987
            accelerating: true

            process ind 1
            process ob unit_type: "Protoss_Reaver"
            hp: 100
            shield: 80
            cooldown: 10
            pos_x: 185
            pos_y: 232
            velocity_x: 1.4453125
            velocity_y: -1.3828125
            angle: 5.522330836388308
            accelerating: true

            process ind 2
            process ob unit_type: "Protoss_Reaver"
            hp: 100
            shield: 80
            cooldown: 8
            pos_x: 287
            pos_y: 80
            velocity_x: -0.70703125
            velocity_y: 0.70703125
            angle: 2.380738182798515
            accelerating: true
            """
            s[ind * 6 + 5] = scale_pos(ob.pos_x - me.pos_x)  # X of relative coordinates, 상대 위치
            s[ind * 6 + 6] = scale_pos(ob.pos_y - me.pos_y)  # Y of relative coordinates, 상대 위치
            s[ind * 6 + 7] = scale_velocity(ob.velocity_x)  # X of velocity
            s[ind * 6 + 8] = scale_velocity(ob.velocity_y)  # Y of velocity
            s[ind * 6 + 9] = scale_angle(ob.angle)  # Angle of head of Reavers, 리버 머리 각도
            s[ind * 6 + 10] = scale_angle(1 if ob.accelerating else 0)  # True if Reaver is accelerating
    return s
```

# 환경 만들기

```python
env = AvoidReavers(action_type=0, move_angle=30, move_dist=2, frames_per_step=1, verbose=0, no_gui=False)
```
* action_type : action의 타입 설정, 0(이산), 1 and 2(연속)
* move_angle : ex) action_type이 0(이산)일 경우에 move_angle이 30이라면, 360 / 30 = 12 방향으로 움직일 수 있게 된다.
* move_dist : 움직일 거리
* frame_per_step : 한 액션당 몇 프레임을 스킵할 것인지

# USE CNN
 Similarly, in
deep SARSA learning, the value function approximation is still
with the convolution neural network (CNN).             

논문에서는 CNN사용했다고 하는데 코드에서는 그냥 NN을 사용했다.
한 프레임에 표현되는 드랍쉽과 리버의 행동이 전부 표현이 되어있기 때문이다.          
ex) 드랍쉽의 방향, 가중치, 리버들의 방향, 가
```python
    # 뉴럴 네트워크
    # Create a model
    model = Sequential()
    model.add(Dense(50, input_dim=state_size, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.summary()
    
    """
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 50)                1200      
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                2550      
    _________________________________________________________________
    dense_3 (Dense)              (None, 30)                1530      
    _________________________________________________________________
    dense_4 (Dense)              (None, 13)                403       
    =================================================================
    Total params: 5,683
    Trainable params: 5,683
    Non-trainable params: 0
    """
```

# Epsilon
탐험을 위해 사용          
Similar to DQN, given the current state s, the action a is selected by e -greedy method.             
a' is the next action selected bye -greedy.          

```python
class DeepSARSAgent(Agent):
    def forward(self, observation):
    """
    observation
    (23,) # shape
    [ 4.          2.          1.03125     0.93359375 -0.640625    3.리버1
      5.         -0.484375   -1.9375      0.421875   -0.6816901   6.리버2
     12.          1.4140625  -1.4140625   0.7578125  -0.6816901  13. 리버3
      2.         -2.1210938   2.1210938  -0.2421875  -0.6816901 ] 드랍쉽
    forward action 4
    """

    # e-greedy
    # epsilon 행동에 걸리면 랜덤하게 행동
    # 아니면 모델의 예측에 맞게 행동
    if self.train_mode and np.random.rand() <= self.epsilon:
        action = random.randrange(self.action_size)
    else:
        state = np.float32(observation)
        q_values = self.model.predict(np.expand_dims(state, 0))
        action = np.argmax(q_values[0])

    # backfoward 할 때 바로 쓰임
    # set emory for training
    self.recent_observation = observation
    self.recent_action = action
    return [action]
```

# Loss function
Q(S,A) <- Q(S,A) + alpha(R + gammaQ(S',A') - Q(S,A))

Loss function = (R + r * Q(s', a') - Q(s, a))^2
에이전트가 다음 스텝에서 취한 행동의 값을 직접 사용
```python
    def backward(self, reward, terminal):
        """ Updates the agent's network
                """
        self.observations.append([self.recent_observation, self.recent_action, reward, terminal])

        # 스텝 0이면 리턴하기 때문에 observations의 크기는 2로 유지된다.
        if self.step == 0:
            return

        # e-greedy 부분
        # Decaying the epsilon
        # 랜덤하게 행동할 확률을 줄여나간다.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Use a memory to train
        experience = self.observations.popleft()
        
        # S, A, R
        state = np.float32(experience[0]) # 상황
        action = experience[1]            # 행동
        reward = experience[2]            # 보상 목표 도착(+1), 리버랑 부딪힘(-1), 맵 밖을 선택(-0.1)
        done = experience[3]              # 목표 도착? True, False
        
        """
        experience
        
        state
        array([ 3.        ,  1.        ,  0.4296875 ,  0.41015625, -0.609375,
        5.        ,  6.        , -1.3046875 , -1.515625  ,  0.2421875 ,
       -0.68169011,  7.        , 17.        ,  2.        ,  0.        ,
       -1.        , -0.68169011, 15.        ,  2.        , -0.70703125,
        0.70703125, -0.2421875 , -0.68169011])
        
        action
        7
        
        reward
        -0.10000000149011612
        
        done
        False]
        """
        
        # 현재 모델을 통해 next_state와 action을 얻는다.
        # Get next action on next state from current model
        # S', A'는 e-greedy를 적용
        next_state = np.float32(self.recent_observation)
        next_action = self.forward(next_state)

        # Compute Q values for target network update
        # Q(S,A) <- Q(S,A) + alpha(R + gammaQ(S',A') - Q(S,A))
        # e-greedy를 적용하지 않고 모델을 통해 예상
        target = self.model.predict(np.expand_dims(state, 0))[0]
        if done: # 목표 도착하면 보상
            target[action] = reward
        else: # 아니면 Loss
            # target = R + Gamma * Q(S', A')
            target[action] = (reward + self.discount_factor *
                              self.model.predict(np.expand_dims(next_state, 0))[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        """
        action size 13
        target 
        [[-0.08781486  0.24054345  0.86039156 -0.4356818  -0.5917005  -0.16091128
           0.01158049  0.44049442  0.90022075 -0.5263128  -0.17700948 -0.9974993 0.2576919 ]]
        """
        
        # 학습
        self.model.fit(np.expand_dims(state, 0), target, epochs=1, verbose=0)
        return
```

# 메모리 터지는 거 방지해주는 코드

```python
# GPU 메모리가 터지지 않게 조절해주는 코드
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
```
