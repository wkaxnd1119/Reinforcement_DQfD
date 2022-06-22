# Reinforcement Learning (DQfD)
Using Reinforcement Learning (DQfD) to play cartpole(OpenAI Gym)
![image](https://user-images.githubusercontent.com/85381860/144034127-ffcaedda-dec0-456f-9151-61ebb41ca2bc.png)


# DQfD 설명
강화 학습은 어려운 의사 결정문제를 해결하는데 좋은 성과를 보였다. 하지만 이
러한 알고리즘들은 많은 양의 데이터가 필요하고 학습하는 과정에 있어 많은 시
행착오를 겪기 때문에 학습이 느리다는 단점이 있다. 본 논문에서는 이러한 단점
을 극복하기 위한 알고리즘 DQfD(Deep Q-learning from Demonstrations)를 설명
한다.
최근 몇 년간 의사 결정 문제 해결에 있어 성공적인 케이스가 있었는데 대표적으
로 DQN을 활용한 Atari 게임 학습, end-to-end policy 탐색을 활용한 로봇 모터
제어 등이 있다. 그런데도 아직 이러한 알고리즘들을 자율주행과 비행 같은 현실
세계 환경에 적용하는 데 어려움이 있다. 특히 학습하는데, 있어 수백만의 step을
통해 좋은 정책을 찾아낼 수 있으며 이 부분도 정확도 측면에서 완벽한 모델이
구축되어 있다는 가정하에 가능한 일이다. 실제 환경에 대한 학습 효율을 올리기
위해서 Agent는 실제 도메인에서의 상황과 액션을 학습하여 학습 초기부터 어느
정도의 좋은 성능을 지니고 시작해야 한다.
DQfD는 초기에 Demonstration 데이터(temporal difference(TD) loss와
Demonstrator loss)를 사용하여 사전 학습을 시킨다. Supervised loss는
Demonstrator의 action을 모방학습하고 TD loss는 강화학습을 통해 value function
을 향상한다. 초기 학습 후, Agent는 학습된 정책을 domain(실제 환경 모델)에 적
용한다. Agent의 네트워크는 demonstration과 자체적으로 생성한 데이터가 혼합
되어 업데이트된다. Demonstration과 자체 생성 데이터의 비율을 정하는 것이 이
알고리즘의 성능을 개선하는데, 있어 아주 중요하다. Google의 성과 중 하나는
prioritized replay mechanism을 이용해 자동적으로 비율을 조정하는 것이다. 이러
한 DQfd 알고리즘은 42개 게임 중 41개의 게임에서 PDD DQN 보다 더 향상된
성능을 보였다.
모방학습의 개념은 기존에도 계속해서 연구되었다. 그 중 대표적인 알고리즘이
DAGGER이다. DAGGER는 전문가의 정책을 기준으로 새로운 환경에서 정책을 정
하는 기법을 사용한다. 이 알고리즘의 문제점으로는 학습 과정에서 전문가로부터
Feedback을 받아야 한다는 점이다. 그뿐만 아니라 DAGGER는 강화학습을 겸비한
모방학습 기법이 아니기 때문에 DQfD 처럼 전문가의 역량을 넘을 수 없다.
사전 학습의 목표는 Demonstrator를 모방하여 학습하여 TD updates와 같이 모델
을 학습시키는 것이다. 사전 학습 단계에서 Agent는 Demonstration Data로부터
mini-batch를 샘플링 및 Update 하는데 최소화할 loss는 다음과 같다.
1) Q-learning loss :
2) N-step double Q-learning loss
3) Supervised large margin classification loss
4) L2 regularization loss
Supervised Loss는 효율적인 사전 학습을 하는 데 있어 핵심이다. Demonstrator의
data는 부족하므로 실제 환경에서는 사전에 학습된 데이터가 없을 가능성이 크다. 
만약 Q-learning 네트워크만으로 사전 학습을 시킨다면 다음 state의 Max Value
만을 찾아 학습시킬 것이다. Large Margin Classification Loss를 추가함으로써
Demonstrator와 다른 action을 취할 시, Margin만큼의 상태 값을 차감하는 것이
다. Margin값은 Demonstrator와 같은 action을 취할 시 0이 되고, 다를 경우 양수
의 숫자를 더해준다.
N-step DQN 사전 학습을 통해 성능을 향상하고 L2 정규화를 통해 과적합 현상
을 방지시킨다. 사전 학습이 끝나면 Replay Buffer에 데이터를 쌓고 용량이 차면
가장 오래된 데이터 순으로 지운다. 단, Demonstrator의 data는 지우지 않는다.
DQfD와 PDD DQN의 성능을 비교한 결과, 시작부터 DQfD의 성능이 전반적으로
좋은 것을 확인할 수 있었다. 실제 환경에서는 초반부터 빠르게 학습하는 것이
중요한데 이런 점에서 DQfD는 PDD에 비해 대부분 초반 학습이 빨랐다.
본 논문에서 소개한 주제는 추천 시스템이나 자율주행 같은 일상생활에 적용할
수 있다. 정확한 시뮬레이션 모델은 없지만, 기존에 전문가가 했던 행동들의 데이
터들을 기반으로 학습을 시키는 DQfD의 장점을 이용하면 학습을 효율을 증가시
킬 수 있다.
