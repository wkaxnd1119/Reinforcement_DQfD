import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import torch.optim as optim 
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# batch_size = 32
env = gym.make('CartPole-v1')
def get_demo_traj():
    a = np.load("./demo_traj_2.npy", allow_pickle=True)
    return a

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################
class DQfDNetwork(nn.Module):
    def __init__(self):
        super(DQfDNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 32) 
        self.fc2 = nn.Linear(32, 64) 
        self.fc3 = nn.Linear(64, 2) 
        
        self.demo_buffer = deque()
        self.time_step = 0
        
        ## TODO

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x) 
        return x 
    
    
    def select_action(model, env, state, eps):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)
    
        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(values.cpu().numpy())
    
        return action
    
    
    def train(batch, current, target, gamma= 0.95):
        # print('Batch', batch)
    
        states, actions, next_states, rewards, is_done, _ = batch[0], batch[1], batch[2], batch[3], batch[4]
    
        q_values = current(states)
    
        next_q_values = current(next_states)
        next_q_state_values = target(next_states)
    
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + gamma * next_q_value * (1 - is_done)
    
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        return loss
      
    

class ReplayMemory: 
    def __init__(self): 
        self.replay_memory = deque() 
        # _, self.action_batch, _, _, _ =  self.replay_memory

    def write(self, state, action, reward, next_state, done, demo_data): 
        # if len(self.replay_memory) >= 50000: 
        #     self.replay_memory.popleft() 
        self.replay_memory.append((state, action, reward, next_state, done, demo_data)) 
        return  


    def sample(self, batch_size): 
         
        minibatch = random.sample(self.replay_memory, batch_size) 
        batch_s, batch_a, batch_r, batch_n, batch_t, batch_demo = [], [], [], [], [], []

        for transition in minibatch:
            s,a,r,n,t,demo = transition 

            batch_s.append(s) 
            batch_a.append([a]) 
            batch_r.append([r]) 
            batch_n.append(n) 
            batch_t.append([t]) 
            batch_demo.append([demo])
            
    
        return torch.tensor(batch_s, dtype=torch.float), torch.tensor(batch_a, dtype=torch.int64),\
            torch.tensor(batch_r, dtype=torch.float), torch.tensor(batch_n,dtype=torch.float), \
                torch.tensor(batch_t, dtype=torch.float), batch_demo
    

                  
    
    
    
        ## TODO

##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################

EPSILON_END = 0.01
class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.n_EPISODES = n_episode
        self.step = 1
        self.pred_q = DQfDNetwork()
        self.target = DQfDNetwork()
        self.replay = ReplayMemory()
        # self.demo = ReplayMemory()
        self.action_batch = []
        self.state_batch = []
        self.demo_data = []
        self.opt = optim.Adam(self.pred_q.parameters(), lr= 0.005, weight_decay= 1e-5) #L2 reg
        self.demoReplay = deque()
        self.count = 0
        ## TODO
    # def run_DDQN(self):
        
        
    def loss_l(self, ae, a):
        return 0 if ae == a else 0.8
    
    def loss_jeq(self, samples):
        jeq = 0
        s,a,r,n,t, is_demo = samples
        # print('=======a=====', is_demo)
        for i in range(len(samples)):
            ae = a[i]
            # print( '=======',   ae)           
            max_value = float("-inf")
            demo = is_demo[i]
            for action in range(2):

                temp_val = self.pred_q(s)
                # print(' self.pred_q(s)',  self.pred_q(s))
                max_value = max(temp_val[i][action] + self.loss_l(ae, action), max_value)
                max_value = torch.tensor(max_value)
                # print('max_value{}'.format(max_value))
                # print('temp_val[i][ae]{}'.format(temp_val[i][ae]))
                demo = torch.tensor(demo)

            jeq +=  demo* (max_value - temp_val[i][ae])
            # print(jeq)
    
        return jeq


    def store_demo(self, state, action, reward, next_state, done, demoEpisode):
        # state = torch.Tensor(state)
        # next_state = torch.Tensor(next_state)
        # data = (state, action, reward, next_state, done, demoEpisode)
        # episodeReplay.append(data)
        self.replay.write(state, action, reward, next_state, done, demoEpisode)
        

    def get_action(self, state):
        state = torch.Tensor(state).to(device)
        out = self.pred_q(state) 
    
        # select a random action wih probability eps
        
        if np.random.rand(1) < self.epsilon: 
            act = np.random.randint(0, 2) 
        else:
            act = out.argmax().item() 
    
        return act
    
    def update_eps(self):
        self.epsilon = max(EPSILON_END , 1-(self.start*0.0015)) #Epslion 크기를 최소 0.01 보장 
    
    
    def update(self):
        self.opt.zero_grad()
        samples = self.replay.sample(32)
        # print('Samples========', samples[0])
        s,a,r,n,t,is_demo = samples
        # print('=====Sample=====',samples[4])
        
        q_values = self.pred_q(s).gather(1,a)
        # print('=====q_vlue{}'.format(q_values))
    
        # get V(s')
        next_q_values = self.pred_q(n)
        # print('next_state_values', next_q_values)
        #computing expected Q values
        a_prime = torch.argmax(next_q_values.detach(), dim=1)
        # print('a_prime', a_prime)
        a_prime = a_prime.unsqueeze(1)
        # print('a_prime', a_prime)

        
        q_target_next = self.target(n).squeeze(1)
        # print('q_target_next', q_target_next)
        next_q_value = q_target_next.gather(1, a_prime)

        # print('next_q_value', next_q_value)
        # print('R is', r)
        expected_q_value = r +( 0.95 *next_q_value*(1-t))
   
        
        # cal

        # print('expected_q_value', expected_q_value)
        
        # l_dq = r + 0.95 * (q_target_next_s_a - q_values)
        
        mse_loss = nn.MSELoss()
        
        l_dq = mse_loss(expected_q_value.detach(), q_values)
        
    
        # # q_value = q_values.gather(1, a.a(1)).squeeze(1)
        # next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        # expected_q_value = r + 0.95 * next_q_value * (1 - t)

        # l_dq = (q_values - expected_q_value.detach()).pow(2).mean()
        
        
        # Q_predict =  self.pred_q(s).gather(1, a)
        
        # # get Q-Target
        # v_s = next_state_values = self.q_target(n)
        
        # #computing expected Q values
        # v_s = v_s.detach().max(1)[0]
        # v_s = v_s.unsqueeze(1)
        
        # Q_target = (v_s * DISCOUNT_RATE) + r
        
        # l_dq = nn.MSELoss(Q_target, Q_predict)
                                            
        # l_dq = self.pred_q.train(samples, self.pred_q, self.target)
        l_jeq = self.loss_jeq(samples)
        
        J = l_dq + l_jeq # 총 loss
        J.backward()
 
        self.opt.step()
        
        if self.count >= 100:
            self.count = 0
            self.target.load_state_dict(self.pred_q.state_dict())
        else: 
            self.count += 1
        self.update_eps()
        

        # l_dq = self.pred_q.train(sample_batch, self.pred_q, self.target)
        # l_jeq = self.loss_jeq(batch)

            

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######
        
        # PRETRAIN Section
        
        demo_play = get_demo_traj()
        self.start = 0
        for i in range(len(demo_play)):
            for j in demo_play[i]: 
                s,a,r,s_,done = j
                # print('-----done---', done)
                self.store_demo(s,a,r,s_,done, 1)

        for i in range(1000):
            if i % 100 == 0:
                print('pretraining:', i)
            self.update()
            self.start += 1
            
        ## TODO


        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########
            epi_count = 0

            ## TODO

            done = False
            state = env.reset()

            while not done:
                ## TODO

                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                # reward = reward if not done or score == 499 else -100
                self.replay.write(state, action, reward, next_state, done, 0)
                self.update()
                state = next_state
                done_mask = 0 if done is True else 1
                
                # print(len(self.replay.replay_memory))

                ## TODO

                next_state, reward, done, _ = env.step(action)
                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########

                ## TODO

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if (done_mask == 0):
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########
            print('에피소드 Reward ==', test_episode_reward)
                # epi_count += 1
            print(len(self.replay.replay_memory))

                ## TODO

            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########

            ## TODO

        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########

