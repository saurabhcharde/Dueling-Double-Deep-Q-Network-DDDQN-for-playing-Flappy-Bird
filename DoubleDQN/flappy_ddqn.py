#try to train the agent by giving state parameters from game.getGameState() as input instead of screen pixels
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
import os
import time
import traceback
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.pyplot as plt

class DQN():
    
    def __init__(self,state_space,action_space):
        
        self.state_space=state_space
        self.action_space=action_space
        self.learning_rate=0.001
        self.epsilon=1
        self.epsilon_decay=0.999
        self.epsilon_min=0.001
        self.batch_size=32
        self.discount_factor = 0.99
        self.min_memory_len=2000
        self.model=self.build_model()
        self.target_model=self.build_model()
        self.memory=deque(maxlen=10000)
        
    def build_model(self):
        
        model=Sequential()
        model.add(Dense(units=24,input_dim=self.state_space,activation='relu'))
        model.add(Dense(units=24,activation='relu'))
        model.add(Dense(units=self.action_space,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def get_updated_weights(self):
        self.target_model.set_weights(self.model.get_weights())
        
        
    def store_experience(self,exp):
        self.memory.append(exp)
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    
    def get_action(self,state):        
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_space)
        else:
            qvalues=self.model.predict(state)
            return np.argmax(qvalues[0])
        
    def train_model(self):
        
        if(len(self.memory)<self.min_memory_len):
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_space))
        update_target = np.zeros((batch_size, self.state_space))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][np.argmax(target[i])])

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
            

if __name__=="__main__":
    
    try:
       
        game = FlappyBird()
        p = PLE(game, fps=30, display_screen=True)
        p.init()
        #env=gym.make('CartPole-v0')  
        state_space=8
        action_space=len(p.getActionSet())
        
        agent=DQN(state_space,action_space)
        #agent.model.load_weights('Flappy_ddqn.h5')
        final_score=0
        y=list()
        x=list()
        for e in range(1,3000):
            
            print("episode{}/{}".format(e,3000))
            #state=env.reset()
            state=p.reset_game()
            state = np.reshape(state, [1, state_space])
            done=False 
            score=0
            display_score=0
            while not done:
                #env.render()
                action=agent.get_action(state)
                next_state , reward ,done = p.act(action)
                #next_state,reward,done,info=env.step(action)
                next_state = np.reshape(next_state, [1, state_space])               
                agent.store_experience((state,action,reward,next_state,done))               
                agent.train_model()               
                state=next_state 
                if(reward<0):
                    reward=-1
                display_score+=reward
                '''
                if(reward>0):
                    score+=1
                time.sleep(0.01)
                '''
                if done:
                    agent.get_updated_weights()
                    print("episode:", e, "  score:", display_score)
                    final_score+=display_score
                    y.append(final_score/e)
                    x.append(e)
                    #print("episode:", e, "  score:", score, "  memory length:",
                          #len(agent.memory), "  epsilon:", agent.epsilon)    
            if e % 50 == 0:
                agent.model.save_weights("Flappy_ddqn.h5")
                #plt.plot(agent.epochs,agent.loss)
            if e % 500 ==0:    
                plt.plot(x,y)
                plt.title('DDQN')
                plt.ylabel('cumulative reward')
                plt.xlabel('episodes')
                plt.show()
                
    except:
        tb = traceback.format_exc()
        print(tb)
        os.system('pause')
        