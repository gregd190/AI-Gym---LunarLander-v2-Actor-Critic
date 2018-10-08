# # A2C Solution to OpenAI Gym LunarLander-v2 environment
# 
# ## Introduction:
#     
# ### OpenAI Gym Environment Description:
# "Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
# "
# 
# ### Discussion
# Due to the continuous and varied nature of the state space, discretising the state space at a sufficiently high resolution would result in an impractically large number of possible states. A conventional Q-table type solution is therefore impractical.
# An actor - Critic method is used, training 2 neural networks, an 'actor' network to determine the optimal action, and a 'critic' network to estimate the potential reward of the action. 
# 
# Keras, as a frontend for Tensorflow, is used to create and train the neural networks. 
# 
# As 'solved' is considered 200 points, the networks will be trained and optimized until the average over the last 100 iterations exceeds this value. 
# 
# This version enables multiprocessing to allow training of multiple models simulteously to speed up the hyperparameter search on a CPU-only tensorflow setup.
# It is set to train 3 models at a time, as that provided best performance on a 4 core machine, but can be easily changed to suit higher-performance cpus. 

#Import the various gym, keras, numpy and libraries we will require

import gym
import gym.spaces
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import time

from collections import deque
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from multiprocessing import Pool

# ### Creating the models
# 
# Functions for model creation allow for flexibility in network size to allow for comparison of network sizes. 
# 
# Adam is used as the optimizer, as it has proven efficient on prior problems.
# 

def build_model_critic(num_input_nodes, num_output_nodes, lr, size):
    
    model = Sequential()
    
    model.add(Dense(size[0], input_shape = (8,), activation = 'relu'))
    
    for i in range(1,len(size)):
        model.add(Dense(size[i], activation = 'relu'))
    
    
    model.add(Dense(num_output_nodes, activation = 'linear')) 
    
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss = 'mse', optimizer = adam, metrics=['acc'])
    
    return model

def build_model_actor(num_input_nodes, num_output_nodes, lr, size):
    
    model = Sequential()
    
    model.add(Dense(size[0], input_shape = (num_input_nodes,), activation = 'relu'))
    
    for i in range(1, len(size)):
        model.add(Dense(size[i], activation = 'relu'))
    
    model.add(Dense(num_output_nodes, activation = 'softmax')) 
    
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = adam)
    
    return model

# ### Deciding on an Action
# 
# Action state is very simple - one of 4 possible actions (do nothing, or fire left, right or main engine). Action is selected randomly from the 4 actions, with the probability of a given action being chosen being proportional to the probability the actor network give for that action being the optimal action. This inherently encourages exploration in the early stages of training, and moves to a exploitation strategy as the network becomes more sure of itself. 
# 

def decide_action(actor, state):

    flat_state = np.reshape(state, [1,8])
    action = np.random.choice(4, 1, p = actor.predict(flat_state)[0])[0]
    
    return(action)

# ### Running episodes
# 
# The simulation is run for a predefined number of episodes.
# 
# For each step, the state, action, resulting state, reward and whether or not the step completed the episode (the boolean 'done') were saved in a list 'memory'.
# 
# For each episode the totalreward is saved in an array 'totrewardarray'.
# 
# Each episode is limited to 1000 timesteps, to cut short scenarios where the lander (which contains infinite fuel) refusing to land in the early stages of training.
# 
# The episodes run until either the predefined number of episodes are completed, or the problem is considered solved (average totalreward of last 100 episodes exceeds 200). 

def run_episodes(env, actor, r = False, iters=1):
    
    memory = deque()
    
    totrewardarray = []
    
    bestyet = float('-inf')
            
    i = 0
            
    while i < iters:
        
        i += 1
        state = env.reset()
        
        totalreward = 0
        
        cnt = 0 
        
        done = False
        
        while not done and cnt <1000:
            
            cnt += 1
            
            if r:
                env.render()
            
            action = decide_action(actor, state)
        
            observation, reward, done, _ = env.step(action)  
            
            totalreward += reward
                
            state_new = observation 
            
            memory.append((state, action, reward, state_new, done))
            
            state = state_new
            
        totrewardarray.append(totalreward)

    return(memory, totrewardarray)

# ### Training the Networks
# 
# Now the memory list gathered from running the episodes to a training function which trains the networks. 
# 
# The training data is shuffled so it is not presented to the networks in order. 
# 
# The discount factor, 'gamma', is another hyperparameter that will need to be optimised. 

def train_models(actor, critic, memory, gamma):

    random.shuffle(memory)
    
    for i in range(len(memory)):
            
        state, action, reward, state_new, done = memory[i]
            
        flat_state_new = np.reshape(state_new, [1,8])
        flat_state = np.reshape(state, [1,8])
        
        target = np.zeros((1, 1))
        advantages = np.zeros((1, 4))

        value = critic.predict(flat_state)
        next_value = critic.predict(flat_state_new)

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + gamma * (next_value) - value
            target[0][0] = reward + gamma * next_value
        
        actor.fit(flat_state, advantages, epochs=3, verbose=0)
        critic.fit(flat_state, target, epochs=3, verbose=0)     
        
# ### Running episodes without training
# 
# Sometimes we might want to run episodes without saving data for training, for instance if we want to render a few episodes of the trained network, or if we want to assess the performance of a trained network. This is simply a modification of the 'run_episodes' function. 
# 
# It includes a render option (boolean 'r') which turns on or off rendering the episode. 

def play_game(env, iters, r = True):
    
    totalrewardarray = []
    
    for i in range(iters):
    
        totalreward = 0
        cnt = 0
        
        state = env.reset()
        
        done = False
        
        while not done and cnt <1000:
            
            cnt += 1
            
            if r:
                env.render()
                                        
            action = decide_action(actor, state)
                
            observation, reward, done, _ = env.step(action)  
            
            totalreward += reward
            
            state_new = observation 
            
            state = state_new
            
        totalrewardarray.append(totalreward)
        
    return totalrewardarray


# Function to run episodes, train the networks and output an array of the rewards from each episode to allow us to plot the performance of the models. 

def run_and_train(a_size, c_size, alr, clr, gamma, numepisodes):
    
    env = gym.make('LunarLander-v2')
        
    totrewardarray = []
    
    actor = build_model_actor(num_input_nodes = 8, num_output_nodes = 4, lr = alr, size = a_size)
    critic = build_model_critic(num_input_nodes = 8, num_output_nodes = 1, lr= clr, size = c_size)
    
    totrewardarray = [] 
    
    best = float('-inf')
    
    episodes = 0
    
    while episodes < numepisodes:   
        
        memory, temptotrewardarray = run_episodes(env, actor, r = False, iters = 1)
        
        totrewardarray.extend(temptotrewardarray)
        
        episodes = len(totrewardarray)
        
        if episodes >= 100:
            score = np.average(totrewardarray[-101:-1])
            if score>best:
                best = score
            if episodes%100==0:
                print('Episode ', episodes, 'of',numepisodes, 'Average Reward (last 100 eps)= ', score, 'Best = ', best)
            
        train_models(actor, critic, memory, gamma)
        
    avgarray = []
    cntarray = []

    for i in range(100,len(totrewardarray),10):
        avgarray.append(np.average(totrewardarray[i-100:i]))
        cntarray.append(i)
    
    params = 'A_size='+str(a_size)+'C_size='+str(c_size)+'ALR='+str(alr)+'CLR='+str(clr)+'gamma='+str(gamma)+'best='+str(round(best,1))
    
    return(cntarray, avgarray, params)
    
if __name__ == '__main__':
    
    with Pool(3) as p:
            x = p.starmap(run_and_train, [([64,64,64],[64,64,64], [5e-6], [5e-4], [0.999], 1500),([64,64],[64,64], [5e-6], [5e-4], [0.999], 1500),([128],[128], [5e-6], [5e-4], [0.999], 1500)])
    
    for i in range(len(x)):
        plt.plot(x[i][0], x[i][1], label = x[i][2])
        
    plt.legend(loc='best')
    plt.show()
    

    

