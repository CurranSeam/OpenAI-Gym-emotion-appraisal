import gym
import numpy as np
from WolfDifferentRewardsEnvironment import WolfDifferentRewardsEnv
from WolfExplorationEnvironment import WolfExplorationEnv
from WolfDifferentTimestepEnvironment import WolfDifferentTimestepEnv
from WolfJoyDistressEnvironment import WolfJoyDistressEnv
from WolfFixedGoalEnvironment import WolfFixedGoalEnv
import random
from IPython.display import clear_output
from time import sleep

frames = []
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

"""Training the agent"""

# Hyperparameters
'''
alpha = 0.3
gamma = 1.0001
epsilon = 0.1 
'''

alpha = 0.1
gamma = 0.6
epsilon = 0.1
eps = 1e-20


# For plotting metrics 
all_epochs = []
all_penalties = []

#env = WolfExplorationEnv() 
#env = WolfDifferentRewardsEnv() 
env = WolfDifferentTimestepEnv()
#env = WolfFixedGoalEnv()

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(1, 10001): #original value 100001
    #state = env.exploreReset() # Prey in states(0, 7, 14). Intial(36, 37, 38)
    #state = env.rewardsReset() # Different rewards scenario
    state = env.centerReset()   # Different Timestep scenario
    #state = env.fixedGoalReset() # Fixed goal scenario

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, last_state, reward, done, info = env.step(action) 

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])  
              
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        
        q_table[state, action] = new_value    
        
        if reward == 0: # original of -10
            penalties += 1         

        state = next_state
        epochs += 1
    
    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
 
v = np.zeros(env.nS)  # initialize value-function
joy = np.zeros(env.nS)
for i in range(1, 11):
        print(joy)
        #print(v)
        # Value iteration
        prev_v = np.copy(v)
        for s in range(env.nS):
            for a in range(env.nA):
                q_sa = [sum([p*(r + 0.9 * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
                v[s] = max(q_sa)
                
                for p, s_, r, _ in env.P[s][a]:
                    joy[s] = (r + v[s] - prev_v[s_]) * (1 - p)
                        
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
#v = [np.log2(i) for i in v if i > 0]
#joy = [np.log2(i) if i > 0 else i for i in joy if i > 0]
v = np.log2(v, out=np.zeros_like(v), where=(v!=0))
joy = np.log2(joy, out=np.zeros_like(joy), where=(joy > 0))
print(len(joy))
#joy = [0 if np.isnan(x) else x for x in joy]
print(joy)

#print(env.P)
print("Training finished.\n")

####################################################################

"render wolf: set environment to illustration's state"
#env.s = 37  # Exploration Scenario
#env.s = env.rewardsReset()   # Different rewards scenario
env.s = env.centerReset() # Different Timestep scenario
#env.s = env.fixedGoalReset() # Fixed goal scenario

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 5

for i in range(episodes):
    #state = env.exploreReset()  # for exploration scenario
    #state = env.rewardsReset() # for the different rewards scenario
    state = env.centerReset()
    #state = env.fixedGoalReset() # Fixed goal scenario
    
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, last_state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
            
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            #'reward': q_table[state, action]
        }
    )
    epochs += 1

    total_penalties += penalties
    total_epochs += epochs
        
#print(env.episode)
print_frames(frames)