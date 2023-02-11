import random 
import time
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Set Values
sidewalk = np.zeros(8*25)
litterChance = .2
wallChance = .1

# Place Litter and Walls
for tile in range(len(sidewalk)):
  litterDecide = random.random()
  wallDecide = random.random()
  sidewalk[tile] = -1
  if litterChance > litterDecide:
    sidewalk[tile] = .5
  if wallChance > wallDecide:
    sidewalk[tile] = -20

# Make 2D array
sidewalk = np.reshape(sidewalk,(8,25))

# Assign Start and End
for tile in range(sidewalk.shape[0]):
  sidewalk[tile][0] = -1

for tile in range(sidewalk.shape[0]):
  sidewalk[tile][-1] = 100

# # Flatten
# sidewalk = sidewalk.ravel()

def step_math(state, action):
  #0= Forward, 1= Right, 2= Left
  if action == 0:
    move_direction = 1    
  elif action ==1:
    move_direction = -25
  else:
    move_direction = 25
  new_state = state + move_direction
  return new_state

def steps(exploration_rate, sidewalk, state, q_table):
  exploration_rate_threshold = random.uniform(0, 1)
  new_state = -1
  # Exploitation
  if exploration_rate_threshold > exploration_rate: 
    action = np.argmax(q_table[state,:]) 
    new_state = step_math(state, action)

  # Exploration
  else: 
    while (new_state <0) or (new_state >= 150):
      action = random.randint(0,actions-1)
      new_state = step_math(state, action)
      

  flat_sidewalk = sidewalk.ravel()
  # print(new_state)
  reward = flat_sidewalk[new_state]

  if (reward == 100) or (reward == -100):
    done = True
  else:
    done = False


  return new_state, reward, done, action

# Set Values 
rewards_all_episodes = []
total_episodes = 0
num_episodes = 5000
max_steps_per_episode = 40

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

actions = 3

# Make Qmap
q_table  = np.zeros([150,actions])

# Make Grass
for i in range(sidewalk.shape[1]):
  q_table[i][1] = -100
  q_table[-i-1][2] = -100
# for i in range(0,q_table.shape[0],25):
#   q_table[i][2] = -100
  

# print(q_table)

complete = 0
for episode in range(num_episodes):
    state = 0
    done = False
    rewards_current_episode = 0
    storeState = []
    for step in range(max_steps_per_episode): 
      new_state, reward, done, action = steps(exploration_rate, sidewalk, state, q_table)
      # print(new_state, reward, done, action)
      
      # Update Q-table for Q(s,a)
      q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

      state = new_state
      rewards_current_episode += reward 
      storeState.append(state)  
      
      
      if done == True: 
        complete += 1
        # print("Last State Location: ", new_state)
        # print("Reward: ", reward)
        break
    total_episodes +=1
    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)
print("Amount completed this batch: ", complete)
print("Total amount of itterations: ", total_episodes)

cmap = ListedColormap(["black", "grey", "magenta", "yellow", "lawngreen"])

fig, axs = plt.subplots(1, figsize=(16, 3), squeeze=False)

for [ax, cmap] in zip(axs.flat, [cmap]):
    ax.set_title('Reward Earned: ' + str(rewards_current_episode-5) + ", Iterations: ")
    psm = ax.pcolor(sidewalk, cmap=cmap, vmin=-12, vmax=17, edgecolors='k', linewidths=2)
    fig.colorbar(psm, ax=ax)


arr = np.zeros([150])
for i in storeState:
  arr[i] = 1
arr = np.reshape(arr,(6,25))

x = [.5]
y = [.5]
for rows in range(len(arr)):
  for columns in range(len(arr[0])):
    if arr[rows][columns] != 0:
      x.append(columns +.5)
      y.append(rows +.5)

import numpy as np
import pprint
import seaborn as sns
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

BOARD_ROWS = 8
BOARD_COLS = 25
START = (1,0)
DETERMINISTIC = False

R_obs = -1
R_lit = 3
a = (0,0)
b = (1,2)
MAX_DIST = distance.euclidean(a,b)
print(MAX_DIST)

class State:
    def __init__(self, module, state=START):
        self.MODS = ['MOD_WALK','MOD_AVOID','MOD_PICKUP']
        self.module = module
        self.board = self.generateBoard()               
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC  
        
    def generateBoard(self):
        board = np.zeros([BOARD_ROWS, BOARD_COLS])
        
        # SIDEWALK
        if self.module == self.MODS[0]:
            # destination
            board[1:BOARD_ROWS-1,BOARD_COLS-1] = 10
            # not sidewalk
            board[0,:] = -1
            board[BOARD_ROWS-1,:] = -1
        
        n = BOARD_ROWS*BOARD_COLS
        obj_pos = [x for x in random.sample(range(n), int(0.4*n))]
        m = int(len(obj_pos)/2)
                
        # OBSTACLES
        if self.module == self.MODS[1]:
            obs_pos = obj_pos[0:m]
            for pos in obs_pos:
                # no obstacles or litter on first or last columns
                if (pos%BOARD_COLS != 0) and (pos%BOARD_COLS != BOARD_COLS-1):
                    board[pos//BOARD_COLS][pos%BOARD_COLS] = R_obs
                
        # LITTER
        if self.module == self.MODS[2]:
            lit_pos = obj_pos[m+1:]
            for pos in lit_pos:
                # no obstacles or litter on first or last columns
                if (pos%BOARD_COLS != 0) and (pos%BOARD_COLS != BOARD_COLS-1):
                    board[pos//BOARD_COLS][pos%BOARD_COLS] = R_lit
                    
        return board
        
    def giveReward(self):     
        if self.module == self.MODS[0]:
            return self.board[self.state]
        
        # if on object, penalize by R_obs
        if self.board[self.state] == R_obs:
            return R_obs
        
        # if on litter, reward by R_lit
        if self.board[self.state] == R_lit:
            return R_lit
        
        # else find dist of closest object
        closest_obj = MAX_DIST
        x = self.state[0]
        y = self.state[1]        
        for i in range(x-1,x+1 +1):
            for j in range(y,y+2 +1):
                # check if window is within boards
                if i < 0 or i >= BOARD_ROWS or j < 0 or j >= BOARD_COLS:
                    continue
                
                # check if position has object
                if self.board[i, j] != 0:
                    a = (i,j)
                    dist = distance.euclidean(a,b)
                    if dist < closest_obj:
                        closest_obj = dist
                
        # 2.25 is approx the furthest distance an object can be in this window
        if self.module == self.MODS[1]:
            reward = min(1, closest_obj / MAX_DIST)
        elif self.module == self.MODS[2]:
            reward = max(0, closest_obj / MAX_DIST)
        
        return reward

    def isEndFunc(self):
        if self.state[1] == BOARD_COLS - 1 and self.state[0] > 0 and self.state[0] < BOARD_ROWS-1:
            self.isEnd = True
        if self.state == R_obs:
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        if self.determine:
            if action == "up":
                nxtState = (self.state[0], self.state[1] + 1)
            elif action == "down":
                nxtState = (self.state[0], self.state[1] - 1)
            elif action == "left":
                nxtState = (self.state[0] - 1, self.state[1])
            else:
                nxtState = (self.state[0] + 1, self.state[1])
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= BOARD_ROWS-1):
            if (nxtState[1] >= 0) and (nxtState[1] <= BOARD_COLS-1):
                return nxtState
        return self.state

    def showBoard(self):
        self.board[self.state] = 2
        for i in range(0, BOARD_ROWS):
            print('----'*BOARD_COLS + '-')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 2:
                    token = 'A'
                if self.board[i, j] == 1:
                    token = '1'
                if self.board[i, j] == -1 and self.module == self.MODS[0]:
                    token = '-'
                if self.board[i, j] == R_obs and self.module == self.MODS[1]:
                    token = '*'
                if self.board[i, j] == R_lit and self.module == self.MODS[2]:
                    token = '$'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('----'*BOARD_COLS + '-')

class Agent:
    def __init__(self, module, lr=0.5, epsilon=0.9):
        self.module = module
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State(module=self.module)
        self.isEnd = self.State.isEnd
        self.lr = lr
        self.exp_rate = 1 - epsilon
        self.epsilon = epsilon
        
        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict
    
    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.State.state
            action = max(self.Q_values[current_position])                
#             print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(module=self.module, state=position)

    def reset(self):
        self.states = []
        self.State = State(module=self.module)
        self.isEnd = self.State.isEnd

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.epsilon * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                
        print(f'Game End Reward {round(reward,4)}')

s = State(module='MOD_WALK')
print(s.state)
print(s.showBoard())

random.seed(8)
sidewalk = Agent(module='MOD_WALK', lr=0.5)
print(sidewalk.module)

sidewalk.play(1000)
final_sidewalk = sidewalk.Q_values

pprint.pprint(sidewalk.Q_values)

hm = np.zeros([BOARD_ROWS, BOARD_COLS])
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLS):
        hm[i,j] = final_sidewalk[(i,j)]['right']

ax = sns.heatmap(hm)

print(hm)
plt.imshow(hm)
plt.show()

random.seed(8)
s = State(module='MOD_AVOID')
print(s.state)
print(s.showBoard())

R_obs = R = -1
rounds = 100
m = 'MOD_AVOID'

random.seed(8)
obstacle = Agent(module=m, lr=0.5)
print(obstacle.module)
obstacle.play(rounds)
final_obstacle = obstacle.Q_values

pprint.pprint(obstacle.Q_values)

hm = np.zeros([BOARD_ROWS, BOARD_COLS])
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLS):
        hm[i,j] = final_obstacle[(i,j)]['left']


ax = sns.heatmap(hm).set_title(f'{m}: {rounds} rounds with reward {R}')

random.seed(8)
s = State(module='MOD_PICKUP')
print(s.state)
print(s.showBoard())

R_lit = R = 100
rounds = 1000
m = 'MOD_PICKUP'

random.seed(8)
litter = Agent(module=m)
print(litter.module)

litter.play(rounds)
final_litter = litter.Q_values

hm = np.zeros([BOARD_ROWS, BOARD_COLS])
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLS):
        hm[i,j] = final_litter[(i,j)]['right']
        
ax = sns.heatmap(hm).set_title(f'{m}: {rounds} rounds with reward {R}')

Q_final = np.zeros([BOARD_ROWS, BOARD_COLS])
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLS):
        u = (final_sidewalk[(i,j)]['up'] + final_obstacle[(i,j)]['up'] + final_litter[(i,j)]['up'])/3
        d = (final_sidewalk[(i,j)]['down'] + final_obstacle[(i,j)]['down'] + final_litter[(i,j)]['down'])/3
        l = (final_sidewalk[(i,j)]['left'] + final_obstacle[(i,j)]['left'] + final_litter[(i,j)]['left'])/3
        r = (final_sidewalk[(i,j)]['right'] + final_obstacle[(i,j)]['right'] + final_litter[(i,j)]['right'])/3
        
        Q_final[i,j] = u
        
ax = sns.heatmap(hm)

BOARD_ROWS = 8
BOARD_COLS = 25
START = (1,0)
DETERMINISTIC = False

alpha = [0.2, 0.5, 0.7]
reward = [1, 10, 100]
rounds = [100, 1000, 2500]

random.seed(8)

module = 'MOD_WALK'

s = State(module=module)
print(s.state)
print(s.showBoard())

random.seed(8)

Q_sidewalk = []

for a in alpha:  
    for n in rounds:
        agent = Agent(module=module, lr=a)
        agent.play(n)
        Q_sidewalk.append(agent.Q_values)
        print(f'\t\t with alpha {a} and rounds {n}')

action = 'up'

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
fig.suptitle(f'Q_Tables ({action}) for Sidewalk Module')

A = [0.2,0.2,0.2, 0.5,0.5,0.5, 0.7,0.7,0.7]
R = [100,1000,2500, 100,1000,2500, 100,1000,2500]

for s in range(9):
    hm = np.zeros([BOARD_ROWS, BOARD_COLS])
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            hm[i,j] = Q_sidewalk[s][(i,j)][action]
    sns.heatmap(hm, ax=axes.flat[s]).set_title(f'MOD_WALK: {R[s]} rounds - alpha {A[s]}')

random.seed(8)

module = 'MOD_AVOID'

s = State(module=module)
print(s.state)
print(s.showBoard())

random.seed(8)

Q_obstacles = []

for a in alpha:  
    for r in reward:
        R_obs = -r
        agent = Agent(module=module, lr=a)
        agent.play(n)
        Q_obstacles.append(agent.Q_values)
        print(f'\t\t with alpha {a} and reward {-r}')

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
fig.suptitle(f'Q_Tables ({action}) for Obstacle Module')

A = [0.2,0.2,0.2, 0.5,0.5,0.5, 0.7,0.7,0.7]
Re = [1,10,100, 1,10,100, 1,10,100]

for s in range(9):
    hm = np.zeros([BOARD_ROWS, BOARD_COLS])
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            hm[i,j] = Q_obstacles[s][(i,j)][action]
    sns.heatmap(hm, ax=axes.flat[s]).set_title(f'MOD_AVOID: alpha {A[s]} - reward {-Re[s]}')

random.seed(8)

module = 'MOD_PICKUP'

s = State(module=module)
print(s.state)
print(s.showBoard())

random.seed(8)

Q_litter = []

for a in alpha:  
    for r in reward:
        R_obs = r
        agent = Agent(module=module, lr=a)
        agent.play(n)
        Q_litter.append(agent.Q_values)
        print(f'\t\t with alpha {a} and reward {r}')

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
fig.suptitle(f'Q_Tables ({action}) for Litter Module')

A = [0.2,0.2,0.2, 0.5,0.5,0.5, 0.7,0.7,0.7]
Re = [1,10,100, 1,10,100, 1,10,100]

for s in range(9):
    hm = np.zeros([BOARD_ROWS, BOARD_COLS])
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            hm[i,j] = Q_litter[s][(i,j)][action]
    sns.heatmap(hm, ax=axes.flat[s]).set_title(f'MOD_PICKUP: alpha {A[s]} - reward {Re[s]}')

action = 'left'
Q_final = np.zeros([BOARD_ROWS, BOARD_COLS])
for i in range(BOARD_ROWS):
    for j in range(BOARD_COLS):
        a = (Q_sidewalk[4][(i,j)][action] + Q_obstacles[5][(i,j)][action] + Q_litter[8][(i,j)][action])/3        
        Q_final[i,j] = a
        
ax = sns.heatmap(hm)
