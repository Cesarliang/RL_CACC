from __future__ import absolute_import
from __future__ import print_function
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pdb

import os
import sys
import optparse
import random

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

record = open('log_1_1.txt','w')
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 128)
        self.affine2 = nn.Linear(128, 128)
        self.affine3 = nn.Linear(128, 64)
        #print("hidden layer", self.affine1.weight)
        # actor's layer
        # the number of actions is 4, leading vehicle is not controlled
        self.policy_mean = nn.Linear(64, 4)
        self.policy_log_std = nn.Linear(64,4)
        #5 vehicles, each has 5 gear values to choose
        #self.gear = nn.Linear(128,25)

        # critic's layer
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.relu(x)
        x = self.affine3(x)
        x = F.relu(x)
        #x = F.relu(self.affine1(x))
        #if torch.any(torch.isnan(x)):
            #pdb.set_trace()
        
        #print("hidden layer", x)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        mean = self.policy_mean(x)
        log_std = self.policy_log_std(x)
        #h = self.gear(x)
        #log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        #gear_prob1 = F.softmax(self.gear(x), dim=-1)
        #gear_prob2 = F.softmax(self.gear(x), dim=-1)
        #gear_prob3 = F.softmax(self.gear(x), dim=-1)
        #gear_prob4 = F.softmax(self.gear(x), dim=-1)
        #gear_prob5 = F.softmax(self.gear(x), dim=-1)
        #gear_prob = F.softmax(h.reshape(5,5), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        #return mean, log_std, gear_prob1, gear_prob2, gear_prob3, gear_prob4, gear_prob5, state_values
        return mean, log_std, state_values
		
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
eps = np.finfo(np.float32).eps.item()
scaler = MinMaxScaler()


def sumo_reset():
    #reset the state of each episode
    traci.load(["-c", "samplenet.sumocfg"])

    traci.edge.setMaxSpeed('12', 20)
    
    while traci.vehicle.getIDCount() < 5:
          
        traci.simulationStep()
 
    ini_pos = np.zeros(5)
    ini_speed = np.zeros(5)
    ini_pos_dev = np.zeros(4)
    ini_speed_dev = np.zeros(4)
    order = 0

    for veh_id in traci.vehicle.getIDList():
        
        ini_pos[order]= traci.vehicle.getPosition(veh_id)[0]
        ini_speed[order] = traci.vehicle.getSpeed(veh_id)
        order =order + 1
    
    for order in range(4): 
          
        ini_pos_dev[order] = ini_pos[order]-ini_pos[order+1]
        ini_speed_dev[order] = ini_speed[order]-ini_speed[order+1]

    #record.write(str(pos) + '   ' + str(speed) + '\n') 
    #record.write(str(pos_dev) + '   ' + str(speed_dev) + '\n') 
    
    return ini_pos.astype(np.float32), ini_speed.astype(np.float32), ini_pos_dev.astype(np.float32), ini_speed_dev.astype(np.float32)
	
def select_action(current_pos, current_speed, current_posdev, current_speeddev, epsilon=1e-6):

    #current_state = np.append(current_pos, current_speed)
    #current_state = np.append(current_state, current_posdev)
    #current_state = np.append(current_state, current_speeddev)
    current_state = np.append(current_posdev, current_speeddev)
    current_state = current_state.reshape(-1,1)
    current_state = scaler.fit_transform(current_state)
    current_state = current_state.reshape(8,)
    current_state = torch.from_numpy(current_state).float()
    normal_mean, normal_log_std, state_value = model(current_state)
    normal_std = normal_log_std.exp()
    
    #print("current state", current_state)
    #print("action selection probability", normal_mean)

    normal = Normal(0, 1)
    z0      = normal.sample()
    action = 2.5*torch.tanh(normal_mean + normal_std*z0)
    
    #model.saved_actions.append(SavedAction(Normal(normal_mean, normal_std).log_prob(normal_mean+ normal_std*z0) + torch.log(25 - action.pow(2) + epsilon), state_value))     
    model.saved_actions.append(SavedAction(Normal(normal_mean, normal_std).log_prob(normal_mean+ normal_std*z0), state_value))
    #model.saved_actions.append(SavedAction(Normal(normal_mean, normal_std).log_prob(normal_mean+ normal_std*z0), state_value))
    
    return action.detach().numpy()
	
def take_action(current_pos, current_speed, current_posdev, current_speeddev, action):

    order = 0
          
    for veh_id in traci.vehicle.getIDList():
        
        traci.vehicle.setSpeedFactor(veh_id, 1)
        traci.vehicle.setTau(veh_id, 1)
        
        if veh_id == 'veh0':
           traci.vehicle.setSpeed(veh_id, 12)

        else:
        #updated_speed is equal to the current_speed plusing the acceleration (i.e., action)
           traci.vehicle.setSpeed(veh_id, max(0, current_speed[order] + action[order-1]))
    
        order = order + 1
             
    traci.simulationStep()
            
    updated_pos = np.zeros(5)
    updated_speed = np.zeros(5)
    updated_pos_dev = np.zeros(4)
    updated_speed_dev = np.zeros(4)
    reward = 0
    order = 0

    for veh_id in traci.vehicle.getIDList():
          
        updated_pos[order]= traci.vehicle.getPosition(veh_id)[0]
        updated_speed[order] = traci.vehicle.getSpeed(veh_id)
        order =order + 1
    
    for order in range(4): 
          
        updated_speed_dev[order] = updated_speed[order]-updated_speed[order+1]
        updated_pos_dev[order] = (updated_pos[order]-updated_pos[order+1])/(updated_speed[order+1] + 0.001)
        #updated_pos_dev[order] = (updated_pos[order]-updated_pos[order+1])
            
        if updated_pos_dev[order] >= 1.3:
            reward = reward - np.square(updated_pos_dev[order]-1.3)-np.square(updated_speed_dev[order]) - np.square(current_speed[order+1] + action[order] - updated_speed[order+1])
            #reward = reward - np.log(np.square(updated_pos_dev[order]-1)) - np.square(updated_speed_dev[order]) - np.square(action[order])
        
        else:
            reward = reward - 10000
        
        #reward = reward + (25-np.square(action)).sum()、
    
    return updated_pos.astype(np.float32), updated_speed.astype(np.float32), updated_pos_dev.astype(np.float32), updated_speed_dev.astype(np.float32), reward.astype(np.float32)
    
def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    
    del(model.rewards[-1])
    del(saved_actions[-1])
    
    #scaling = np.array([model.rewards]).reshape(-1,1)
    #scaling = scaler.fit_transform(scaling)
    #scaling = scaling.reshape(scaling.shape[0],)
    #model.rewards = scaling.tolist()
    

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        # pdb.set_trace()
        #R = r + 0.9 * R
        R = r + R
        returns.insert(0, R)
    # pdb.set_trace()
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std(unbiased = False) + eps)
    returns = returns.float()
    #print(returns.dtype)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        #advantage = R - value
        #print(log_prob.dtype)
        #print(advantage.dtype)

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    print("loss", loss)
    #loss = loss.float()
    # pdb.set_trace()
    #print(loss.dtype)

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
	
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options
	
def main():
# this is the main entry point of this script
#if __name__ == "__main__":
    options = get_options()
    #num_run = 0

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    #if options.nogui:
    sumoBinary = checkBinary('sumo')
    #else:
        #sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "samplenet.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    
    num_episode = 50000
         
    A = np.zeros((num_episode, 200, 4))
    P = np.zeros((num_episode, 200, 5))
    V = np.zeros((num_episode, 200, 5))
    PD = np.zeros((num_episode, 200, 4))
    VD = np.zeros((num_episode, 200, 4))
    epreward_set = []
    
    for i_episode in range(num_episode):

        # reset environment and episode reward
        pos, speed, pos_dev, speed_dev = sumo_reset()
        ep_reward = 0
        step = 0
    
        #while traci.simulation.getMinExpectedNumber() > 0:
        while traci.vehicle.getIDCount() >= 5:
            action = select_action(pos, speed, pos_dev, speed_dev)
             
            #print(action)
            
            # simulationStep() is called in take_action to proceed the simulation
            pos, speed, pos_dev, speed_dev, reward = take_action(pos, speed, pos_dev, speed_dev, action)
            
            #print(reward.dtype)
            record.write(str(i_episode) + '   ' + str(step) + '   ' + str(action) + '\n') 
            record.write(str(i_episode) + '   ' + str(step) + '   ' + str(pos) + '   ' + str(speed) + '\n') 
            record.write(str(i_episode) + '   ' + str(step) + '   ' + str(pos_dev) + '   ' + str(speed_dev) + '\n')
            #record.write(str(i_episode) + '   ' + str(step) + '   ' + str(reward) + '\n')             
            
            A[i_episode, step, :] = action
            P[i_episode, step, :] = pos
            V[i_episode, step, :] = speed
            PD[i_episode, step, :] = pos_dev
            VD[i_episode, step, :] = speed_dev
            
            model.rewards.append(reward)
            ep_reward += reward
            step +=1
             
        print("episode", i_episode, "---", "episode reward", ep_reward)
        epreward_set.append(ep_reward)
        #print("hidden layer", model.affine1.weight)
        # perform backprop
        finish_episode()

    traci.close()
    epreward_set = np.array(epreward_set)
    np.save("epreward_set.npy",epreward_set)
    record.write(str(epreward_set) + '\n')
    np.save("action.npy",A)
    np.save("position.npy",P)
    np.save("speed.npy",V)
    np.save("pos_dev.npy",PD)
    np.save("speed_dev.npy",VD)
    record.close()
	
if __name__ == '__main__':
    main()	
