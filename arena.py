# What is an agent?
import torch

class Agent:
    def __init__(self, predator, p_num, energy, pos, yaw, spe):
        self.predator = predator; self.p_num = p_num; self.energy = energy
        self.pos = pos; self.yaw = yaw; self.spe = spe
        self.age = 0
        self.action = torch.tensor([0.0, 0.0])
        self.hidden = None
        self.to_push = []



# How to make physicsClients.
import pybullet as p

def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data/")
    return(physicsClient)



# Get arena from image.
import numpy as np
from math import pi, sin, cos
import cv2
from itertools import product
import random

from utils import args
from utils import get_arg

class Arena():
    def __init__(self, args = args, GUI = False):
        self.args = args
        self.arena_map = cv2.imread("arenas/" + args.arena_name + ".png")
        self.w, self.h, _ = self.arena_map.shape
        self.physicsClient = get_physics(GUI, self.w, self.h)
        self.open_spots = [(x,y) for x, y in product(range(self.w), range(self.h)) \
                          if self.arena_map[x,y].tolist() == [255, 255, 255]]
        self.used_spots = []
        self.already_constructed = False

    def start_arena(self):
        if(not self.already_constructed):
            self.wall_ids = []
            for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
                if(not (self.arena_map[loc] == [255]).all()):
                  pos = [loc[0],loc[1],.5]
                  ors = p.getQuaternionFromEuler([0,0,0])
                  cube = p.loadURDF("cube.urdf",pos,ors, useFixedBase = True, physicsClientId = self.physicsClient)
                  color = self.arena_map[loc][::-1] / 255
                  color = np.append(color, 1)
                  p.changeVisualShape(cube, -1, rgbaColor=color, physicsClientId = self.physicsClient)
                  self.wall_ids.append(cube)
                  self.already_constructed = True
    
    def make_agent(self, predator):
        pos = random.choice(self.open_spots)
        while(pos in self.used_spots):
            pos = random.choice(self.open_spots)
        self.used_spots.append(pos)
        yaw = random.uniform(0, 2*pi)
        spe = get_arg(self.args, predator, "min_speed")
        energy = get_arg(self.args, predator, "energy")
        color = [1,0,0,1] if predator else [0,0,1,1]
        file = "ted_duck.urdf"
        
        pos = (pos[0], pos[1], .5)
        orn = p.getQuaternionFromEuler([pi/2,0,yaw])
        p_num = p.loadURDF(file,pos,orn,
                           globalScaling = self.args.pred_size if predator else self.args.prey_size, 
                           physicsClientId = self.physicsClient)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        p.resetBaseVelocity(p_num, (x,y,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(p_num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        agent = Agent(predator, p_num, energy, pos, yaw, spe)
        return(agent)
    
    def make_flower(self):
        pos = random.choice(self.open_spots)
        while(pos in self.used_spots):
            pos = random.choice(self.open_spots)
        self.used_spots.append(pos)
        roll  = random.uniform(0, 2*pi)
        pitch = random.uniform(0, 2*pi)
        yaw   = random.uniform(0, 2*pi)
        file  = "ted_sphere.urdf"
        
        pos = (pos[0], pos[1], .5)
        orn = p.getQuaternionFromEuler([roll,pitch,yaw])
        color = [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),1]
        p_num = p.loadURDF(file,pos,orn,
                            globalScaling = self.args.flower_size,
                            physicsClientId = self.physicsClient)
        p.resetBaseVelocity(p_num, (0,0,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(p_num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        return(p_num)
    
    def wall_collisions(self, p_num):
        col = False
        for wall in self.wall_ids:
            if 0 < len(p.getContactPoints(p_num, wall, physicsClientId = self.physicsClient)):
              col = True
        return(col)
    
    def all_wall_collisions(self, agent_list):
        collisions = []
        for agent in agent_list:
            collisions.append(self.wall_collisions(agent.p_num))
        return(collisions)

    def agent_collisions(self, p_num_1, p_num_2):
        return(0 < len(p.getContactPoints(
            p_num_1, p_num_2, physicsClientId = self.physicsClient)))
    
    def agent_dist(self, p_num_1, p_num_2):
        pos_1,_ ,_ = self.get_pos_yaw_spe(p_num_1)
        pos_2,_ ,_ = self.get_pos_yaw_spe(p_num_2)
        x = pos_1[0] - pos_2[0]
        y = pos_1[1] - pos_2[1]
        return((x**2 + y**2)**.5)
    
    def all_agent_dists(self, agent_list):
        if(len(agent_list) == 1): return([100])
        min_dists = []
        for i, agent_1 in enumerate(agent_list):
            dists = []
            for agent_2 in [agent for j,agent in enumerate(agent_list) if j!=i]:
                dists.append(self.agent_dist(agent_1.p_num, agent_2.p_num))
            min_dists.append(min(dists))
        return(min_dists)
    
    def all_flower_dists(self, agent_list, flower_list):
        if(flower_list == []): return([100]*len(agent_list))
        min_dists = []
        for i, agent in enumerate(agent_list):
            dists = []
            for flower in flower_list:
                dists.append(self.agent_dist(agent.p_num, flower))
            minimum = min(dists) if min(dists) != 0 else .001
            min_dists.append(minimum)
        return(min_dists)
        
    
    def get_pos_yaw_spe(self, p_num):
        pos, ors = p.getBasePositionAndOrientation(p_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(p_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)



if __name__ == "__main__":
    arena = Arena()
