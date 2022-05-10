import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from tqdm import tqdm
import keyboard
from math import pi

from utils import args, delete_these, get_rolling_average, \
    duration, reset_start_time, remove_folder, make_folder, plot_wins, plot_losses, plot_rewards, \
    save_pred_prey, load_pred_prey, delete_with_name
from pred_prey_env import PredPreyEnv
from how_to_play import episode, hand_episodes
from model import Agent



env_gui = PredPreyEnv(args, GUI = True)


class Trainer():
    def __init__(
            self, args = args, training_agent = "both", play_by_hand = False,
            save_folder = "default", 
            pred_load_folder = None, pred_load_name = "last",
            prey_load_folder = None, prey_load_name = "last",
            restart_if = {"pred" : {300 : .4}},
            done_if =    {"pred" : {200 : .99}}):
        
        self.args = args
        self.training_agent = training_agent
        self.attempts = 0
        self.start_pred_condition = self.args.pred_condition
        self.start_prey_condition = self.args.prey_condition
        self.save_folder = save_folder
        self.pred_load_folder = pred_load_folder; self.pred_load_name = pred_load_name
        self.prey_load_folder = prey_load_folder; self.prey_load_name = prey_load_name
        self.restart_if = restart_if; self.done_if = done_if
        
        self.env = PredPreyEnv(args, GUI = False)
        
        self.pred_episodes, self.prey_episodes = None, None
        self.restart()
        if(play_by_hand):
            self.pred_episodes, self.prey_episodes = hand_episodes(
                self.get_GUI(), self.pred, self.prey,
                "by_hand", self.prey_condition)
            
    def get_GUI(self):
        global env_gui
        env_gui.change(self.args, True)
        return(env_gui)
    
    def restart(self):
        reset_start_time()
        remove_folder(self.save_folder)
        make_folder(self.save_folder)
        self.attempts += 1
        self.e = 0
        self.pred = Agent()
        self.prey = Agent()
        if(self.pred_load_folder != None):
            self.pred, self.prey = load_pred_prey(
                self.pred, self.prey, suf = self.pred_load_name, folder = self.pred_load_folder,
                load = "pred")
        if(self.prey_load_folder != None):
            self.pred, self.prey = load_pred_prey(
                self.pred, self.prey, suf = self.prey_load_name, folder = self.prey_load_folder,
                load = "prey")
        if(self.pred_episodes != None and self.prey_episodes != None):
            self.pred.episodes = self.pred_episodes
            self.prey.episodes = self.prey_episodes
        save_pred_prey(self.pred, self.prey, save = "both", suf = self.e, folder = self.save_folder)
        self.args.pred_condition = self.start_pred_condition
        self.args.prey_condition = self.start_prey_condition
        self.wins = []; self.wins_rolled = []
        self.pred_losses = np.array([[None]*5])
        self.prey_losses = np.array([[None]*5])
      
    def close(self):
        self.env.close(forever = True)
        

      
    def one_episode(self, push = True, GUI = False):        
        if(GUI == False): GUI = keyboard.is_pressed('q') 
        if(GUI): env = self.get_GUI()
        else:    env = self.env
        
        pred_win, rewards = \
            episode(env, self.pred, self.prey, push)
        if(keyboard.is_pressed('q') ): plot_rewards(rewards)
            
        return(int(pred_win))


    def epoch(self, episodes_per_epoch = 3, batch_size = 32, iterations = 4):
        for _ in range(episodes_per_epoch):
            win = self.one_episode()
            self.wins.append(win)
            self.wins_rolled.append(get_rolling_average(self.wins))
      
        if(type(self.args.pred_condition) in [int, float]):
            self.args.pred_condition *= .99
        if(type(self.args.prey_condition) in [int, float]):
            self.args.prey_condition *= .99
        
        if(self.training_agent in ["pred", "both"]):
            pred_losses = self.pred.learn(batch_size = batch_size, iterations = iterations)
        else: pred_losses = np.array([[None]*5]*iterations)
        self.pred_losses = np.concatenate([self.pred_losses, pred_losses])
        
        if(self.training_agent in ["prey", "both"]):
            prey_losses = self.prey.learn(batch_size = batch_size, iterations = iterations)
        else: prey_losses = np.array([[None]*5]*iterations)
        if(iterations == 1):  pred_losses = np.expand_dims(pred_losses,0); prey_losses = np.expand_dims(prey_losses,0)
        self.pred_losses = np.concatenate([self.pred_losses, pred_losses])
        self.prey_losses = np.concatenate([self.prey_losses, prey_losses])
    
        if(keyboard.is_pressed('q') ): 
            plot_losses(self.pred_losses, True, too_long = 300)
            plot_losses(self.prey_losses, False, too_long = 300)


    def restart_or_done(self):
        
        restart = False
        for agent in self.restart_if.keys():
            for epochs in self.restart_if[agent].keys():
                    if self.e > epochs:
                        pred_wins = self.wins_rolled[-1]
                        if((agent == "pred" and pred_wins < self.restart_if[agent][epochs]) or
                           (agent == "prey" and pred_wins > self.restart_if[agent][epochs])):
                            restart = True
                            
        done = False
        for agent in self.done_if.keys():
            for epochs in self.done_if[agent].keys():
                    if self.e > epochs:
                        pred_wins = self.wins_rolled[-1]
                        if((agent == "pred" and pred_wins >= self.done_if[agent][epochs]) or
                           (agent == "prey" and pred_wins <= self.done_if[agent][epochs])):
                            done = True
        return(restart, done)
                        
        

    def train(
            self, 
            max_epochs = 1000, how_often_to_show_and_save = 25):
        
        self.pred.train(); self.prey.train()
        while(self.e < max_epochs):
            self.e += 1
            if(self.e % 5 == 0):  
                print("\nEpoch {}, {} attempt(s). {}.".format(self.e, self.attempts, duration()))
                print("Predator condition: {}. Prey condition: {}.".format(
                    self.args.pred_condition, self.args.prey_condition))
            self.epoch()
            if(self.e % how_often_to_show_and_save == 0): 
                plot_wins(self.wins_rolled, name = "wins_{}".format(str(self.e).zfill(5)), folder = self.save_folder)
                plot_losses(self.pred_losses, True, too_long = 300)
                plot_losses(self.prey_losses, False, too_long = 300)
                save_pred_prey(self.pred, self.prey, save = "pred", suf = self.e, folder = self.save_folder)
                save_pred_prey(self.pred, self.prey, save = "prey", suf = self.e, folder = self.save_folder)
                
            restart, done = self.restart_or_done()
            if(restart):
                print("This isn't working. Starting again!")
                delete_these(True, self.pred, self.prey, self.wins, 
                   self.wins_rolled, self.pred_losses, self.prey_losses)
                self.restart()
            if(done or self.e >= max_epochs):
                print("\n\nFinished!\n\n")
                print("\n\nPredator condition: {}. Prey condition: {}.\n".format(
                    self.args.pred_condition, self.args.prey_condition))
                save_pred_prey(self.pred, self.prey, save = "both", suf = "last", folder = self.save_folder)
                delete_with_name("wins", folder = self.save_folder, subfolder = "plots")
                plot_wins(self.wins_rolled, name = "wins_last", folder = self.save_folder)
                plot_losses(self.pred_losses, True, too_long = None, name = "losses", folder = self.save_folder)
                plot_losses(self.prey_losses, False, too_long = None, name = "losses", folder = self.save_folder)
                break
    
    def test(self, size = 100):
        self.pred.eval(); self.prey.eval()
        pred_wins = 0
        for i in tqdm(range(size)):
            w = self.one_episode(push = False, GUI = True)
            pred_wins += w
        print("Predator wins {} out of {} games ({}%).".format(pred_wins, size, round(100*(pred_wins/size))))
