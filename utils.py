# Issues:
    # Ducks can't push flowers

# Parameters for an arena.
import argparse
from math import pi

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena_name',                 type=str,   default = "empty_arena")    
    parser.add_argument('--flowers',                    type=int,   default = 1)
    parser.add_argument('--flower_size',                type=float, default = .7)
    parser.add_argument('--lr',                         type=float, default = .001)
    parser.add_argument('--alpha',                      type=float, default = None)
    parser.add_argument('--gamma',                      type=float, default = .99)
    parser.add_argument('--eta',                        type=float, default = .9)
    parser.add_argument('--tau',                        type=float, default = 1e-2)
    parser.add_argument('--d',                          type=int,   default = 2)
    

    parser.add_argument('--pred_condition',                         default = 1)
    parser.add_argument('--pred_start',                 type=int,   default = 1)
    parser.add_argument('--pred_size',                  type=float, default = 10)
    parser.add_argument('--pred_max_age',               type=int,   default = 500)
    parser.add_argument('--pred_energy',                type=float, default = 3000)
    parser.add_argument('--pred_energy_per_speed',      type=float, default = 1)
    parser.add_argument('--pred_energy_per_degree',     type=float, default = 1)
    parser.add_argument('--pred_energy_from_prey',      type=float, default = 1000)
    parser.add_argument('--pred_image_size',            type=int,   default = 16)
    parser.add_argument('--pred_min_speed',             type=float, default = 10)
    parser.add_argument('--pred_max_speed',             type=float, default = 50)
    parser.add_argument('--pred_max_yaw_change',        type=float, default = pi/2)
    parser.add_argument('--pred_reward_agent_col',      type=float, default = 1)
    parser.add_argument('--pred_reward_flower_col',     type=float, default = 0)
    parser.add_argument('--pred_reward_wall_col',       type=float, default = -1)
    parser.add_argument('--pred_reward_agent_closer',   type=float, default = 10)
    parser.add_argument('--pred_reward_flower_closer',  type=float, default = 0)
    
    parser.add_argument('--prey_condition',                         default = "pin")
    parser.add_argument('--prey_start',                 type=int,   default = 1)
    parser.add_argument('--prey_size',                  type=float, default = 10)
    parser.add_argument('--prey_max_age',               type=int,   default = 500)
    parser.add_argument('--prey_energy',                type=float, default = 3000)
    parser.add_argument('--prey_energy_per_speed',      type=float, default = 1)
    parser.add_argument('--prey_energy_per_degree',     type=float, default = 1)
    parser.add_argument('--prey_energy_from_flower',    type=float, default = 1000)
    parser.add_argument('--prey_image_size',            type=int,   default = 16)
    parser.add_argument('--prey_min_speed',             type=float, default = 10)
    parser.add_argument('--prey_max_speed',             type=float, default = 50)
    parser.add_argument('--prey_max_yaw_change',        type=float, default = pi/2)
    parser.add_argument('--prey_reward_agent_col',      type=float, default = -1)
    parser.add_argument('--prey_reward_flower_col',     type=float, default = 1)
    parser.add_argument('--prey_reward_wall_col',       type=float, default = -1)
    parser.add_argument('--prey_reward_agent_closer',   type=float, default = -10)
    parser.add_argument('--prey_reward_flower_closer',  type=float, default = 10)
    
    return parser.parse_args()

args = get_args()

def change_args(**kwargs):
    args = get_args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return(args)

def get_arg(args, predator, arg):
    if(predator): arg = "pred_" + arg
    else:         arg = "prey_" + arg
    return(getattr(args, arg))



### A few utilities
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal



import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



# Track seconds starting right now. 
import datetime
start_time = datetime.datetime.now()
def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)




# Monitor GPU memory.
def get_free_mem(string = ""):
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("\n{}: {}.\n".format(string, f))

# Remove from GPU memory.
def delete_these(verbose = False, *args):
    if(verbose): get_free_mem("Before deleting")
    del args
    torch.cuda.empty_cache()
    if(verbose): get_free_mem("After deleting")
  
  
  
# How to get an input from keyboard.
def get_input(string, okay, default = None):
    if(type(okay) == list): okay = {i+1:okay[i] for i in range(len(okay))}
    while(True):
        inp = input("\n{}\n{}\n".format(string, okay))
        if(inp == "" and default != None): 
          if(type(default) == int): 
              print("Default: {}.\n".format(okay[default]))
              return(okay[default])
          else:     
              print("Default: {}.\n".format(default))                
              return(default)
        try: 
            if(inp in okay.values()): return(inp)
            else:                     return(okay[int(inp)])
        except: 
            print("\n'{}' isn't a good answer.".format(inp))
            
            
            
# How to get rolling average.
def get_rolling_average(wins, roll = 100):
    if(len(wins) < roll):
        return(sum(wins)/len(wins))
    return(sum(wins[-roll:])/roll)       


# How to add discount to a list.
def add_discount(rewards, GAMMA = .99):
    rewards.reverse()
    discounted = []; d = 0
    for r in rewards:
        if(r != 0): d = r
        discounted.insert(0, d)
        d *= GAMMA
    return(discounted)
            
            
            
            
            
            
# How to save plots.
import matplotlib.pyplot as plt
import os
import shutil

def remove_folder(folder):
    files = os.listdir("saves")
    if(folder not in files): return
    shutil.rmtree("saves/" + folder)

def make_folder(folder):
    if(folder == None): return
    files = os.listdir("saves")
    if(folder in files): return
    os.mkdir("saves/"+folder)
    os.mkdir("saves/"+folder+"/plots")
    os.mkdir("saves/"+folder+"/preds")
    os.mkdir("saves/"+folder+"/preys")

def save_plot(name, folder = "default"):
    make_folder(folder)
    plt.savefig("saves/"+folder+"/plots/"+name+".png")
  
def delete_with_name(name, folder = "default", subfolder = "plots"):
    files = os.listdir("saves/{}/{}".format(folder, subfolder))
    for file in files:
        if(file.startswith(name)):
            os.remove("saves/{}/{}/{}".format(folder, subfolder, file))
  
  
  
  
  
  
# How to plot an episode's rewards.
def plot_rewards(rewards, name = None, folder = "default"):
    total_length = max([len(rewards["pred"][i]) for i in range(len(rewards["pred"]))] + 
                        [len(rewards["prey"][i]) for i in range(len(rewards["prey"]))])
    x = [i for i in range(1, total_length + 1)]
    plt.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    for pred_reward in rewards["pred"]:
        plt.plot(x[:len(pred_reward)], pred_reward, color = "lightcoral", label = "Predator") # Predators
    for prey_reward in rewards["prey"]:
        plt.plot(x[:len(prey_reward)], prey_reward, color = "turquoise", label = "Prey")  # Prey
    plt.legend(loc = 'upper left')
    plt.title("Rewards")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
  
  
# How to plot losses.
def plot_losses(losses, pred = True, too_long = None, name = None, folder = "default"):
    
    color_1 = "red" if pred else "blue"
    color_2 = "lightcoral" if pred else "turquoise"
    title = "Predator Losses" if pred else "Prey Losses"
    if(name != None):
        name = "pred_" + name if pred else "prey_" + name
    
    length = len(losses)
    x = [i for i in range(1, length + 1)]
    if(too_long != None and length > too_long):
        x = x[-too_long:]; losses = losses[-too_long:]
    actor_x  = [x_ for i, x_ in enumerate(x) if losses[i][0] != None]
    actor_y = [l[0] for l in losses if l[0] != None]
    critic_x = [x_ for i, x_ in enumerate(x) if losses[i][1] != None]
    critic_1_y = [l[1] for l in losses if l[1] != None]
    critic_2_y = [l[2] for l in losses if l[2] != None]
    
    if(len(critic_x) >= 1):
        fig, ax1 = plt.subplots() 
        ax2 = ax1.twinx() 
        ax1.plot(actor_x, actor_y, color = color_1, label = "Actor")
        ax2.plot(critic_x, critic_1_y, color = color_2, linestyle = "--", label = "Critic")
        ax2.plot(critic_x, critic_2_y, color = color_2, linestyle = ":", label = "Critic")
        ax1.legend(loc = 'upper left')
        ax2.legend(loc = 'lower left')
        plt.title(title)
        plt.xlabel("Training iterations")
        ax1.set_ylabel("Actor losses")
        ax2.set_ylabel("Critic losses")
        if(name!=None): save_plot(name, folder)
        plt.show()
        plt.close()
  
  
  
  
  
# How to plot predator victory-rates.
def plot_wins(wins, max_len = None, name = None, folder = "default"):
    total_length = len(wins)
    x = [i for i in range(1, len(wins)+1)]
    if(max_len != None and total_length > max_len):
        x = x[-max_len:]
        wins = wins[-max_len:]
    plt.plot(x, wins, color = "gray")
    plt.ylim([0, 1])
    plt.title("Predator win-rates")
    plt.xlabel("Episodes")
    plt.ylabel("Predator win-rate")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
  
  
  

  
  
  
  
  
  
# How to save/load pred/prey

def save_pred_prey(pred, prey, save = "both", suf = "", folder = None):
    if(folder == None): return
    if(type(suf) == int): suf = str(suf).zfill(5)
    if(save == "both" or save == "pred"):
        torch.save(pred.state_dict(), "saves/" + folder + "/preds/pred_{}.pt".format(suf))
    if(save == "both" or save == "prey"):
        torch.save(prey.state_dict(), "saves/" + folder + "/preys/prey_{}.pt".format(suf))

def load_pred_prey(pred, prey, load = "both", suf = "last", folder = "default"):
    if(type(suf) == int): suf = str(suf).zfill(5)
    if(load == "both" or load == "pred"):
        pred.load_state_dict(torch.load("saves/" + folder + "/preds/pred_{}.pt".format(suf)))
    if(load == "both" or load == "prey"):
        prey.load_state_dict(torch.load("saves/" + folder + "/preys/prey_{}.pt".format(suf)))
    return(pred, prey)