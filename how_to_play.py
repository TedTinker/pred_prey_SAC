import torch

from utils import get_input, plot_rewards, add_discount

def episode(env, pred_brain, prey_brain, push = True):
    
    obs_list = env.reset()  
    done = False
    while(done == False):
        with torch.no_grad():
            obs_list, _, done, pred_win = env.step(obs_list, pred_brain, prey_brain, push)
    rewards = {
        "pred" : [[pushing[4] for pushing in agent.to_push] for (agent, _) in env.dead_agents if agent.predator],
        "prey" : [[pushing[4] for pushing in agent.to_push] for (agent, _) in env.dead_agents if not agent.predator]}
    env.close()

    return(pred_win, rewards)







# How to play one move by hand.
def move_by_hand(env, agent_name):
    print("Observations:")
    if(agent_name == "pred"):    agent, speed, energy = env.pred, env.pred_spe, env.pred_energy
    else:                        agent, speed, energy = env.prey, env.prey_spe, env.prey_energy
    env.render(agent_name)
    print("Speed: {}, Energy: {}".format(speed, energy))
    action = None
    while(action == None):
        yaw   = input("\nChange in yaw? Min {}, max {}.\n".format(-1, 1))
        if(yaw == ""): yaw = 0
        speed = input("\nChange in speed? Min {}, max {}.\n".format(-1, 1))
        if(speed == ""): speed = 1
        try:
            yaw = float(yaw); speed = float(speed)
            assert(yaw >= -1 and yaw <= 1 and \
                   speed >= -1 and speed <= 1)
            action = torch.tensor([yaw, speed])
        except: 
            print("'{}, {}' not a valid move. Try again.".format(yaw, speed))
    return(action)
    













# How to play a whole episode by hand.
def hand_episodes(env, pred, prey, energy = None, pred_condition = None, prey_condition = None):
    play = get_input("Play a game by hand?", ["y", "n"], default = "n")
    if(play == "n"): return(None, None)
    while(play == "y"):
        if(energy == None):
            energy = int(get_input("Energy?", ["2000", "3000", "4000"], default = "3000"))
        min_dif = get_input("Minimum pred-difficulty?", [str(i) for i in range(101)])
        max_dif = get_input("Maximum pred-difficulty?", [str(i) for i in range(101)])
        if(pred_condition == None):
            pred_condition = get_input("Predator condition?", ["by_hand", "pin", "random", "none"], default = 1)
        if(prey_condition == None):
            prey_condition = get_input("Prey condition?", ["by_hand", "pin", "random", "none"], default = 2)
        to_push_pred, to_push_prey, pred_win, rewards = episode(
            env, pred, prey, min_dif = int(min_dif), max_dif = int(max_dif), energy = energy, 
            pred_condition = pred_condition, prey_condition = prey_condition)
        print("\nGAME OVER! {}\n".format("Predator wins." if pred_win else "Prey wins."))
        plot_rewards(rewards)
        max_repeats = 32
        repeat = get_input("Remember this how many times?", [str(i) for i in range(1,max_repeats+1)])
        for _ in range(int(repeat)):
            for i in range(len(to_push_pred)):
                pred.episodes.push(to_push_pred[i])
                prey.episodes.push(to_push_prey[i])
        play = get_input("Play another game by hand?", ["y", "n"], default = "n")
    return(pred.episodes, prey.episodes)