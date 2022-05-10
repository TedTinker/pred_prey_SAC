import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim
from torchinfo import summary as torch_summary

import random
import numpy as np
import scipy.stats 
sf = scipy.stats.norm.sf

from utils import args
from buffer import RecurrentReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

hidden_size = 128




class Transitioner(nn.Module):
    
    def __init__(self, rgbd_input = (16,16,4)):
        super(Transitioner, self).__init__()
        
        example = torch.zeros(rgbd_input).unsqueeze(0).permute(0,3,1,2)

        self.cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4, 
                out_channels = 8,
                kernel_size = (3,3),
                padding = (1,1)
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1))
        )
        
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1] + 2 # Plus speed, energy
        
        self.lstm = nn.LSTM(
            input_size = quantity,
            hidden_size = hidden_size,
            batch_first = True)
        
        self.encode = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU())
        
        self.decode = nn.Sequential(
            nn.Linear(64+2, hidden_size),
            nn.LeakyReLU())
        
        self.mu = nn.Linear(hidden_size, 16*16*4 + 2)
        self.log_std_linear = nn.Linear(hidden_size, 16*16*4 + 2)
        
        self.cnn.apply(init_weights)
        self.lstm.apply(init_weights)
        self.encode.apply(init_weights)
        self.decode.apply(init_weights)
        
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        
        
    def just_encode(self, image, speed, energy, hidden = None):
        if(len(image.shape) == 4):  sequence = False
        else:                       sequence = True
        image = torch.permute(image, (0,1,-1,2,3) if sequence else (0, -1, 1, 2))
        if(sequence): 
            batch_size = image.shape[0]
            image = image.reshape(image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4])
        image = self.cnn(image).flatten(1)
        if(sequence): 
            image = image.reshape(batch_size, image.shape[0]//batch_size, image.shape[1])
        x = torch.cat([image, speed, energy], -1)
        if(not sequence):
            x = x.view(x.shape[0], 1, x.shape[1])
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(x)
        else:               x, hidden = self.lstm(x, (hidden[0], hidden[1]))
        if(not sequence):
            x = x.view(x.shape[0], x.shape[-1])
        encoding = self.encode(x)
        return(encoding, hidden)
        
    def forward(self, image, speed, energy, action, hidden = None):
        encoding, hidden = self.just_encode(image, speed, energy, hidden)
        x = torch.cat((encoding, action), dim=-1)
        decoding = self.decode(x)
        mu = self.mu(decoding)
        log_std = self.log_std_linear(decoding)
        return(mu, log_std, hidden)
    
    def get_next_state(self, image, speed, energy, action, hidden = None):
        mu, log_std, hidden = self.forward(image, speed, energy, action, hidden)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        next_state = torch.tanh(mu + e * std).cpu()
        return(next_state, hidden)
        
    def probability(self, 
                    next_image, next_speed, next_energy, 
                    image, speed, energy, action, hidden = None):
        mu, log_std, hidden = self.forward(image, speed, energy, action, hidden)
        
        next_state = torch.cat([
            next_image.flatten(2), 
            next_speed, 
            next_energy], -1)
        
        std = log_std.exp()
        z = torch.abs((next_state - mu)/std).detach().numpy()
        p = sf(z)*args.eta
        p = p[:,:,0] * p[:,:,1] * p[:,:,2]
        p = torch.tensor(p)
                
        return(p.unsqueeze(-1))





class Actor(nn.Module):
    
    def __init__(
            self, 
            log_std_min=-20, 
            log_std_max=2):
        
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU())
        self.mu = nn.Linear(hidden_size, 2)
        self.log_std_linear = nn.Linear(hidden_size, 2)

    def forward(self, encode):
        x = self.lin(encode)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, encode, epsilon=1e-6):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob
        
    def get_action(self, encode):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]



class Critic(nn.Module):

    def __init__(self):
        
        super(Critic, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(64+2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, encode, action):
        x = torch.cat((encode, action), dim=-1)
        x = self.lin(x)
        return x
    
    
    


    
    



class Agent():
    
    def __init__(
            self, action_prior="uniform"):
        
        self.steps = 0
        
        self.target_entropy = -2  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=args.lr) 
        self._action_prior = action_prior
        
        self.transitioner = Transitioner()
        self.trans_optimizer = optim.Adam(self.transitioner.parameters(), lr=args.lr)     
                   
        self.actor = Actor().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)     
        
        self.critic1 = Critic().to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.lr, weight_decay=0)
        self.critic1_target = Critic().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic().to(device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.lr, weight_decay=0) 
        self.critic2_target = Critic().to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.memory = RecurrentReplayBuffer(max_episode_len = 10000)
        
        describe_agent(self)
        
    def step(self, image, speed, energy, action, reward, next_state, done, step):
        self.memory.push(image, speed, energy, action, reward, next_state, done, done)
        if self.memory.num_episodes > args.batch_size:
            experiences = self.memory.sample()
            trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss = \
                self.learn(step, experiences, args.gamma)
            return(trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss)
        return(None, None, None, None, None)
            
    def act(self, image, speed, energy, hidden = None, condition = 0):
        speed = torch.tensor(speed).float().to(device).view(1,1)
        energy = torch.tensor(energy).float().to(device).view(1,1)
        encoded, hidden = self.transitioner.just_encode(image.unsqueeze(0), speed, energy, hidden)
        action = self.actor.get_action(encoded).detach()
        
        if(condition == "pin"):
            action = torch.tensor([-1,-1])
        elif(condition == "random" or random.uniform(0,1) < condition):
            action = torch.tensor([random.uniform(-1,1), random.uniform(-1,1)])
            
        return action, hidden

    def learn(self, batch_size, iterations):
        
        if(iterations != 1):
            losses = []
            for i in range(iterations):
                losses.append(self.learn(batch_size, 1))
            return(np.concatenate(losses))
        
        self.steps += 1

        try:
            images, speeds, energies, actions, rewards, dones, _ = self.memory.sample()
        except:
            return(np.array([[None, None, None, None, None]]))
        
        # Train transitioner
        pred_next_state, _ = self.transitioner.get_next_state(images[:,:-1], speeds[:,:-1], energies[:,:-1], actions)
        trans_loss = F.mse_loss(pred_next_state, 
                                torch.cat([
                                    images[:,1:].flatten(2), 
                                    speeds[:,1:], 
                                    energies[:,1:]], -1))
        self.trans_optimizer.zero_grad()
        trans_loss.backward()
        self.trans_optimizer.step()
        
        encoded, _ = self.transitioner.just_encode(images[:,:-1], speeds[:,:-1], energies[:,:-1])
        encoded = encoded.detach()
        next_encoded, _ = self.transitioner.just_encode(images[:,1:], speeds[:,1:], energies[:,1:])
        next_encoded = next_encoded.detach()
        
        # Update rewards with curiosity
        curiosity = self.transitioner.probability(images[:,1:], speeds[:,1:], energies[:,1:],
                                                  images[:,:-1], speeds[:,:-1], energies[:,:-1], actions)
        rewards += curiosity
        
        # Train critics
        next_action, log_pis_next = self.actor.evaluate(next_encoded)
        # log_pis_next has size 2, because it's action-space. What to do?
        Q_target1_next = self.critic1_target(next_encoded.to(device), next_action.to(device))
        Q_target2_next = self.critic2_target(next_encoded.to(device), next_action.to(device))
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if args.alpha == None:
            Q_targets = rewards.cpu() + (args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (args.gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - args.alpha * log_pis_next.squeeze(0).cpu()))
        
        Q_1 = self.critic1(encoded, actions).cpu()
        print()
        print(Q_1.shape, Q_targets.shape)
        print()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        Q_2 = self.critic2(encoded, actions).cpu()
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Train actor
        if self.steps % args.d == 0:
            if args.alpha == None:
                self.alpha = torch.exp(self.log_alpha)
                actions_pred, log_pis = self.actor.evaluate(encoded)
                alpha_loss = -(self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                Q = torch.min(
                    self.critic1(encoded, actions_pred), 
                    self.critic2(encoded, actions_pred))
                actor_loss = (self.alpha * log_pis.squeeze(0).cpu() - Q.cpu() - policy_prior_log_probs).mean()
            
            else:
                alpha_loss = None
                actions_pred, log_pis = self.actor.evaluate(encoded)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0
                Q = torch.min(
                    self.critic1(encoded, actions_pred.squeeze(0)), 
                    self.critic2(encoded, actions_pred.squeeze(0)))
                actor_loss = (args.alpha * log_pis.squeeze(0).cpu() - Q.cpu()- policy_prior_log_probs ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic1, self.critic1_target, args.tau)
            self.soft_update(self.critic2, self.critic2_target, args.tau)
            
        else:
            alpha_loss = None
            actor_loss = None
        
        if(trans_loss != None): trans_loss = trans_loss.item()
        if(alpha_loss != None): alpha_loss = alpha_loss.item()
        if(actor_loss != None): actor_loss = actor_loss.item()
        if(critic1_loss != None): critic1_loss = critic1_loss.item()
        if(critic2_loss != None): critic2_loss = critic2_loss.item()

        return(np.array([[trans_loss, alpha_loss, actor_loss, critic1_loss, critic2_loss]]))
                     
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def state_dict(self):
        return(
            self.transitioner.state_dict(),
            self.actor.state_dict(),
            self.critic1.state_dict(),
            self.critic1_target.state_dict(),
            self.critic2.state_dict(),
            self.critic2_target.state_dict())

    def load_state_dict(self, state_dict):
        self.transitioner.load_state_dict(state_dict[0])
        self.actor.load_state_dict(state_dict[1])
        self.critic1.load_state_dict(state_dict[2])
        self.critic1_target.load_state_dict(state_dict[3])
        self.critic2.load_state_dict(state_dict[4])
        self.critic2_target.load_state_dict(state_dict[4])
        self.memory = RecurrentReplayBuffer()

    def eval(self):
        self.transitioner.eval()
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def train(self):
        self.transitioner.train()
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()



def describe_agent(agent):
    print("\n\n")
    print(agent.transitioner)
    print()
    print(torch_summary(agent.transitioner, 
                        ((1, 16, 16, 4), # Image
                         (1, 1),         # Speed
                         (1, 1),         # Energy
                         (1, 2))))       # Action
    
    print("\n\n")
    print(agent.actor)
    print()
    print(torch_summary(agent.actor, (1, 64)))
    
    print("\n\n")
    print(agent.critic1)
    print()
    print(torch_summary(agent.critic1, ((1, 64),(1,2))))
    
if __name__ == "__main__":
    agent = Agent()
    describe_agent(agent)