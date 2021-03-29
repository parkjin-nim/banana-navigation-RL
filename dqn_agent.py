import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device) # Q(S,A;w)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) # fixated target Q(S',a;w-)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) # Replay memory

        self.t_step = 0                                # count time step 
    
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done) # Save experience in replay memory
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:            # Learn every UPDATE_EVERY time steps.
            if len(self.memory) > BATCH_SIZE:          # If enough samples available, get a batch amount and learn
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()                      #turn eval. mode on 
        with torch.no_grad():                           #turn grad ops off
            action_values = self.qnetwork_local(state) 
        self.qnetwork_local.train()                     #turn train mode back on

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_target = rewards + (gamma * Q_targets_next * (1 - dones))   
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    
    
class DQAgent():

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device) # Q(S,A;w)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) # fixated target Q(S',a;w-)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed) # Replay memory

        self.t_step = 0                                # count time step 
    
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done) # Save experience in replay memory
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:            # Learn every UPDATE_EVERY time steps.
            if len(self.memory) > BATCH_SIZE:          # If enough samples available, get a batch amount and learn
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()                      #turn eval. mode on 
        with torch.no_grad():                           #turn grad ops off
            action_values = self.qnetwork_local(state) 
        self.qnetwork_local.train()                     #turn train mode back on

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        argmax_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)  #agmax_action = argmax Q(S',A; w)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions).detach() 
        Q_target = rewards + (gamma * Q_targets_next * (1 - dones))               # Q(S',argmaxaction; w-)
        Q_expected = self.qnetwork_local(states).gather(1, actions)               # Q(S,A; w)
        loss = F.mse_loss(Q_expected, Q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


class PERDQAgent():
    def __init__(self, state_size, action_size, seed, alpha=0.6, max_t=1000, initial_beta=0.4):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device) # Q(S,A;w)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device) # fixated target Q(S',a;w-)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = PRBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, alpha)
        self.alpha = alpha
        self.initial_beta = initial_beta
        self.max_t = max_t
        
        self.t_step = 0                                # count time step 
    
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done) # Save experience in replay memory
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:            # Learn every UPDATE_EVERY time steps.
            if len(self.memory) > BATCH_SIZE:          # If enough samples available, get a batch amount and learn
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()                      #turn eval. mode on 
        with torch.no_grad():                           #turn grad ops off
            action_values = self.qnetwork_local(state) 
        self.qnetwork_local.train()                     #turn train mode back on

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def weighted_mse_loss(self, expected, target, weights):

        out = (expected-target)**2
        out = out * weights.expand_as(out)
        loss = out.mean(0)  # or sum over whatever dimensions
        return loss
    
    def get_beta(self):
        """
        Controls how much prioritization to apply. 
        They argue that training is highly unstable at the beginning, 
        and that importance sampling corrections matter more near the end of training.         
        """
        inc_frac = min(self.t_step / self.max_t, 1.0)   
        new_beta = self.initial_beta + inc_frac * (1. - self.initial_beta)
        return new_beta
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def learn(self, experiences, gamma):
        
        # get a batch size of samples from buffer, using random.choice n choose r w/ p
        states, actions, rewards, next_states, dones = experiences 

        # based on Double-DQN, get td_errors
        argmax_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions).detach()
        Q_target = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # get weights using a new annealed beta
        new_beta = self.get_beta()
        weights = self.memory.get_weights(beta=new_beta)
        
        # update all experiences 'priority_i = |td_errors_i| + eta' in memory
        td_errors   = Q_target - Q_expected
        self.memory.update_priorities(td_errors)

        # compute wmse loss function
        loss = self.weighted_mse_loss(Q_expected, Q_target, weights)
        
        # GD to compute delta w = LR * weight_i * td_error_i * ∇w Q(Si,Ai;w)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network using interpolation factor TAU
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


    
class PRBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """
        self.priorities of buffer_size stores ((abs(f_tderr) + self.eta) ** self.alpha) for each experience
        self.sample_indexes stores a batch size of samples index to the buffer chosen out of the current buffer_size
        self.cum_priorities keeps track of the sum of all experience priorities in the current buffer_size
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.seed = random.seed(seed)
        self.alpha = alpha  
        self.priorities = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.cum_priorities = 0.
        self.eta = 1e-6
        self.sample_indexes = []
        self.max_priority = 1.**self.alpha
    
    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        if len(self.priorities) >= self.buffer_size:
            self.cum_priorities -= self.priorities[0]
            
        self.priorities.append(self.max_priority) 
        self.cum_priorities += self.priorities[-1]
    
    def sample(self):
        na_probs = None
        if self.cum_priorities:
            na_probs = np.array(self.priorities)/self.cum_priorities

        # out of the population, choose a batch size of samples, using prob.
        popul = len(self.memory)
        self.sample_indexes = np.random.choice(popul, size=min(popul, self.batch_size), p=na_probs)


        experiences = [self.memory[i] for i in self.sample_indexes]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def get_weights(self, beta):

        # calculate weight for each experience of a batch, ((N x P(i)) ^ -β)
        N = len(self.memory)
        weights = [(N * self.priorities[i] / self.cum_priorities) ** -beta for i in self.sample_indexes]

        # normalize weights to 0~1
        max_weight = (N * min(self.priorities) / self.cum_priorities) ** -beta
        weights = [w / max_weight for w in weights]
            
        return torch.tensor(weights, device=device, dtype=torch.float).reshape(-1, 1)

    
    def update_priorities(self, td_errors):
        """
        Priority pi is |δi| + ε.  s.t. |δi|:positive, ε:selection starvation proof
        Store pi^α                s.t  α=0:random selection α=1:only priority selection
        """
        for i, f_tderr in zip(self.sample_indexes, td_errors):
            f_tderr = float(f_tderr)
            self.cum_priorities -= self.priorities[i]
            
            self.priorities[i] = ((abs(f_tderr) + self.eta) ** self.alpha)
            self.cum_priorities += self.priorities[i]
            
        self.max_priority = max(self.priorities)
        # done with updates, empty the sample bucket.
        self.sample_indexes = []
        
        
    def __len__(self):
        return len(self.memory)

