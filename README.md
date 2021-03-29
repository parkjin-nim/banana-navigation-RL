[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: ./figures/DQN.png "DQN"
[image3]: ./figures/DDQN.png "Double-DQN"
[image4]: ./figures/PREDDQN.png "Prioritized Experience Replay"

# Learning collecting bananas

In this project, three deep-reinforcement (TD)learning agents were trained to navigate and collect bananas in a large square game world environment provided by the [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents). Agents implemented and trained in the project include DQN, Double-DQN, and PERDDQN(Prioritized Experience Replay Double-DQN) agents. A trained agent collects bananas as below. 

![Trained Agent][image1]


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes. Below video footage shows what the reinforcement learning actually looks like quite well.



### Project Details

[DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [Double-DQN](https://arxiv.org/pdf/1509.06461.pdf), and [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)(PERDDQN) agents were implemented by the papers. The video below shows the process how it is trained!

[![Watch the video](https://i9.ytimg.com/vi/dJYvvBxebkc/mq1.jpg?sqp=COj9hoMG&rs=AOn4CLDk5cPpvhxWSforXf-GgwBJO9aR9Q)](https://youtu.be/dJYvvBxebkc)


### **DQN**

Deep-Q Network uses experience replay and fixed Q-targets. It records transition(St,At,Rt+1,St+1) in replay memory. Then it samples a random mini-batch of transitions(s,a,r,s′) from the replay memory. When optimizing MSE between Q-network prediction and Q-learning targets using the gradient descent. The moving target problem is solved by setting Q-learning targets w.r.t. old, fixed parameters w-.

```
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    Q_target = rewards + (gamma * Q_targets_next * (1 - dones))   
    Q_expected = self.qnetwork_local(states).gather(1, actions)
    loss = F.mse_loss(Q_expected, Q_target)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
```


### **Double-Q DQN**

Deep Q-Learning tends to [overestimate action values](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf). Double Q-Learning has been shown to work well in practice to help with this. It further seperates the Q-learning target into two approximators, one for choosing the next best-action the other for evaluation. Old, fixed parameters w- in DQN is reused for the evaluation. 

![Trained Agent][image3]

```
    argmax_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)             
    Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions).detach() 
    Q_target       = rewards + (gamma * Q_targets_next * (1 - dones))                    
    Q_expected     = self.qnetwork_local(states).gather(1, actions)                      
    loss = F.mse_loss(Q_expected, Q_target)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

```

### **Prioritized Experience Replay DDQN**

![Trained Agent][image4]

Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability. Based on DDQN, PERDDQN periodically choose a batch size of samples out of the buffer_size of memory and update probability for each sampled experience. The sample probabilities are then used to calculate positive normalized weights, called 'Importance Sampling' weights. The cost function is wmse to down-weight and correct the bias imposed by the prioritiezed sampling.

```
    argmax_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
    Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions).detach()
    Q_target = rewards + (gamma * Q_targets_next * (1 - dones))
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    new_beta = self.get_beta()
    weights = self.memory.get_weights(beta=new_beta)
    td_errors   = Q_target - Q_expected
    self.memory.update_priorities(td_errors)
    loss = self.weighted_mse_loss(Q_expected, Q_target, weights)
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     
```

--


### Getting Started

1. Note that the environment Banana is slightly different from the one in the Unity ML-agent. Download the following Udacidy's modified version of environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Clone the the repository


### Instructions

1. Run the `jupyter notebook` 

2. Follow and execute the cells in `Navigation.ipynb` to test agents one by one in sequence.

3. Note that this notebook was run on macOS. `os.environ['KMP_DUPLICATE_LIB_OK']='True'` was set to fix a display library collision at the first cell on **4.It's your turn section**.
