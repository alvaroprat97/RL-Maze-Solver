############################################################################
############################################################################
# AGENT FOR: ALVARO PRAT , ap5915
# CID: 01066209
############################################################################
############################################################################

import numpy as np
import torch
import matplotlib.pyplot as plt

class Agent:

    # Function to initialise the agent
    def __init__(self):

        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Random initial distance to goal. Used to check if we have arrived to the goal
        self.distance_to_goal = 1

        # Hyperparameter Settings
        self.delta = 0.00002 # Initial delta is small to cover the map well during exploration
        self.init_size = 100000 # Used to be a million, now 100,000 as it never gets to 100,000 steps anyways
        self.weighted_alpha = 0.7 # Weighting factor for prioritised experience replay buffer
        self.weighted_epsilon = 0.000000001 #
        self.epsilon = 1 # Initial value for epsilon
        self.batch_size = 128
        self.epsilon_min = 0.01 # Minimum epsilon during training
        #self.actions = np.arange(0,4) # discrete actions 0 ,1, 2, 3
        self.actions = np.arange(0,3) # Now, there are only 3 actions in action space
        self.episode_length = 5000 # Set the episode length (you will need to increase this)
        self.target_steps = 30 # Steps till target Q-network updates
        self.gamma = 0.9 # Bellman Discount factor
        self.exploration_steps = 2500 # SEMI EXPLORATORY STEPS at initialisation with epsilon
        self.explore = True # Let it explore initially
        self.bPumped = False # Flag to determine if we have increased our epsilon (when it gets stuck)

        # Use DQN and ReplayBuffer Instances
        self.dqn = DQN(self.gamma)
        self.ReplayBuffer = ReplayBuffer(self.batch_size,self.init_size,self.weighted_alpha,self.weighted_epsilon)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):

        # IF we are not exploring and we have not pumped up epsilon in the episode, if
        # we get stuck we increase epsilon
        if (self.explore is False) and (self.bPumped is False) and (self.epsilon < 0.9):
            if self.num_steps_taken > 4000:
                self.epsilon += 0.1
                self.epsilon = np.clip(self.epsilon,0,95)
                # print('FORCED INCREASE OF EPSILON TO {}'.format(self.epsilon))
                self.bPumped = True

        # Exit the episode if num_steps is epsiode length or if we reached the goal
        if self.num_steps_taken % self.episode_length == 0 or (self.distance_to_goal < 0.03):
            self.total_reward = 0 # RE-set reward at the beginning of the episode
            self.distance_to_goal = 1 # Dummy variable to re-initialise reward-not exact but will be re-written
            self.num_steps_taken = 0
            self.bPumped = False #Re-initialise this variable so that in the next episode we can still pump up epsilon if we get stuck again!
            return True
        else:
            return False

    # Function to decrease our epsilon depending if we are exploring or if epsilon is above its threshold
    def decrease_epsilon(self):
        if self.epsilon < self.epsilon_min:
            pass
        elif self.ReplayBuffer.counter < self.exploration_steps: # Look to see if the step counter has overflown the allowed exploration steps
            pass
        else:
            if self.explore:
                self.explore = False # Change the exploration flag so that exploration stops
                # print('Stopped exploring')
            else:
                self.increase_delta() # Dynamic delta, we increase delta in less sensitive points
                self.epsilon -= self.delta

    # Dynamic delta increase
    def increase_delta(self):
        if self.epsilon < 0.3:
            self.delta = 3e-5 # Decrease epsilon decay rate when 1//epsilon gradient is high
        if self.epsilon < 0.7:
            self.delta = 6e-5 # Increase epsilon decay rate a lot when 1/(1-epsilon) gradient is very low
        if self.epsilon < 0.8:
            self.delta = 3e-5 # Increase epsilon decay when 1/(1-epsilon) gradient is low

    # From a prefered action action_star, we select randomly an epsilon action from the action space
    def epsilon_pick(self,action_star):
        length = len(self.actions)
        p = np.ones((length))*self.epsilon/length
        p[action_star] = 1 - self.epsilon + self.epsilon/length
        return np.random.choice(self.actions,p = p)

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        discrete_action_array = self.move(discrete_action)
        continuous_action = np.array(discrete_action_array, dtype=np.float32)
        return continuous_action

    # Take an epsilon-greedy action (uses epsilon_pick equation) and prefered action is found from DQN
    def epsilon_greedy_action(self):
        # FIND ARGMAX A from predicted Q(S,A)
        prediction = self.dqn.q_network.forward(torch.tensor(self.state))
        discrete_actions = torch.argmax(prediction).item()

        # FIND DISCRETE ACTION FROM e-Greedy, when it is completely greedy, we pick epsilon star!
        discrete_action = self.epsilon_pick(discrete_actions)
        continuous_action = self._discrete_action_to_continuous(discrete_action)

        # Decrease epsilon by delta
        self.decrease_epsilon()
        self.action = discrete_action
        # For visualisation only, as when greedy we only care about next state (greedy in visualisation)
        return continuous_action #transition

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Get the epsilon greedy action
        continuous_action = self.epsilon_greedy_action()
        # Store the action; this will be used later, when storing the transition
        return continuous_action

    # Function to compute the reward. No MDP vialoations: only use next_state and distance_to_goal
    def _compute_reward(self, distance_to_goal, next_state):
        reward = 0 # Initially all instantaneous rewards are 0
        if self.state is next_state:
            reward = -0.1*distance_to_goal # Penalise hitting the wall far away from the goal state
        if self.state[0] < next_state[0]:
            reward = (1-distance_to_goal) # Make move right the prefered direction. Note by default, moving up or down will have no rewards, unless hitting a wall
        if distance_to_goal < 0.2:
            reward += 0.1 # Make rewards near the goal higher
        if distance_to_goal < 0.03:
            reward +=0.1 # Rewards at the goal are the highest (still, not too high, in order to control Q-gradients)
        return reward

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self._compute_reward(distance_to_goal, next_state)
        self.distance_to_goal = distance_to_goal

        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # Find trainsition weight, max of all buffer + weighted epsilon: to ensure non-zero values
        transition_weight = np.max(np.abs(self.ReplayBuffer.deque_weights[:self.ReplayBuffer.counter+1])) + self.weighted_epsilon
        # Store the trainsition and weight
        self.ReplayBuffer.collections_deque_append(transition,transition_weight) # COUNTER + 1

        # Train the model now...
        self.training_process()
        return None

    # Q-network training function, get the batch, train and find the loss, update weights for experience replay buffer.
    def training_process(self):
        # Wait until there is a full batch
        if self.ReplayBuffer.waiting_train(batch_size=self.batch_size):
                return None

        # Get weighted batch
        batch = self.ReplayBuffer.get_weighted_batch(batch_size=self.batch_size,alpha=self.weighted_alpha)

        # Train the DQN with weighted batch
        loss = self.dqn.train_q_network(batch, self.batch_size, self.target_steps)

        # Update weights for the batch
        self.ReplayBuffer.update_weights(self.dqn.TD_delta,self.weighted_epsilon)

        return None

    # Function to get the greedy action for a particular state: using the Q-network
    def get_greedy_action(self, state):
        prediction = self.dqn.q_network.forward(torch.tensor(state))
        discrete_action = torch.argmax(prediction).item()
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        return continuous_action

    # Move function is a kind of dictionary that transforms actions into movements
    def move(self, discrete_action):
        # 0 1 2 3 : N E S W
        step_size = 0.02
        if discrete_action == 0:
            return [0, step_size] # NORTH X_MOVEMENT, Y_MOVEMENT
        elif discrete_action == 1:
            return [step_size,0]
        elif discrete_action == 2:
            return [0,-step_size]
        elif discrete_action == 3: # It doesn't do this action anyomre as action space is reduced to 0 1 2
            return [-step_size,0]
        return [0,0] # IF WRONG ACTION, just in case, do not move

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self,gamma):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Create a target Q-network for TD prediction
        self.target_q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.train_counter = 0
        self.gamma = gamma
        self.TD_delta = 0

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, batch, batch_size, step_reboot_interval):
        # Once we have trained for 20 times, we update the target network!
        if self.train_counter%step_reboot_interval == 0:
            self.update_target_q_network()
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_bellman_loss(batch, batch_size)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Iterate the training counter
        self.train_counter += 1
        # Return the loss as a scalar
        return loss.item()

    # Update target network as a copy of the Q network.
    # Detatch function not required as target network is not set with an optimiser
    def update_target_q_network(self):
        dqn_dict =  torch.nn.Module.state_dict(self.q_network)
        torch.nn.Module.load_state_dict(self.target_q_network,dqn_dict)
        return 0

    # Find the loss using the bellman equation, also store the TD error values to update the experience Replay Buffer
    def _calculate_bellman_loss(self, batch, batch_size):

        states = np.array([batch[i][0] for i in range(batch_size)])
        actions = np.array([batch[i][1] for i in range(batch_size)])
        rewards = np.array([batch[i][2] for i in range(batch_size)])
        states_dash = np.array([batch[i][3] for i in range(batch_size)])

        states_tensor = torch.tensor(states)
        actions_tensor = torch.tensor(actions)#.long()
        rewards_tensor = torch.tensor(rewards)
        states_dash_tensor = torch.tensor(states_dash)

        # UNSQUEEZE REWARD TENSORS TO Nx1
        action_tensor = torch.unsqueeze(actions_tensor.long(),1)
        rewards_tensor = torch.unsqueeze(rewards_tensor.float(),1)

        network_prediction = self.q_network.forward(states_tensor) # PREDICT FOR ALL actions the Q Values
        predicted_Q = torch.gather(network_prediction,1,action_tensor) # Q network prediction of Q at state
        target_network_prediction = self.target_q_network.forward(states_dash_tensor) # Successor state prediction from target network
        actions_dash_tensor = torch.argmax(target_network_prediction,dim = 1) # Best action at successor state tensor
        bellman_temp = torch.max(target_network_prediction,dim = 1)[0] # Q value predicted by network for successor state
        bellman_Q = rewards_tensor + torch.unsqueeze(bellman_temp,1)*self.gamma # R + gamma*Q

        # Store TD deltas for weighted buffer in numpy array
        self.TD_delta = (bellman_Q - predicted_Q).tolist()

        bellman_loss = torch.nn.MSELoss()(predicted_Q,bellman_Q)
        return bellman_loss

# Replay buffer class
class ReplayBuffer:

    def __init__(self,batch_size = 100, init_size = 1000000, weighted_alpha = 1, weighted_epsilon = 0.1):
        self.deque = [None]*init_size #Didn't use collections deque, generated my own.
        self.deque_weights = np.zeros((init_size,))
        self.batch_size = batch_size
        self.weighted_alpha = weighted_alpha
        self.weighted_epsilon = weighted_epsilon
        self.init_size = init_size

        self.counter = 0 # variable used to count deque
        self.batch_counter = 0 # variable used for random suffling
        self.full_deque_sweeps = 0 # Number of overflows of the deque
        return None

    # Function which generates its own collections deque. appends weights and transitions to the deques
    def collections_deque_append(self, transition, weight):
        if self.counter == self.init_size:
            self.full_deque_sweeps += 1
            self.counter = 0
            print("Performed a full sweep, now we pop old values... Sweep {}".format(self.full_deque_sweeps))
        self.deque[self.counter] = transition
        self.deque_weights[self.counter] = weight
        self.counter += 1
        return 0

    # Update the weights given a new batch set of TD errors in the DQN
    def update_weights(self,td_delta,epsilon):
        self.deque_weights[self.weight_indexes] = np.abs(np.array(td_delta).flatten()) + epsilon
        return 0

    # Get prioritised experience replay buffer batch
    def get_weighted_batch(self,batch_size,alpha):
        if self.counter < batch_size:
            raise Exception("Not enough data to train your batch")
        weights_array = (self.deque_weights[:self.counter])**alpha # Update weights with weighting factor alpha
        buffer_probability = weights_array/np.sum(weights_array) # Find the probability of choosing each transition from the deque
        self.weight_indexes = np.random.choice(range(self.counter),batch_size,p = buffer_probability) # Weighted pick in transitions according to TD deltas
        return [self.deque[rand_index] for rand_index in self.weight_indexes]

    # wait until batch_size is smaller than counter
    def waiting_train(self,batch_size):
        if self.counter < batch_size:
            return True
        return False

    # Get a random batch, no prioritised experience replay buffer
    def get_batch(self,batch_size):
        if self.counter < batch_size:
            raise Exception("Not enough data to train your batch")
        rand_indexes = np.random.randint(0,self.counter,batch_size)
        batch = [self.deque[rand_index] for rand_index in rand_indexes]
        return batch
