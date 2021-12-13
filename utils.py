from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor NN class
    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        Actor class initializer.

        :param state_dim: state dimensions, is the input size of the nn.
        :param action_dim: action dimension, is the output size of the nn.
        :param max_action: maximum number of possible actions.
        """
        super(Actor, self).__init__()

        # Neural Network
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

        # Store the number of actions and max action
        self.num_actions = action_dim
        self.max_action = max_action

    def forward(self, x):
        """
        Perform a forward on the neural network and retrieve the output action.

        :param x: input state
        :return: list of possible actions and its values
        """
        # Retrieve the output of the NN
        return self.max_action * torch.tanh(self.layers(x))


class ReplayBuffer(object):
    """
    Replay buffer class
    """
    def __init__(self, capacity):
        """
        Replay buffer initializer.

        :param capacity: maximum length of the replay buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition into the buffer.

        :param state: current state
        :param action: current action performed
        :param reward: reward achieved
        :param next_state: state where the agent will be when performing the action
        :param done: if the episode have been ended
        """
        # Expand state and next state
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # Store transition
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Retrieve a random sample from the buffer.

        :param batch_size: batch size for the sample
        :return: states, actions, rewards, next states and done batch info.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        """
        Get the length of the buffer.

        :return: buffer length
        """
        return len(self.buffer)
