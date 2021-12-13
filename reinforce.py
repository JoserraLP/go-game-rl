import torch
import torch.nn as nn
import numpy as np


class REINFORCE(nn.Module):
    """
    REINFORCE algorithm class.
    """

    def __init__(self, num_inputs, num_actions):
        """
        REINFORCE class initializer.

        :param num_inputs: state dimensions, is the input size of the nn.
        :param num_actions: action dimension, is the output size of the nn.
        """
        super(REINFORCE, self).__init__()

        # Neural Network
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

        # Store the number of actions
        self.num_actions = num_actions

    def forward(self, x):
        """
        Perform a forward on the neural network and retrieve the output action.

        :param x: input state
        :return: list of possible actions and its values
        """
        # Retrieve the output of the NN
        return self.layers(x)

    def select_action(self, state, valid_moves):
        """
        Retrieve a valid action from the NN output.

        :param state: actual state of the agent.
        :param valid_moves: valid moves allowed in the game.
        :return: action represented as an int.
        """
        # Parse state
        state = torch.FloatTensor(state).unsqueeze(0)
        # Get the action probabilities
        prob = self.forward(state)
        action_probs = prob[0][0][0].cpu().data.numpy()
        # Select random valid action based on the probabilities
        action = np.random.choice(self.num_actions, p=action_probs)
        while valid_moves[action] == 0:
            action = np.random.choice(self.num_actions, p=action_probs)
        return action
