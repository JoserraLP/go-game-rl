import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    DQN algorithm class.
    """

    def __init__(self, num_inputs, num_actions):
        """
        DQN class initializer.

        :param num_inputs: state dimensions, is the input size of the nn.
        :param num_actions: action dimension, is the output size of the nn.
        """
        super(DQN, self).__init__()

        # Neural Network
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Store the number of actions
        self.num_actions = num_actions
        self.max_action = num_actions - 1

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
        # Get the action from the NN
        # Parse state and get Q-Value
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value = self.forward(state)
        # Order the actions by Q-Value
        actions = np.argsort(q_value[0][0][0].cpu().data.numpy())[::-1]
        # Retrieve only the valid action
        i = 0
        action = actions[i]
        while i < len(q_value[0][0][0]) and valid_moves[action] == 0:
            action = actions[i]
            i += 1
        return action
