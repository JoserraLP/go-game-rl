import tkinter.font as tkFont
from tkinter import Tk, IntVar, Label, Button, CENTER, Radiobutton, N

import numpy as np
import torch

import gym
from ddpg import DDPG
from dqn import DQN
from reinforce import REINFORCE
from td3 import TD3


class GUI:
    """
    GUI class
    """

    def __init__(self):
        """
        GUI initializer
        """

        # Window settings
        self.window = Tk()
        self.window.title('RL Go Game')
        self.window.geometry(str(500) + 'x' + str(330))

        # Window font
        font = tkFont.Font(family="Arial", size=16)

        # Message to the user
        label = Label(self.window, text="Please, select the agent you want to play against:")
        label['font'] = font
        label.pack(anchor=N, padx=20, pady=10)

        # Define all the possible options with game modes based on the agent
        self._selection = IntVar()
        r1 = Radiobutton(self.window, text="DQN (Easy)", bg="#90F61B", variable=self._selection, value=1)
        r1['font'] = font
        r1.pack(anchor=CENTER, padx=10, pady=10)

        r2 = Radiobutton(self.window, text="REINFORCE (Medium)", bg="#F6C21B", variable=self._selection, value=2)
        r2['font'] = font
        r2.pack(anchor=CENTER, padx=10, pady=10)

        r3 = Radiobutton(self.window, text="DDPG (Medium-Hard)", bg="#F6911B", variable=self._selection, value=3)
        r3['font'] = font
        r3.pack(anchor=CENTER, padx=10, pady=10)

        r4 = Radiobutton(self.window, text="TD3 (Hard)", bg="#F6401B", variable=self._selection, value=4)
        r4['font'] = font
        r4.pack(anchor=CENTER, padx=10, pady=10)

        # Button to start the game
        self.submit_btn = Button(self.window, text="Play game", command=self.play_game)
        self.submit_btn['font'] = tkFont.Font(family="Arial", size=16)
        self.submit_btn.pack(anchor=CENTER)

    def play_game(self):
        """
        Method to play the game when the submit button has been pressed.
        """

        # Default game values
        board_size = 7
        komi = 0

        # Initialize environment and retrieve its information
        go_env = gym.make('gym_go:go-v0', size=board_size, komi=komi)
        state_dim = np.prod(list(go_env.observation_space.shape))
        action_dim = go_env.action_space.n

        # Define empty agent policy
        policy = None

        # Retrieve selected agent
        agent_selected = self._selection.get()
        if agent_selected == 1:  # DQN
            # Overwrite the state dim as it is trained with other values
            state_dim = go_env.observation_space.shape[1]
            # Define policy and load its NN weights
            policy = DQN(num_inputs=state_dim, num_actions=action_dim)
            policy.layers.load_state_dict(torch.load('./models/dqn/model.h5', map_location=torch.device('cpu')))
        elif agent_selected == 2:  # REINFORCE
            # Overwrite the state dim as it is trained with other values
            state_dim = go_env.observation_space.shape[1]
            # Define policy and load its NN weights
            policy = REINFORCE(num_inputs=state_dim, num_actions=action_dim)
            policy.layers.load_state_dict(torch.load('./models/reinforce/model.h5', map_location=torch.device('cpu')))
        elif agent_selected == 3:  # DDPG
            # Define policy and load its NN weights
            policy = DDPG(state_dim, action_dim, max_action=action_dim - 1)
            policy.load("ddpg", directory='./models/ddpg')
        elif agent_selected == 4:  # TD3
            # Define policy and load its NN weights
            policy = TD3(state_dim, action_dim, max_action=action_dim - 1)
            policy.load("td3", directory='./models/td3')

        # Game loop
        done = False
        # Game not ended
        while not done:
            # Retrieve action from the user
            action = go_env.render(mode="human")
            # Perform action on the game
            state, reward, done, info = go_env.step(action)

            # If the game has not ended
            if go_env.game_ended():
                break

            # The agent selects an action
            action = policy.select_action(state, go_env.valid_moves())
            # Perform action on the game
            state, reward, done, info = go_env.step(action)
        # Finally render game when ended
        go_env.render(mode="human")

    def run_gui(self):
        # Tkinter window loop
        self.window.mainloop()
