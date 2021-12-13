import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from utils import Actor


class Critic(nn.Module):
    """
    Critic NN class
    """

    def __init__(self, state_dim, action_dim):
        """
        Critic class initializer.

        :param state_dim: state dimensions, is the input size of the nn.
        :param action_dim: action dimension, is the output size of the nn.
        """

        super(Critic, self).__init__()
        # Defining the first Critic neural network

        self.layers_1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 1, 128),  # 1 is the action dim -> Int
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        # Defining the second Critic neural network
        self.layers_2 = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 1, 128),  # 1 is the action dim -> Int
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        # Store the number of actions
        self.action_dim = action_dim

    def forward(self, state, action):
        """
        Perform a forward on both neural networks and retrieve their outputs actions.

        :param state: input state
        :param action: input action
        :return: q_value of the action and state pair
        """
        state_action = torch.cat([state, action], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = self.layers_1(state_action)
        # Forward-Propagation on the second Critic Neural Network
        x2 = self.layers_2(state_action)
        return x1, x2

    def Q1(self, state, action):
        """
        Perform a forward on the first neural network and retrieve its output action.

        :param state: input state
        :param action: input action
        :return: q_value of the action and state pair
        """
        # Retrieve the output of the first Critic
        return self.layers_1(torch.cat([state, action], 1))


class TD3(object):
    """
    TD3 algorithm class.
    """

    def __init__(self, state_dim, action_dim, max_action):
        """
        TD3 class initializer.

        :param state_dim: state dimensions, is the input size of the nn.
        :param action_dim: action dimension, is the output size of the nn.
        :param max_action: maximum number of possible actions.
        """
        # Define the actor and actor target NNs
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        # Load weights from actor, with same parameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        # Define actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        # Define the critic and critic target NNs
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # Load weights from critic, with same parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Define critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        # Define class variables
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, valid_moves):
        """
        Retrieve a valid action from the actor NN output.

        :param state: actual state of the agent.
        :param valid_moves: valid moves allowed in the game.
        :return: action represented as an int.
        """
        state = torch.Tensor(state.reshape(1, -1))
        # Get possible actions from the actor NN and sort it
        actions = self.actor(state).cpu().data.numpy()[0]
        actions = np.argsort(actions)[::-1]
        # Retrieve only the valid action
        i = 0
        action = actions[i]
        while i < len(actions) and valid_moves[action] == 0:
            action = actions[i]
            i += 1
        return action

    def train(self, env, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_freq=2):
        """
        Training process of the TD3 algorithm.

        :param env: environment
        :param replay_buffer: memory buffer
        :param iterations: number of possible iterations (games)
        :param batch_size: batch size for the sampling from memory. Default to 100.
        :param discount: discount factor. Default to 0.99.
        :param tau: target network update rate. Default to 0.0005
        :param policy_freq: policy network update frequency. Default to 2.
        """
        # Number of iterations
        for it in range(iterations):

            # First we are going to sample a batch of transitions (s, a, r, s', d) from the memory
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                batch_size)
            states = torch.Tensor(batch_states)
            next_states = torch.Tensor(batch_next_states)
            actions = torch.Tensor(batch_actions)
            rewards = torch.Tensor(batch_rewards)
            dones = torch.Tensor(batch_dones)

            # Reshape the batches of states, actions and next_states to use them into
            # the NNs
            states = states.reshape((-1, self.state_dim))
            actions = actions.reshape((-1, 1))
            next_states = next_states.reshape((-1, self.state_dim))

            # Retrieve next action from the actor target NN and get those valid
            next_actions = self.actor_target(next_states)

            # Define next_valid_actions list
            next_valid_actions = []
            # Process and select only the valid action
            for next_action in next_actions:
                # Retrieve valid moves
                valid_moves = env.valid_moves()
                # Sort the next_actions
                next_actions = np.argsort(next_action.cpu().data.numpy())[::-1]
                # Retrieve only the valid action
                i = 0
                action = next_actions[i]
                while i < len(next_actions) and valid_moves[action] == 0:
                    action = next_actions[i]
                    i += 1
                # Append the next_valid_action
                next_valid_actions.append(action)

            # Reshape the next_valid_actions batch
            next_valid_actions = torch.FloatTensor(next_valid_actions).reshape((-1, 1))

            # The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,
            # a’) and Qt2(s’,a’) as outputs 
            target_Q1, target_Q2 = self.critic_target(next_states, next_valid_actions)

            # We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the 
            # discount factor 
            target_Q = rewards + ((1 - dones) * discount * target_Q).detach()

            # The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,
            # a) as outputs 
            current_Q1, current_Q2 = self.critic(states, actions)

            # We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(
            # Q2(s,a), Qt) 
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Update critic networks with backpropagation of the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the
            # first Critic model 
            if it % policy_freq == 0:

                # Retrieve policy loss action from the actor NN and get those valid
                policy_loss_actions = self.actor.forward(states)
                policy_loss_valid_actions = []
                # Process and select only the valid action
                for policy_loss_action in policy_loss_actions:
                    # Retrieve valid moves
                    valid_moves = env.valid_moves()
                    # Sort the next_actions
                    actions = np.argsort(policy_loss_action.cpu().data.numpy())[::-1]
                    # Retrieve only the valid action
                    i = 0
                    action = actions[i]
                    while i < len(actions) and valid_moves[action] == 0:
                        action = actions[i]
                        i += 1
                    # Append the next_valid_action
                    policy_loss_valid_actions.append(action)

                # Reshape the next_valid_actions batch
                policy_loss_valid_actions = torch.FloatTensor(policy_loss_valid_actions).reshape((-1, 1))

                # Calculate the actor loss (policy loss)
                policy_loss = -self.critic.Q1(states, policy_loss_valid_actions).mean()

                # Update actor networks with backpropagation of the loss
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                # Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        """
        Save the trained models: actor and critic.

        :param filename: name of the trained models
        :param directory: directory where the file will be stored
        """
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        """
        Load pre-trained models: actor and critic.

        :param filename: name of the trained models
        :param directory: directory where the file will be stored
        """
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename),
                                              map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename),
                                               map_location=torch.device('cpu')))
