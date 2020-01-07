import copy

import torch.nn.functional as F
from torch import optim, nn

from agents.AgentBase import AgentBase
from models.QNetworks import DuelingDQN
from replay_buffers.prioritized_experience_replay import PrioritizedReplayBuffer
from replay_buffers.replay_buffer import ReplayBuffer
from utils import *


class DQNAgent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 agent_name: str,
                 model: nn.Module,
                 replay_buffer,
                 state_size: int,
                 action_size: int,
                 discount: float = 0.99,  # discount factor
                 tau: float = 1e-3,  # for soft update of target parameters
                 learning_rate: float = 0.001,  # learning rate
                 update_every: int = 4,  # how often to update the network
                 is_double_dqn: bool = True,  # If we want it to be double dqn
                 seed: int = 0,
                 save_path: Path = None,
                 state_normalizer=RescaleNormalizer(),  # Todo: implement this
                 log_level: int = 0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        super().__init__(agent_name, save_path=save_path, state_normalizer=state_normalizer, log_level=log_level,
                         seed=seed)
        self.state_size = state_size
        self.action_size = action_size

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.discount = discount
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.is_double_dqn = is_double_dqn
        # Q-Network
        self.QNet_local = model.to(self.device)
        self.QNet_target = copy.deepcopy(model).to(self.device)

        self.optimizer = optim.Adam(self.QNet_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = replay_buffer
        self.per = isinstance(replay_buffer, PrioritizedReplayBuffer)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        error = None  # used in priorities only
        self.memory.add(state, action, reward, next_state, done, agent_idx=None, error=error)
        # Learn every UPDATE_EVERY time steps.
        self.total_steps += 1
        if self.total_steps % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.is_full_enough():
                self.learn(self.discount)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.QNet_local.eval()
        with torch.no_grad():
            action_values = self.QNet_local(state)
        self.QNet_local.train()

        # Epsilon-greedy action selection
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        idxs, experiences, is_weights = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        if self.is_double_dqn:
            # Getting the max action of local network (using weights w)
            max_actions_Snext_local = self.QNet_local(next_states).detach().max(1)[1].unsqueeze(1)
            # Getting the Q-value for these actions (using weight w^-)
            Q_targets_next = self.QNet_target(next_states).detach().gather(1, max_actions_Snext_local)
        else:
            # Get max predicted Q values (for next states) from target model
            # Find the max predicted Q values for next states (This is from the target model)
            Q_targets_next = self.QNet_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states (TD-target)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.QNet_local(states).gather(1, actions)

        # Compute loss
        if self.per:
            errors = torch.abs(Q_expected - Q_targets).detach().cpu()
            self.memory.batch_update(idxs, errors)
            is_weights = torch.from_numpy(is_weights).float().to(self.device)
            loss = (is_weights * F.mse_loss(Q_expected, Q_targets)).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.QNet_local, self.QNet_target, self.tau)

    def eval_step(self, state):
        """
        TODO: will be implemented when i use next time.
        :param state:
        :return:
        """
        pass

    def record_step(self, state):
        """
        TODO: will be implemented when i use next time.
        :param state:
        :return:
        """
        pass

    def save_all(self):
        super().save("qnetwork_local", self.QNet_local)
        super().save("qnetwork_target", self.QNet_target)
        super().save_stats("state_normalizer")

    def load_all(self, load_path: Path = None):
        self.QNet_local.load_state_dict(super().load_state_dict("qnetwork_local", load_path))
        self.QNet_target.load_state_dict(super().load_state_dict("qnetwork_target", load_path))
        self.load_stats("state_normalizer", load_path)


if __name__ == '__main__':
    # Example
    # Hardcoded for sake of the example
    action_size = 4
    state_size = 10
    buffer_size = 2 ** 10
    batch_size = 64
    seed = 0

    action_val_max = 10
    action_val_min = -10

    replay_buffer = ReplayBuffer(action_size, buffer_size, batch_size, seed)
    model = DuelingDQN(state_size, action_size, seed)
    agent = DQNAgent(agent_name="DuelingDoubleDQN",
                     model=model,
                     replay_buffer=replay_buffer,
                     state_size=state_size,
                     action_size=action_size,
                     is_double_dqn=True)
