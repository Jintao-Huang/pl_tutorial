# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 



# [Setup]
import imp
import os
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, IterableDataset
from zmq import device

DEBUG = True


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# [Memory]

Experience = namedtuple(
    "Experience",
    # ndarray; int; float; bool; ndarray
    field_names=["state", "action", "reward", "done", "new_state"],  # new_state用于计算new_state的预测q值
)
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        # [N, 4], [N], [N], [N], [N, 4]
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool8),  # torch.bool
            np.array(next_states),
        )
class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
    def __len__(self):
        return self.sample_size


# [Agent]

class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.state = None
        self.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        # state
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.from_numpy(self.state)[None]  # [N, I]
            state = state.to(device)

            q_values = net(state)  # [N, O]
            action = torch.argmax(q_values, dim=1).item()
        return int(action)

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float,
        device,
    ) -> Tuple[float, bool]:

        action = self.get_action(net, epsilon, device)
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            if DEBUG:
                print("done")
            self.reset()
        return reward, done  # self.state

# [DQN Lightning Module]

from copy import deepcopy

class DQNLightning(LightningModule):

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "CartPole-v1",
        gamma: float = 0.99,
        sync_steps: int = 10,
        replay_size: int = 1000,
        warm_start_steps: int = 1000,
        eps_decay_T: int = 1000,  # 
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        sample_size: int = 200,  # samples_in_epoch
    ) -> None:

        super(DQNLightning, self).__init__()
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.old_net = deepcopy(self.net)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0.  # 上一episode总reward
        self.episode_reward = 0.
        self.loss_fn=nn.MSELoss()
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        for i in range(steps):
            if DEBUG:
                self.env.render()
            self.agent.play_step(self.net, epsilon=1., device=device)

    # def forward(self, x: Tensor) -> Tensor:
    #     output = self.net(x)
    #     return output

    def dqn_mse_loss(self, batch: List[Tensor]) -> Tensor:
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions[:, None]).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.old_net(next_states).max(1)[0]
            next_state_values[dones] = 0.  # done, 预测为0

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return self.loss_fn(state_action_values, expected_state_action_values)

    def training_step(self, batch: List[Tensor], batch_idx) -> OrderedDict:

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_decay_T,
        )
        if DEBUG:
            self.env.render()
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if self.global_step % self.hparams.sync_steps == 0:
            self.old_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": self.total_reward,
            "reward": reward,
            "train_loss": loss,  # 当前loss; prog_bar有一个loss_mean
        }
        self.log_dict(log, prog_bar=True)
        # status = {
        #     "steps": self.global_step,
        #     "total_reward": self.total_reward,
        # }
        return loss
        # return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.hparams.sample_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()



# [Trainer]

model = DQNLightning()  # env="LunarLander-v2"

trainer = Trainer(
    gpus=1,
    max_epochs=200,
)
trainer.logger._default_hp_metric = False
trainer.fit(model)
