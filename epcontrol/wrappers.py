# Deep reinforcement learning for large-scale epidemic control
# Copyright (C) 2020  Pieter Libin, Arno Moonens, Fabian Perez-Sanjines.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to pieter.libin@ai.vub.ac.be or arno.moonens@vub.be.

import random
from typing import Sequence
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gym.spaces import Box, Discrete, MultiBinary
import numpy as np

import epcontrol.compartments.AgeSEIR as AgeSEIR

class MultiAgentSelectObservation(ObservationWrapper):
    """Get the observation of some districts."""

    def __init__(self, env, districts_ids: Sequence[int], maac: bool = False):
        assert isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1
        super(MultiAgentSelectObservation, self).__init__(env)

        self.districts_ids = np.asarray(districts_ids)
        self.n_obs_vals_per_agent = self.unwrapped.observation_space_per_agent.shape[0]

        obs_space_per_agent = env.unwrapped.observation_space_per_agent
        n_districts = len(districts_ids)
        self.maac = maac
        if maac:
            self.observation_space = [obs_space_per_agent] * n_districts
        else:
            self.observation_space = Box(low=np.tile(obs_space_per_agent.low, n_districts),
                                         high=np.tile(obs_space_per_agent.high, n_districts))

    def observation(self, observation):
        obs = np.reshape(observation, (-1, self.n_obs_vals_per_agent))[self.districts_ids]
        if self.maac:
            return obs
        return obs.flatten()


class GetSingleAgentObservation(MultiAgentSelectObservation):
    """Get the observation of a single agent out of a multi-agent single-rank Box observation."""

    def __init__(self, env, district_id: int = None):
        if district_id is None:
            district_id = random.randint(0, self.env.unwrapped.n_agents - 1)
        super(GetSingleAgentObservation, self).__init__(env, [district_id])


class MultiAgentSelectAction(ActionWrapper):
    def __init__(self, env, districts_ids: Sequence[int], other_agents_action: int, maac: bool = False):
        super(MultiAgentSelectAction, self).__init__(env)
        assert isinstance(env.action_space, MultiBinary) and other_agents_action in [0, 1]
        self.districts_ids = np.asarray(districts_ids)
        self.other_agents_action = other_agents_action
        self.n_agents = env.unwrapped.n_agents
        if maac:
            self.action_space = [Discrete(2)]*len(districts_ids)
        else:
            self.action_space = MultiBinary(len(districts_ids))


    def action(self, action):
        full_action = np.full((self.n_agents,), self.other_agents_action)
        full_action[self.districts_ids] = action
        return full_action

    def reverse_action(self, action):
        return np.asarray(action)[self.districts_ids]

class SingleAgentAction(MultiAgentSelectAction):
    def __init__(self, env, other_agents_action: int, district_id=None):
        if district_id is None: # If no agent id was given, take a random one
            district_id = random.randint(0, self.env.unwrapped.n_agents - 1)
        super(SingleAgentAction, self).__init__(env, [district_id], other_agents_action)


class SingleAgentDiscreteAction(ActionWrapper):
    """
    Convert a MultiBinary(1) action space to a
    Discrete(2) action space.
    """

    def __init__(self, env):
        super(SingleAgentDiscreteAction, self).__init__(env)
        assert isinstance(env.action_space, MultiBinary) and env.action_space.n == 1

        self.action_space = Discrete(2)

    def action(self, action):
        return [action]

    def reverse_action(self, action):
        raise NotImplementedError()

class MultiAgentSelectReward(Wrapper):
    def __init__(self, env, districts_ids: Sequence[int]):
        super(MultiAgentSelectReward, self).__init__(env)
        self.districts_ids = np.asarray(districts_ids)

        self.districts_susceptibles = self._get_districts_susceptibles()
        self.max_districts_susceptibles = np.max(
            self.env.unwrapped._model.seir_state[self.districts_ids, AgeSEIR.Compartment.S.value, :])

    def _get_districts_susceptibles(self) -> float:
        return np.sum(self.env.unwrapped._model.seir_state[self.districts_ids, AgeSEIR.Compartment.S.value, :])

    def reset(self, **kwargs):
        s = self.env.reset(**kwargs)
        self.districts_susceptibles = self._get_districts_susceptibles()
        return s

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        current_districts_susceptibles = self._get_districts_susceptibles()
        reward = (current_districts_susceptibles - self.districts_susceptibles) / self.max_districts_susceptibles
        self.districts_susceptibles = current_districts_susceptibles
        return observation, reward, done, info


class SingleAgentReward(MultiAgentSelectReward):
    def __init__(self, env, district_id: int = None):
        if district_id is None:
            district_id = random.randint(0, self.env.unwrapped.n_agents - 1)
        super(SingleAgentReward, self).__init__(env, [district_id])


class Agent:
    def __init__(self, district, total_susceptibles):
        self.district = district
        self.total_susceptibles = total_susceptibles

    def reset(self):
        pass


class MAACWrapper(Wrapper):
    """
    Wrapper that accepts a list of actions and repeats the reward and done for each district.
    """

    def __init__(self, env, districts_ids):
        super(MAACWrapper, self).__init__(env)
        self.agents = []
        for district in districts_ids:
            sus = env.unwrapped._model.total_susceptibles_district(district)
            self.agents.append(Agent(district, sus))
        self.n = len(self.agents)
        self.action_space = [Discrete(2)] * self.n

    def step(self, action):
        action = np.nonzero(np.asarray(action))[1] # Convert back from Discrete to MultiBinary per agent
        s, rew, done, info = self.env.step(action)

        #each agent receives the global reward (cooperative setting)
        return s, [rew] * self.n, [done] * self.n, info

"""
Source for following code: https://github.com/arnomoonens/yarll
License:
MIT License

Copyright (c) 2020 Arno Moonens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class NormalizedObservationWrapper(ObservationWrapper):
    """
    Normalizes observations such that the values are
    between 0.0 and 1.0.
    """

    def __init__(self, env):
        super(NormalizedObservationWrapper, self).__init__(env)
        if not isinstance(self.env.observation_space, Box):
            raise AssertionError(
                "This wrapper can only be applied to environments with a continuous observation space.")
        if np.inf in self.env.observation_space.low or np.inf in self.env.observation_space.high:
            raise AssertionError(
                "This wrapper cannot be used for observation spaces with an infinite lower/upper bound.")
        self.observation_space: Box = Box(
            low=np.zeros(self.env.observation_space.shape),
            high=np.ones(self.env.observation_space.shape)
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.env.observation_space.low) / \
            (self.env.observation_space.high - self.env.observation_space.low)


class NormalizedRewardWrapper(RewardWrapper):
    """
    Normalizes rewards such that the values are between 0.0 and 1.0.
    """

    def __init__(self, env, low=None, high=None):
        super(NormalizedRewardWrapper, self).__init__(env)
        self.low = low if low is not None else self.env.reward_range[0]
        self.high = high if high is not None else self.env.reward_range[1]
        self.reward_range = (0.0, 1.0)

    def reward(self, rew):
        return (rew - self.low) / (self.high - self.low)
