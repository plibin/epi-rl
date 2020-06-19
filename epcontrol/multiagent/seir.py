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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import re
import os
from absl import logging
from gym.envs.registration import register, make
import numpy as np
import pandas as pd
from smac.env.multiagentenv import MultiAgentEnv

import epcontrol.census.Flux as Flux
from epcontrol.seir_environment import Granularity
from epcontrol.wrappers import MultiAgentSelectAction, MultiAgentSelectObservation, \
    MultiAgentSelectReward, NormalizedObservationWrapper, NormalizedRewardWrapper
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SEIR_MAE(MultiAgentEnv):
    def __init__(
            self,
            n_districts,
            census_data_path,
            flux_data_path,
            r0,
            n_weeks,
            model_seed,
            budget_per_district_in_weeks,
            monitor_path,
            debug,
            seed,
            run_id,
            iteration_id
    ):

        # Map arguments
        grouped_census = pd.read_csv(census_data_path, index_col=0)
        if n_districts == 11:
            self.district_names = ["Cornwall", "Plymouth", "Torbay", "East Devon", "Exeter", "Mid Devon",
                                   "North Devon", "South Hams", "Teignbridge", "Torridge", "West Devon"]
        elif n_districts == 3:
            self.district_names = ["Cornwall", "West Devon", "Plymouth"]

        self.flux_data_path = flux_data_path
        self.r0 = r0
        self.n_weeks = n_weeks
        self.model_seed = model_seed
        self._seed = seed
        self.budget_per_district_in_weeks = budget_per_district_in_weeks
        self.monitor_path = monitor_path
        self.debug = debug

        self.n_districts = len(self.district_names)
        self.n_agents = len(self.district_names)
        self.n_actions = 2
        self.episode_limit = 60

        self.agents = {}
        self._episode_count = 0
        self._obs = None
        self.debug = debug
        self.episode_reward = 0
        results = 'results/{}/{}/rewards.txt'.format(str(run_id), str(iteration_id))
        os.makedirs(os.path.dirname(results), exist_ok=True)
        self.results_file = open(results, 'w+')
        self.results_file.truncate(0)

        granularity = Granularity.WEEK

        register(id="SEIR-MA-PYMARL-v0",
                 entry_point="epcontrol.seir_environment:SEIREnvironment",
                 max_episode_steps=self.n_weeks * (7 if granularity == Granularity.DAY else 1),
                 kwargs=dict(grouped_census=grouped_census,
                             flux=Flux.Table(self.flux_data_path),
                             r0=self.r0,
                             n_weeks=self.n_weeks,
                             step_granularity=granularity,
                             model_seed=self.model_seed,
                             budget_per_district_in_weeks=self.budget_per_district_in_weeks))

        self._launch()

        # Try to avoid leaking processes on shutdown
        atexit.register(self.close)

    def _launch(self):
        """Launch the environment."""

        self.env = make("SEIR-MA-PYMARL-v0")
        districts_group_ids = [self.env.unwrapped.district_idx(name) for name in self.district_names]
        self.env = NormalizedObservationWrapper(self.env)
        self.env = NormalizedRewardWrapper(self.env)
        self.env = MultiAgentSelectObservation(self.env, districts_group_ids, maac=False)
        self.env = MultiAgentSelectAction(self.env, districts_group_ids, 1, maac=False)
        self.env = MultiAgentSelectReward(self.env, districts_group_ids)

        self.env.seed(self._seed)


    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        if self._episode_count == 0:
            # Launch env
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward


        try:
            # Update observations
            self._obs = self.env.reset()
        except:
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        return self.get_obs(), self.get_state()

    def _restart(self):
        """Restart the environment.
        """
        try:
            # set all default variables
            self.agents = {}
            self.episode_reward = 0
            self._obs = None
            self.debug = debug
        except:
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the process and launches a new one. """
        #close environment and lunch again
        #self.env.close()
        self._launch()

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions = np.array([int(a) for a in actions])

        # Send action request
        self._obs, reward, terminated, info = self.env.step(actions)

        info = {"info": False}

        self.episode_reward += reward
        # manage terminated
        if terminated:
            self.results_file.write('{}\n'.format(self.episode_reward))
            self.episode_reward = 0
            self._episode_count += 1

        return reward, terminated, info

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def get_obs_agent(self, agent_id):

        # return self.env.get_obs_by_district(agent_id)
        # It is not necessary
        return 0

    def get_obs(self):
        obs = np.split(self._obs, self.n_agents)
        return obs

    def get_state(self):
        return self._obs

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 17

    def get_state_size(self):
        """Returns the size of the global state."""

        return self.get_obs_size() * self.n_agents


    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions of one agent."""
        return [1] * self.n_actions

    def get_avail_actions(self):

        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close env."""
        self.results_file.close()
        self.env.close()


    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def save_replay(self):
        """Not implemented."""
        pass
