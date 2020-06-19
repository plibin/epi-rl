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

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from gym import Env, spaces
import pandas as pd

import epcontrol.census.Flux as Flux
import epcontrol.compartments.AgeSEIR as AgeSEIR

from epcontrol.UK_SEIR_Eames import UK

class Granularity(Enum):
    DAY = 0
    WEEK = 1

class Outcome(Enum):
    ATTACK_RATE = 0
    PEAK_DAY = 1

class SEIREnvironment(Env):
    """SEIR environment."""
    def __init__(self,
                 grouped_census: pd.DataFrame,
                 flux: Flux,
                 r0: float,
                 n_weeks: int,
                 rho: Optional[float] = 1,
                 gamma: Optional[float] = (1 / 1.8),
                 delta: Optional[float] = 0.5,
                 mu: Optional[float] = None,
                 sde: Optional[bool] = True,
                 outcome: Optional[Outcome] = Outcome.ATTACK_RATE,
                 step_granularity: Optional[Granularity] = Granularity.WEEK,
                 budget_per_district_in_weeks: Optional[int] = None,
                 model_seed: Optional[str] = "Greenwich",
                 seed: Optional[int] = None,) -> None:
        super(SEIREnvironment, self).__init__()

        if mu is None:
            mu = np.log(r0)*.6

        if seed is not None:
            np.random.seed(seed)

        district_names = grouped_census.index.to_list()

        self.start_budget_per_district_in_weeks = budget_per_district_in_weeks
        self._model_params = [delta, r0, rho, gamma, district_names, grouped_census,
                              flux, mu, sde]
        self._model_seed = model_seed
        self._model = self._make_model()
        self._total_susceptibles = self._model.total_susceptibles()

        self.n_districts = len(district_names)
        self.n_weeks = n_weeks

        budget = -1
        if budget_per_district_in_weeks is None:
            budget = np.inf
        else:
            if step_granularity == Granularity.DAY:
                budget = budget_per_district_in_weeks * 7
            elif step_granularity == Granularity.WEEK:
                budget = budget_per_district_in_weeks
            else:
                raise ValueError("Wrong granularity")
        self.start_budget = budget
        self.budgets = np.full((self.n_districts,), self.start_budget, dtype=np.float32)

        self.step_granularity = step_granularity

        self.outcome = outcome

        self.max_sus = np.max(self._model.seir_state[:, AgeSEIR.Compartment.S.value, :])

        seir_values_per_agent = np.product(self._model.seir_state.shape[1:])

        lows = np.zeros((seir_values_per_agent * self.n_districts + \
            (self.n_districts if self.start_budget_per_district_in_weeks is not None else 0),))
        max_seir_value = np.max(self._model.seir_state)
        seir_highs_per_agent = np.full((seir_values_per_agent,), max_seir_value, dtype=np.float32)
        to_concat_per_agent = [seir_highs_per_agent]
        to_concat = [np.tile(seir_highs_per_agent, self.n_districts)]
        if self.start_budget_per_district_in_weeks is not None:
            max_budget = self.start_budget_per_district_in_weeks * \
                (7 if self.step_granularity == Granularity.DAY else 1)
            to_concat_per_agent.append([max_budget])
            to_concat.append(np.full((self.n_districts,),
                                     max_budget))
        highs_per_agent = np.concatenate(to_concat_per_agent, axis=0)
        highs = np.concatenate(to_concat, axis=0)
        self.observation_space = spaces.Box(low=lows,
                                            high=highs)
        # Action can only be 0 or 1 for every district, i.e. close or open schools in that district
        self.action_space = spaces.MultiBinary(self.n_districts)

        self.n_agents = self.n_districts

        self.observation_space_per_agent = spaces.Box(low=np.zeros_like(highs_per_agent),
                                                      high=highs_per_agent)

        self.reward_range = (0, self.max_sus)

        self._current_day = 0

        self.model_weekends = False

        self.infected_history = np.zeros((self.n_weeks * 7) + 1)

        self.total_closures = 0

    def _make_model(self) -> UK:
        model = UK(*self._model_params)
        model.seed(self._model_seed)
        return model


    def _get_obs(self) -> np.ndarray:
        """Get an observation from the environment.
        Return a copy of the model state to avoid references to the same array,
        which can change at every step.
        """
        to_concat = [np.reshape(self._model.seir_state, (self.n_districts, -1))]
        if self.start_budget_per_district_in_weeks is not None:
            clip_max = self.n_weeks * (7 if self.step_granularity == Granularity.DAY else 1)
            clipped_budgets = np.clip(self.budgets, 0, clip_max)
            to_concat.append(np.expand_dims(clipped_budgets, axis=1))
        complete_state = np.concatenate(to_concat, axis=1)
        return complete_state.flatten()

    def district_idx(self, district_name: str) -> int:
        return self._model.district_idx(district_name)

    def reset(self) -> np.ndarray:
        self._current_day = 0
        self.budgets.fill(self.start_budget)
        self._model.reset()
        self._model.seed(self._model_seed)
        self._total_susceptibles = self._model.total_susceptibles()
        self.infected_history = np.zeros((self.n_weeks * 7) + 1)
        self.total_closures = 0
        return self._get_obs()

    def _weekend(self, week_day):
        return week_day in (5, 6)

    def _step_day(self, week_day, school_states):
        if self.model_weekends and self._weekend(week_day):
            self._model.step(self._current_day, np.zeros(self.n_districts))
        else:
            self._model.step(self._current_day, school_states)

    def _reduce_budget(self, school_states):
        self.budgets -= np.invert(school_states.astype(np.bool))

    def step(self, action: Sequence[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        school_states = np.array(action, copy=True)
        assert school_states.ndim == 1, f"Expected a 1-dimensional list, got {school_states}."

        # reduce the budget
        if self.step_granularity == Granularity.DAY:
            week_day = self._current_day % 7
            if not self._weekend(week_day):
                self._reduce_budget(school_states)
        elif self.step_granularity == Granularity.WEEK:
            self._reduce_budget(school_states)

        # Schools of districts with no budget left must stay open
        school_states[self.budgets < 0] = 1
        self.total_closures += (school_states.size - np.sum(school_states))

        if self.step_granularity == Granularity.DAY:
            week_day = self._current_day % 7
            self._step_day(week_day, school_states)
            self._current_day += 1
            self.infected_history[self._current_day] = self._model.total_infected()
        elif self.step_granularity == Granularity.WEEK:
            for d in range(7):
                self._step_day(d, school_states)
                self._current_day += 1
                self.infected_history[self._current_day] = self._model.total_infected()
        else:
            raise ValueError("Wrong granularity")

        done = self._current_day >= (self.n_weeks * 7)

        current_susceptibles = self._model.total_susceptibles()

        reward = -1
        if self.outcome == Outcome.ATTACK_RATE:
            reward = (current_susceptibles - self._total_susceptibles)
        elif self.outcome == Outcome.PEAK_DAY:
            if done:
                reward = self._model.peak_day(self.infected_history)
            else:
                reward = 0
        else:
            raise ValueError("Wrong outcome")

        self._total_susceptibles = current_susceptibles

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        pass
