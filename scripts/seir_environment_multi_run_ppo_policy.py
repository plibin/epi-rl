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

import argparse
from pathlib import Path
import re

from gym.envs.registration import register, make
import numpy as np
import pandas as pd
from stable_baselines import PPO2

import epcontrol.census.Flux as Flux
from epcontrol.seir_environment import Granularity
from epcontrol.UK_RL_school_weekly import run_model
from epcontrol.wrappers import MultiAgentSelectAction, MultiAgentSelectObservation, \
    NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--flux", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--runs", type=int, required=True)
parser.add_argument("--paths", nargs="+", type=Path, required=True)
parser.add_argument("--n_districts", type=int, required=True)

args = parser.parse_args()

DISTRICTS_GROUP = None
if args.n_districts == 11:
    DISTRICTS_GROUP = ["Cornwall", "Plymouth", "Torbay", "East Devon", "Exeter", "Mid Devon",
                       "North Devon", "South Hams", "Teignbridge", "Torridge", "West Devon"]
elif args.n_districts == 3:
    DISTRICTS_GROUP = ["Cornwall", "West Devon", "Plymouth"]
else:
    raise ValueError("Wrong number of districts. Should be 3 or 11.")

n_weeks = 43
granularity = Granularity.WEEK

register(id="SEIRmulti-v0",
         entry_point="epcontrol.seir_environment:SEIREnvironment",
         max_episode_steps=n_weeks * (7 if granularity == Granularity.DAY else 1),
         kwargs=dict(grouped_census=pd.read_csv(args.census, index_col=0),
                     flux=Flux.Table(args.flux),
                     r0=args.R0,
                     n_weeks=n_weeks,
                     step_granularity=granularity,
                     model_seed=args.district_name,
                     budget_per_district_in_weeks=args.budget_in_weeks))

env = make("SEIRmulti-v0")
DISTRICTS_GROUP_IDS = [env.unwrapped.district_idx(name) for name in DISTRICTS_GROUP]

def districts_susceptibles(env, districts_ids):
    total = 0
    for idx in districts_ids:
        total += env.unwrapped._model.total_susceptibles_district(idx)
    return total

def evaluate(env, models, districts_ids, num_steps):
    obs = env.reset()
    sus_before = districts_susceptibles(env, districts_ids)
    for _ in range(num_steps):
        actions = []
        for district, ob in zip(DISTRICTS_GROUP, obs):
            action, _states = models[district].predict(ob)
            actions.append(action)
        action = np.concatenate(actions)
        obs, _, _, _ = env.step(action)
    sus_after = districts_susceptibles(env, districts_ids)
    attack_rate = 1.0 - (sus_after / sus_before)
    return attack_rate

env = NormalizedObservationWrapper(env)
env = NormalizedRewardWrapper(env)
env = MultiAgentSelectObservation(env, DISTRICTS_GROUP_IDS, maac=True)
env = MultiAgentSelectAction(env, DISTRICTS_GROUP_IDS, 1)

no_closures = [1] * n_weeks
weekends = False
(baseline_pd, baseline_ar, inf) = run_model(env.unwrapped._model, n_weeks, weekends, args.district_name, no_closures)

ppo_paths = []
for p in args.paths:
    ppo_paths.append(p)

models = {}
# Order of paths and DISTRICTS_GROUP are assumed to be the same.
for district_name, p in zip(DISTRICTS_GROUP, ppo_paths):
    model = PPO2.load(p / "params.zip")
    models[district_name] = model

# Check that we have all the districts
if set(models.keys()) != set(DISTRICTS_GROUP):
    print("set(models.keys())" + str(set(models.keys())))
    print("set(DISTRICTS_GROUP)" + str(set(DISTRICTS_GROUP)))
assert set(models.keys()) == set(DISTRICTS_GROUP)

print("ar-improvement")
for run in range(args.runs):
    attack_rate = evaluate(env, models, DISTRICTS_GROUP_IDS, n_weeks)
    print(baseline_ar - attack_rate)
env.close()
