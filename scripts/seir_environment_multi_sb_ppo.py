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
import datetime
import os
from pathlib import Path

from gym.envs.registration import register, make
import pandas as pd
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import logger
from stable_baselines import PPO2

import epcontrol.census.Flux as Flux
from epcontrol.seir_environment import Granularity
from epcontrol.wrappers import MultiAgentSelectAction, MultiAgentSelectObservation, \
    MultiAgentSelectReward, NormalizedObservationWrapper, NormalizedRewardWrapper

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--district_name", required=True)
parser.add_argument("--budget_in_weeks", type=int, required=True)
parser.add_argument("--census", type=Path, required=True)
parser.add_argument("--flux", type=Path, required=True)
parser.add_argument("--R0", type=float, required=True)
parser.add_argument("--monitor_path",
                    type=str,
                    default=f"/tmp/SEIR-PPO-{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}/")
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_hidden_layers", type=int, default=0)
parser.add_argument("--n_hidden_units", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument("--n_minibatches", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=128)
parser.add_argument("--max_grad_norm", type=float, default=None)
parser.add_argument("--clip_range", type=float, default=0.2)
parser.add_argument("--total_timesteps", type=int, required=True)

args = parser.parse_args()

ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
tf.Session(config=config).__enter__()

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

districts_group = ["Cornwall", "Plymouth", "Torbay", "East Devon", "Exeter", "Mid Devon",
                   "North Devon", "South Hams", "Teignbridge", "Torridge", "West Devon"]

env = make("SEIRmulti-v0")

districts_group_ids = [env.unwrapped.district_idx(name) for name in districts_group]
env = NormalizedObservationWrapper(env)
env = NormalizedRewardWrapper(env)
env = MultiAgentSelectObservation(env, districts_group_ids)
env = MultiAgentSelectAction(env, districts_group_ids, 1)
env = MultiAgentSelectReward(env, districts_group_ids)

logger.configure(folder=args.monitor_path, format_strs=["csv"])

env = DummyVecEnv([lambda: env])

print(f"tensorboard --logdir {args.monitor_path}")

layers = [args.n_hidden_units] * args.n_hidden_layers

model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log=args.monitor_path,
             ent_coef=args.entropy_coef, learning_rate=args.learning_rate,
             noptepochs=args.n_epochs, nminibatches=args.n_minibatches,
             n_steps=args.n_steps,
             policy_kwargs={"layers": layers})
model.learn(total_timesteps=args.total_timesteps)
model.save(os.path.join(args.monitor_path, "params"))
env.close()
