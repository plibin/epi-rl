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

import collections
from copy import deepcopy
import os
import sys
from functools import partial
from sacred.observers import FileStorageObserver
from smac.env import MultiAgentEnv
import yaml

# Add the pymarl module src folder
path_pymarl = os.path.abspath("epcontrol/multiagent/pymarl/src")
print(path_pymarl)
sys.path.append(path_pymarl)

import main as main
from run import run
from epcontrol.multiagent.seir import SEIR_MAE
from envs import REGISTRY as env_REGISTRY
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

env_REGISTRY["seir"] = partial(env_fn, env=SEIR_MAE)

def _get_config_env(params):
    config_name = "seir"
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), f"{config_name}.yaml"), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, f"{config_name}.yaml error: {exc}"
        return config_dict

def _get_config(params, name):
    config_name = None
    for _i, _v in enumerate(params):
        print(_v.split("=")[0])
        if _v.split("=")[0] == name:
            print("----")
            config_name = _v.split("=")[1]
            del params[_i]
            return config_name

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    run_id = _get_config(params, "--run_id")
    iteration_id = _get_config(params, "--iteration_id")
    params.append("env_args.run_id=" + run_id)
    params.append("env_args.iteration_id=" + iteration_id)
    # Get the defaults from default.yaml
    with open(os.path.join(path_pymarl, "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, f"default.yaml error: {exc}"

    # Load algorithm and env base configs
    env_config = _get_config_env(params)
    alg_config = main._get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    # now add all the config to sacred
    main.ex.add_config(config_dict)

    # Save to disk by default for sacred
    main.logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(main.results_path, "sacred")

    main.ex.observers.append(FileStorageObserver.create(file_obs_path))
    main.ex.run_commandline(params)
