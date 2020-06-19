# Deep reinforcement learning for large-scale epidemic control

This repo contains code that accompanies the paper _Deep reinforcement learning for large-scale epidemic control_, that was accepted for publication at the European Conference of Machine learning (2020). In this paper, we investigate a deep reinforcement learning approach to automatically learn prevention strategies in the context of pandemic influenza.

## Installation

First, clone the repo **recursively**:

```shell
git clone --recursive https://github.com/plibin-vub/epi-rl.git
```

The recursive clone is required to also get <https://github.com/oxwhirl/pymarl>, which is used for our multi-agent RL experiments.

Then, `cd` into the repo and run `pip install -r requirements.txt`.

## Experiments

In our paper we conduct experiments on multiple variants of SEIR environments and we use multiple reinforcement learning algorithms. Here, we explain how to apply those algorithms to the environments yourself. For this, we use scripts in the [scripts directory](./scripts).

### PPO on a single district

To learn PPO (from [Stable Baselines](https://github.com/hill-a/stable-baselines)) on a single district, you can use [seir\_environment\_single\_sb\_ppo.py](./scripts/seir_environment_single_sb_ppo.py). For example, it can be executed as such:

```shell
python scripts/seir_environment_single_sb_ppo.py \
--R0 1.8 \
--district_name Greenwich \
--budget_in_weeks 2 \
--census ./data/great_brittain/census.csv \
--total_timesteps 1000000 \
--outcome ar \
--monitor_path /tmp/SEIR_Greenwich_PPO
```

To see all the possible arguments, run `python scripts/seir_environment_single_sb_ppo.py --help`.

To apply a learnt PPO policy, you can use [seir\_environment\_single\_run\_ppo\_policy.py](./scripts/seir_environment_single_run_ppo_policy.py). For example, a policy learnt using the example above can be applied to the same environment as such:

```shell
python scripts/seir_environment_single_run_ppo_policy.py \
--R0 1.8 \
--district_name Greenwich \
--budget_in_weeks 2 \
--census ./data/great_brittain/census.csv \
--runs 10 \
--outcome ar \
--path /tmp/SEIR_Greenwich_PPO
```

To apply 11 (_Cornwall_, _Plymouth_, _Torbay_, _East Devon_, _Exeter_, _Mid Devon_, _North Devon_, _South Hams_, _Teignbridge_, _Torridge_ and _West Devon_) single-district policies together in 1 environment you can use [seir\_environment\_multi\_run\_ppo\_policy.py](./scripts/seir_environment_multi_run_ppo_policy.py). For example, to apply policies learnt on the 11 districts together:

```shell
python scripts/seir_environment_multi_run_ppo_policy.py \
--R0 1.8 \
--district_name Greenwich \
--budget_in_weeks 2 \
--census ./data/great_brittain/census.csv \
--flux ./data/great_brittain/commute.csv \
--runs 10 \
--paths /tmp/SEIR_Cornwall_PPO /tmp/SEIR_Plymouth_PPO /tmp/SEIR_Torbay_PPO /tmp/SEIR_East_Devon_PPO /tmp/SEIR_Exeter_PPO /tmp/SEIR_Mid_Devon_PPO /tmp/SEIR_North_Devon_PPO /tmp/SEIR_South_Hams_PPO /tmp/SEIR_Teignbridge_PPO /tmp/SEIR_Torridge_PPO /tmp/SEIR_West_Devon_PPO
```

Note that the order of the paths has to be the same as the order of the districts in `DISTRICTS_GROUP` inside the script.

### Computing the ground truth for a single district

To compute the ground truth for a single district, you can use [UK\_RL\_school\_weekly\_search.py](./scripts/UK_RL_school_weekly_search.py). For example, it can be executed as such:

```shell
python scripts/UK_RL_school_weekly_search.py \
--R0 1.8 \
--district Greenwich \
--grouped-census-fn ./data/great_brittain/census.csv \
--weeks 43 \
--budget-weeks 2 \
--outcome ar
```

### PPO on multiple districts

To learn PPO on 11 (same ones as above) districts together, you can use [seir\_environment\_multi\_sb\_ppo.py](./scripts/seir_environment_multi_sb_ppo.py). For example, it can be executed as such:

```shell
python scripts/seir_environment_multi_sb_ppo.py \
--R0 1.8 \
--district_name Greenwich \
--budget_in_weeks 2 \
--census ./data/great_brittain/census.csv \
--flux ./data/great_brittain/commute.csv \
--monitor_path /tmp/SEIR_11districts_PPO \
--total_timesteps 1000000
```

To see all the possible arguments, run `python scripts/seir_environment_multi_sb_ppo.py --help`.

To apply such a learnt policy, you can use [seir\_environment\_joint\_run\_ppo\_policy.py](./scripts/seir_environment_joint_run_ppo_policy.py). For example, a policy learnt using the example above can be applied to the same environment as such:

```shell
python scripts/seir_environment_joint_run_ppo_policy.py \
--R0 1.8 \
--district_name Greenwich \
--budget_in_weeks 2 \
--census ./data/great_brittain/census.csv \
--flux ./data/great_brittain/commute.csv \
--runs 10 \
--path /tmp/SEIR_11districts_PPO \
```

### Plotting a reward curve

To make a plot that shows the reward per episode for any of the experiments executed using PPO, you can use [plot\_ppo\_reward\_curve.py](./scripts/plot_ppo_reward_curve.py). For example, it can be executed as such:

```shell
python scripts/plot_ppo_reward_curve.py \
--path /tmp/SEIR_Cornwall_PPO/PPO2_1 \
--out_file /tmp/SEIR_Cornwall.png
```

### Multi-Agent RL algorithms

To run a multi-agent RL algorithm (from [PyMARL](https://github.com/oxwhirl/pymarl)) on a multi-agent version of the SEIR environment, you can use [seir\_pymarl.py](./scripts/seir_pymarl.py). For example, it can be executed as such:

```shell
python scripts/seir_pymarl.py \
--config=qmix \
--run_id=1 \
--iteration_id=1 \
with gamma=0.95 buffer_size=5000 lr=1e-05 optim_alpha=0.99 epsilon_anneal_time=35171 mixing_embed_dim=32
```

where:

- `--config` is the algorithm to run.
- `--run_id` is the ID of the run.
- `--iteration_id` is  number of the iteration.

More hyperparameters of the algorithm can be configured in `./scripts/seir.yaml`. An explanation and defaults for algorithm hyperparameters can be found in `./epcontrol/multiagent/pymarl/src/config/default.yaml`.

By running this, it will create a `rewards.txt` file inside `./epcontrol/multiagent/pymarl/results/{1}/{2}/`, where `{1}` is the `run_id` and `{2}` the `iteration_id`. This file will contain a reward in each line.
