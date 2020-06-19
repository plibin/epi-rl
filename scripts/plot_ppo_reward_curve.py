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
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

def get_tf_file(folder: Path) -> Path:
    return next(folder.glob("events*"))


def get_tf_rewards(root_path):
    fn = get_tf_file(root_path)

    values = []
    steps = []
    for e in tf.compat.v1.train.summary_iterator(str(fn)):
        for v in e.summary.value:
            if v.tag == "episode_reward":
                values.append(v.simple_value)
                steps.append(e.step)

    if not steps:
        print(f"Run has no steps, skipping...")
        sys.exit(0)

    return (steps, values)


def plot_to_file(filename):
    if filename.endswith(".png"):
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, dpi=1000)
    elif filename.endswith(".pdf"):
        plt.savefig(filename, format='pdf')
    elif filename.endswith(".eps"):
        plt.savefig(filename, format='eps', dpi=1000)
    else:
        sys.exit("Unsupported output format! Exit-ing")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots()

    steps, values = get_tf_rewards(args.path)

    episode_len = steps[1] - steps[0]
    episode = np.divide(steps, episode_len)

    g = sns.lineplot(episode, values, ci="sd")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plot_to_file(args.out_file)
    plt.close()

if __name__ == "__main__":
    main()
