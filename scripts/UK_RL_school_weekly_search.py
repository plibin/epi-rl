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

from argparse import ArgumentParser
import itertools
import numpy as np
import pandas as pd

import epcontrol.census.Flux as flux
from epcontrol.UK_SEIR_Eames import UK

from epcontrol.UK_RL_school_weekly import run_model

parser = ArgumentParser(description="UK_RL_school_weekly_search")

parser.add_argument("--grouped-census-fn", dest="grouped_census_fn", type=str, required=True)
parser.add_argument("--district", dest="district", type=str, required=True)
parser.add_argument("--R0", dest="R0", type=float, required=True)
parser.add_argument("--weeks", dest="weeks", type=int, required=True)
parser.add_argument("--budget-weeks", dest="budget_weeks", type=int, required=True)
parser.add_argument("--mu", dest="mu", type=float, required=False)
parser.add_argument("--outcome", choices=["ar", "pd"], required=True)

args = parser.parse_args()

flux = flux.SingleDistrictStub(args.district)
grouped_census = pd.read_csv(args.grouped_census_fn, index_col=0)

grouped_census = grouped_census.filter(items=[args.district], axis=0)

rho = 1
gamma = 1 / 1.8

mu = np.log(args.R0) * .6
if args.mu is not None:
    mu = args.mu

district_names = [args.district]

delta = .5
model = UK(delta, args.R0, rho, gamma, district_names, grouped_census, flux, mu, sde=False)

weekends = False
school_combinations = [list(i) for i in itertools.product([0, 1], repeat=args.weeks)]
school_combinations = list(filter(lambda l: l.count(0) == args.budget_weeks, school_combinations))

no_closures = [1] * args.weeks
(baseline_pd, baseline_ar) = run_model(model, args.weeks, weekends, args.district, no_closures)

#print header
print("combination," + args.outcome + "_improvement")

for c in school_combinations:
    peak_day, attack_rate = run_model(model, args.weeks, weekends, args.district, c)

    peak_day_improvement = peak_day - baseline_pd
    attack_rate_improvement = baseline_ar - attack_rate

    c_str = "".join(map(str, c))
    if args.outcome == "pd":
        print(c_str + "," + str(peak_day_improvement))
    elif args.outcome == "ar":
        print(c_str + "," + str(attack_rate_improvement))
