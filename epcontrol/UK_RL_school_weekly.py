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

import numpy as np

def run_model(model, weeks, weekends, district, comb):
    model.reset()
    model.seed(district)

    school_states = np.zeros(weeks*7)
    for w in range(weeks):
        for d in range(7):
            idx = (w * 7) + d
            if weekends:
                if d in [5, 6]:
                    school_states[idx] = 0
                else:
                    school_states[idx] = comb[w]
            else:
                school_states[idx] = comb[w]

    sus_before = model.total_susceptibles()
    inf = []

    # phase where there is control
    for t in range(weeks * 7):
        s_t = np.array([school_states[t]])
        model.step(t, s_t)
        inf.append(model.total_infected())

    # phase after the control, to verify that no second peak is caused by the control
    for t in range(weeks * 7):
        s_t = np.array([1])
        model.step(t, s_t)
        inf.append(model.total_infected())

    peak_day = model.peak_day(inf)
    sus_after = model.total_susceptibles()
    attack_rate = 1.0 - (sus_after / sus_before)

    return (peak_day, attack_rate, inf)
