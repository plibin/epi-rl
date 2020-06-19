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

#dis: derivative of the infected counts
#ddis: second derivative of the infected counts
def find_peaks(inf_normalized, dis, ddis, threshold):
    def around_zero(a, b):
        return (a < 0 < b) or (b < 0 < a)

    peaks = []
    for idx, i in enumerate(dis[:-1]):
        if (i == 0 or around_zero(i, dis[idx + 1])) and inf_normalized[idx] > threshold:
            if ddis[idx] < 0:
                peaks.append(idx)

    return peaks
