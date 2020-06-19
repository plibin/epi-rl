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

import numpy as np
from numpy import linalg as LA

class Eames2012(Enum):
    Children = 0
    Adolescents = 1
    Adults = 2
    Elderly = 3

def read_contact_matrix(fn):
    return np.genfromtxt(fn, delimiter=',', dtype=np.float32)

def make_reciprocal(cm, census):
    dim = cm.shape[0]
    reciprocal_cm = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            reciprocal_cm[i][j] = ((census[i] * cm[i][j]) + (census[j] * cm[j][i])) / (2*census[i])
    return reciprocal_cm

def compute_beta(R0, gamma, cm):
    dominant = np.amax(LA.eigvals(cm))
    beta = R0 * gamma / dominant
    return beta

def age_ranges(_):
    return {
        Eames2012.Children : (0, 4),
        Eames2012.Adolescents : (5, 18),
        Eames2012.Adults : (19, 64),
        Eames2012.Elderly : (65, None)
    }
