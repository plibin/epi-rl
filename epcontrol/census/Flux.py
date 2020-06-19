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
import pandas as pd

class Table:
    def __init__(self, fn):
        df = pd.read_csv(fn, index_col=0)
        self.Tij = df.values.astype(np.float32)
        self.district_names = df.columns

class SingleDistrictStub:
    def __init__(self, district_name):
        self.Tij = np.array([[1]]).astype(np.float32)
        self.district_names = np.array([district_name])
