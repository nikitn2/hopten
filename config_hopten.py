#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import json

# Define zero numerically
zero = 1e-20

# Set chi max
CHIMAX = 16384

# Set default dpi for plotting figures
dpi = 50

# Get data output directory
dir_data = os.environ.get('DATA')
if dir_data: dir_data += "/hopten/data/"
else: dir_data = "data/"
dir_figs = dir_data + "figures/"

# Custom encoder to handle numpy types
class json_atmos(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32): return float(obj)
        if isinstance(obj, np.int64): return int(obj)
        if isinstance(obj, np.ndarray): return list(obj)
        return super(json_atmos, self).default(obj)