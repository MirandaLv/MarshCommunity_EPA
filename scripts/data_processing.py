
import os
import json
from os.path import dirname as up

import rasterio
import pandas as pd
import numpy as np

from rasterio.plot import show

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = up(os.getcwd())
vector_dir = os.path.join(root_dir, 'data/vectors')



