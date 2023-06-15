
import os
import json
from os.path import dirname as up

import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np

from rasterio.plot import show

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = up(os.getcwd())

# set file paths
label_path = os.path.join(root_dir, 'processing_data/vectors/marsh_community.shp')
image_path = os.path.join(root_dir, 'processing_data/images/2018-07-01.tif')

# read in label data
labels_gdf = gpd.read_file(label_path)
# read in image data
image_src = rasterio.open(image_path)

# check on the crs of the shapefile


# check on the crs of the images








