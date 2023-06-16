
import os
from os.path import dirname as up

# data processing
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = up(os.getcwd())

# set file paths
label_path = os.path.join(root_dir, 'processing_data/vectors/marsh_community.shp')
# image_path = os.path.join(root_dir, 'processing_data/images/2018-07-01.tif')

# read in label data
labels_gdf = gpd.read_file(label_path)

# read in image data
# image_src = rasterio.open(image_path)
P_planet_bands = glob("...../*B?*.tif")
P_planet_bands.sort()

l = []
for i in P_planet_bands:
  with rasterio.open(i, 'r') as f:
    l.append(f.read(1))

# Image array
arr_st = np.stack(l)


# check on the crs of the shapefile


# check on the crs of the images








