
import os
from os.path import dirname as up

# data processing
import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.plot import show

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep


"""
# 1	Coastal Blue	443 (20)	Yes - with Sentinel-2 band 1
# 2	Blue	490 (50)	Yes - with Sentinel-2 band 2
# 3	Green I	531 (36)	No equivalent with Sentinel-2
# 4	Green	565 (36)	Yes - with Sentinel-2 band 3
# 5	Yellow	610 (20)	No equivalent with Sentinel-2
# 6	Red	665 (31)	Yes - with Sentinel-2 band 4
# 7	Red Edge	705 (15)	Yes - with Sentinel-2 band 5
# 8	NIR	865 (40)	Yes - with Sentinel-2 band 8a
"""

root_dir = up(os.getcwd())

# set file paths
label_path = os.path.join(root_dir, 'data/processing_data/vectors/sample_combined.geojson')
image_path = os.path.join(root_dir, 'data/processing_data/images/subregion_planet_composite.tif')
dem_path = os.path.join(root_dir, 'data/processing_data/images/cbtb_dem_MarshComm_proj.tif')

# read in label data
pts_gdf = gpd.read_file(label_path)
pts_gdf['type_class'] = pts_gdf['type'].apply(lambda x: 1 if x=='juncus' else 2)

# read in image data
image_src = rasterio.open(image_path)
#
# # plot the image in RGB Composite Image
# image_array = image_src.read()
# rgb = ep.plot_rgb(image_array,
#                   rgb=(5,3,1),
#                   figsize=(10, 16))
# # plt.show()
#
# # plot the image in RGB Composite Image with Stretch
# ep.plot_rgb(image_array,
#             rgb=(5,3,1),
#             stretch=True,
#             str_clip=0.2,
#             figsize=(10, 16))
# # plt.show()
#
# # Histograms of the image bands
# colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
#           'maroon', 'purple']
#
# ep.hist(image_array,
#         colors = colors,
#         title=[f'Band-{i}' for i in range(1, 9)],
#         cols=3,
#         alpha=0.5,
#         figsize = (12, 10))
#
# # plt.show()

# get NDVI
# ndvi = (image_array[7] - image_array[5])/ (image_array[7] + image_array[5])
# ndvi = es.normalized_diff(image_array[7], image_array[5])


# Generate a standalone data for analysis
# extract raster values for each sample point
coords = [(x,y) for x, y in zip(pts_gdf.geometry.x, pts_gdf.geometry.y)]

pts_gdf['Raster_Value'] = [x for x in image_src.sample(coords)]
pts_gdf['B1'] = pts_gdf.apply(lambda x: x['Raster_Value'][0], axis=1)
pts_gdf['B2'] = pts_gdf.apply(lambda x: x['Raster_Value'][1], axis=1)
pts_gdf['B3'] = pts_gdf.apply(lambda x: x['Raster_Value'][2], axis=1)
pts_gdf['B4'] = pts_gdf.apply(lambda x: x['Raster_Value'][3], axis=1)
pts_gdf['B5'] = pts_gdf.apply(lambda x: x['Raster_Value'][4], axis=1)
pts_gdf['B6'] = pts_gdf.apply(lambda x: x['Raster_Value'][5], axis=1)
pts_gdf['B7'] = pts_gdf.apply(lambda x: x['Raster_Value'][6], axis=1)
pts_gdf['B8'] = pts_gdf.apply(lambda x: x['Raster_Value'][7], axis=1)

# deleting not use columns
pts_gdf.drop(columns=['Raster_Value'], inplace=True)
pts_gdf.drop(columns=['rand_point'], inplace=True)

# get the DEM data
image_dem_src = rasterio.open(dem_path)
pts_gdf['dem_value'] = [x for x in image_dem_src.sample(coords)]
pts_gdf['dem_value'] = pts_gdf['dem_value'].apply(lambda x: x[0])

# pts_gdf.to_file(os.path.join(root_dir, 'data/processing_data/vectors/points_planet_composite.geojson'), driver='GeoJSON')


# data visualization
df = pd.DataFrame(pts_gdf.drop(columns='geometry'))
df.drop(columns=['type_class'], inplace=True)

# grouped_df = df.groupby('type').agg({'B1': ['mean'], 'B2': ['mean'], 'B3': ['mean'],
#                               'B4': ['mean'], 'B5': ['mean'], 'B6': ['mean'],
#                                'B7': ['mean'], 'B8': ['mean']})

grouped_df = df.groupby('type').mean()
grouped_df = grouped_df.T
index = pd.Index(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
grouped_df = grouped_df.set_index(index)
grouped_df.plot()

plt.show()







