"""
author: Miranda Lv
Date: July, 2023
"""

import os
from os.path import dirname as up

# data processing
import rasterio
from rasterio.enums import Resampling
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterio.crs import CRS

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.plot import show

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep


"""
B1: annual mean VV channel
B2: annual standard deviation VV channel
B3: VV winter mean
B4: VV spring mean
B5: VV summer mean
B6: VV fall mean
B7: annual mean VH channel
B8: annual standard deviation VH channel
B9: VH winter mean
B10: VH spring mean
B11: VH summer mean
B12: VH fall mean
"""


root_dir = up(os.getcwd())

# set file paths
label_path = os.path.join(root_dir, 'data/processing_data/vectors/sample_combined.geojson')
image_path = os.path.join(root_dir, 'data/raw/images/s1_radar_12_band_temporal_composite_v1_projected.tif')
resample_path = os.path.join(root_dir, 'data/processing_data/images/s1_radar_12_band_temporal_composite_v1_resampled.tif')

# image resampling

def img_resample(input_img, upscale_factor, output_img):
    
    dataset = rasterio.open(input_img)
    
    data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear)
    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2]))
    
    new_dataset = rasterio.open(
        output_img,
        'w',
        driver='GTiff',
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=data.dtype,
        crs=CRS.from_epsg(32618), # all data is in EPSG:32618
        transform=transform,
    )

    new_dataset.write(data)
    new_dataset.close()


# # Only run to resample data
# scale_factor = 10./3
# img_resample(image_path, scale_factor, resample_path)



# read in label data
pts_gdf = gpd.read_file(label_path)
pts_gdf['type_class'] = pts_gdf['type'].apply(lambda x: 1 if x=='juncus' else 2)

# read in image data
image_src = rasterio.open(resample_path)


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
pts_gdf['B9'] = pts_gdf.apply(lambda x: x['Raster_Value'][7], axis=1)
pts_gdf['B10'] = pts_gdf.apply(lambda x: x['Raster_Value'][7], axis=1)
pts_gdf['B11'] = pts_gdf.apply(lambda x: x['Raster_Value'][7], axis=1)
pts_gdf['B12'] = pts_gdf.apply(lambda x: x['Raster_Value'][7], axis=1)

# deleting not use columns
pts_gdf.drop(columns=['Raster_Value'], inplace=True)
pts_gdf.drop(columns=['rand_point'], inplace=True)

pre_column = ['id', 'type', 'layer', 'path', 'geometry', 'type_class']
bands_columns = ['annual_mean_VV', 'annual_SD_VV', 'winter_VV', 'spring_VV', 'summer_VV', 'fall_VV',
               'annual_mean_VH', 'annual_SD_VH', 'winter_VH', 'spring_VH', 'summer_VH', 'fall_VH']

pts_gdf.columns = pre_column + bands_columns

# pts_gdf.to_file(os.path.join(root_dir, 'data/processing_data/vectors/points_planet_composite.geojson'), driver='GeoJSON')

pts_gdf.to_file(os.path.join(root_dir, 'data/processing_data/vectors/points_sar_extraction.geojson'), driver='GeoJSON')


# # data visualization
# df = pd.DataFrame(pts_gdf.drop(columns='geometry'))
# df.drop(columns=['type_class'], inplace=True)

# # grouped_df = df.groupby('type').agg({'B1': ['mean'], 'B2': ['mean'], 'B3': ['mean'],
# #                               'B4': ['mean'], 'B5': ['mean'], 'B6': ['mean'],
# #                                'B7': ['mean'], 'B8': ['mean']})

# grouped_df = df.groupby('type').mean()
# grouped_df = grouped_df.T
# index = pd.Index()
# grouped_df = grouped_df.set_index(bands_columns)
# grouped_df.plot()

# plt.show()







