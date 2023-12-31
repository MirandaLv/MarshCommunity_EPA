"""
author: Miranda Lv
Date: Sep, 2023
"""

import os
from os.path import dirname as up
import time
import sys

sys.path.append('../utils')

# data processing
import geopandas as gpd
import numpy as np
import pandas as pd
import helpers
import itertools

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# import lightgbm as lgb
import xgboost as xgb
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

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

"""
juncus: 1
alterniflora: 2
"""
# setting filter parameters
year = '2022'
seasons = {'S1': 'Jan-Mar',
           'S2': 'Apr-Jun',
           'S3': 'July-Sep',
           'S4': 'Oct-Dec'}

sar_annual = {'mean': ['annual_mean_VH', 'annual_mean_VV'],
              'sd': ['annual_sd_VH', 'annual_sd_VV']}

var_column = ['ndwi', 'ndvi', 'g_b', 'r_g', 'NIR_r','B7', 'B2', 'B3', 'B4', 'B5', 'B6', 'B8'] #'NIR_g'
all_seasons = ['S1', 'S2', 'S3', 'S4']

sar_season = None

filter_column = ['{}_{}'.format(r[0], r[1]) for r in itertools.product(var_column, all_seasons)] + ['dem']

########################################################################################################################################################################
# Loading data

all_gdf = gpd.GeoDataFrame()
gdfs = []

for key, v in seasons.items():
    root_dir = up(os.getcwd())
    points_data = os.path.join(root_dir,
                               'data/processing_data/vectors/points_planet_comp_{}_{}.geojson'.format(seasons[key],
                                                                                                      year))

    gdf = gpd.read_file(points_data)

    # scaling band values to SR values
    gdf['B1'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B1']), axis=1)
    gdf['B2'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B2']), axis=1)
    gdf['B3'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B3']), axis=1)
    gdf['B4'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B4']), axis=1)
    gdf['B5'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B5']), axis=1)
    gdf['B6'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B6']), axis=1)
    gdf['B7'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B7']), axis=1)
    gdf['B8'] = gdf.apply(lambda x: helpers.scale_to_sr(x['B8']), axis=1)
    # adding three indices
    gdf['ndvi'] = gdf.apply(lambda x: helpers.cal_ndvi(x['B8'], x['B6']), axis=1)
    gdf['savi'] = gdf.apply(lambda x: helpers.cal_savi(x['B8'], x['B6']), axis=1)
    gdf['ndwi'] = gdf.apply(lambda x: helpers.cal_ndwi(x['B8'], x['B4']), axis=1)

    # adding four additional ratio indices from Martha Gilmore et al. (2004)
    gdf['g_b'] = gdf.apply(lambda x: helpers.ratio_indices(x['B4'], x['B2']), axis=1)
    gdf['r_g'] = gdf.apply(lambda x: helpers.ratio_indices(x['B6'], x['B4']), axis=1)
    gdf['NIR_g'] = gdf.apply(lambda x: helpers.ratio_indices(x['B8'], x['B4']), axis=1)
    gdf['NIR_r'] = gdf.apply(lambda x: helpers.ratio_indices(x['B8'], x['B6']), axis=1)

    gdf = gdf.add_suffix('_{}'.format(key))
    gdfs.append(gdf)

merge_gdf = pd.merge(gdfs[0], gdfs[1], left_index=True, right_index=True)
merge_gdf = pd.merge(merge_gdf, gdfs[2], left_index=True, right_index=True)
merge_gdf = pd.merge(merge_gdf, gdfs[3], left_index=True, right_index=True)

# update dem column so remove the repetitive dem column from different seasons
merge_gdf = merge_gdf.rename(columns={"dem_value_S1": "dem"})

########################################################################################################################################################################
# Load sar for testing, need to set up the sar_season variable

sar_data = os.path.join(root_dir, 'data/processing_data/vectors/points_sar_extraction.geojson')
sar_gdf = gpd.read_file(sar_data)

# merge sar data
merge_gdf = pd.merge(merge_gdf, sar_gdf, left_index=True, right_index=True, suffixes=('', '_sar'))

if sar_season != None:
    # getting sar column for the selected season
    sar_column = sar_annual[sar_season]
    filter_column = filter_column + sar_column

########################################################################################################################################################################
# feature selection
X_data = merge_gdf[filter_column]
y_data = merge_gdf['type_class_S1']
# scaler
scaler = StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.30, random_state=42,
                                                    shuffle=True)  # , stratify = y_data.ravel()

"""K-NNC"""
# calculating the accuracy of models with different values of k
knn_mean_acc = np.zeros(20)
for i in range(1, 21):
    # Train Model and Predict
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    yhat = knn.predict(X_test)
    knn_mean_acc[i - 1] = accuracy_score(y_test, yhat) * 100

loc = np.arange(1, 21, step=1.0)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), knn_mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.savefig('../figures/knn_accuracy_seasonal_{}.png'.format(year))

# # parameter searching
# # Grid Search
# # number of neighbors are chosen based on getting the knn_accuracy.png for each season.
# grid_params = {'weights' : ['uniform','distance'],
#                'metric' : ['minkowski','euclidean','manhattan']}
#
# gs = GridSearchCV(KNeighborsClassifier(5), grid_params, verbose = 1, cv=3, n_jobs = -1)
# # fit the model on our train set
# g_res = gs.fit(X_train, y_train)
# # find the best score
# print(g_res.best_score_)

# Train Model and Predict
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
# Predict the labels of test data
knn_pred = knn.predict(X_test)
print(f"Accuracy with K-NNC: {accuracy_score(y_test, knn_pred) * 100}")
print(classification_report(y_test, knn_pred))

# """SVM"""
# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
# grid.fit(X_train,y_train)
#
# # Accuracy and Classification Report
# grid_predictions = grid.predict(X_test)
# print(f"Accuracy with SVM grid search: {accuracy_score(y_test, grid_predictions)*100}")
# print(classification_report(y_test,grid_predictions))#Output


svm = SVC(C=3.0, kernel='rbf', degree=6, cache_size=1024)
# Fit Data
svm.fit(X_train, y_train)
# Predict labels for test data
svm_pred = svm.predict(X_test)
# Accuracy and Classification Report
print(f"Accuracy with SVM: {accuracy_score(y_test, svm_pred) * 100}")
print(classification_report(y_test, svm_pred))

#
# """
# LightGBM
# -> Figure out parameters
# """
# d_train = lgb.Dataset(X_train, label=y_train)
# # Parameters
# params={}
# params['learning_rate']=0.03
# params['boosting_type']='gbdt' #GradientBoostingDecisionTree
# params['objective']='multiclass' #Multi-class target feature
# params['metric']='multi_logloss' #metric for multi-class
# params['max_depth']=15
# params['num_class']=6 #no.of unique values in the target class not inclusive of the end value
#
# clf = lgb.train(params, d_train, 100)
#
# # prediction
# lgb_predictions = clf.predict(X_test)
# lgb_pred = np.argmax(lgb_predictions, axis=1)
#
# # Accuracy and Classification Report
# print(f"Accuracy with lgb: {accuracy_score(y_test, lgb_pred)*100}")
# print(classification_report(y_test, lgb_pred))


"""
Decision Tree
"""
tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)

print(f"Accuracy with Decision Tree: {accuracy_score(y_test, tree_pred) * 100}")
print(classification_report(y_test, tree_pred))

tree.plot_tree(tree_clf)
dot_data = tree.export_graphviz(tree_clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("../figures/DecisionTree_seasonal_{}".format(year))


"""
Random forest
"""

# split data with no scaling for RF
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42,
                                                    shuffle=True)  # , stratify = y_data.ravel()

feature_names = X_data.columns.tolist()
forest = RandomForestClassifier(random_state=0)

rf_clf = forest.fit(X_train, y_train)

# impurity based feature importance
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# getting variable importance
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.head(10).plot.barh(align='center', color="g") #, align='center', color="g"
ax.set_title("Feature importances") # using MDI
ax.set_ylabel("Mean decrease in impurity")
plt.savefig('../figures/rf_VariableImportance_top10.png'.format(year), facecolor=(1, 1, 1))


fig, ax = plt.subplots()
forest_importances.plot.barh(align='center', color="g") #, align='center', color="g"
ax.set_title("Feature importances") # using MDI
ax.set_ylabel("Mean decrease in impurity")
plt.savefig('../figures/rf_VariableImportance_allvariables.png'.format(year), facecolor=(1, 1, 1))
plt.show()

rf_pred = rf_clf.predict(X_test)
print(f"Accuracy with Random Forest: {accuracy_score(y_test, rf_pred) * 100}")
print(classification_report(y_test, rf_pred))



#
# # feature permutation
# result = permutation_importance(
#     forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
#
# forest_importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature Importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.savefig('../figures/rf_featurePerm_seasonal_{}.png'.format(year), facecolor=(1, 1, 1))