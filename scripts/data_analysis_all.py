
import os
from os.path import dirname as up

# data processing
import geopandas as gpd
import numpy as np

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

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

# Loading data
root_dir = up(os.getcwd())
points_data = os.path.join(root_dir, 'data/processing_data/vectors/points_planet.geojson')

gdf = gpd.read_file(points_data)
gdf['ndvi'] = gdf.apply(lambda x: (x['B8'] - x['B6']) / (x['B8'] + x['B6']), axis=1)

# feature selection
X_data = gdf[['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'ndvi']]
y_data = gdf['type_class']
# scaler
scaler = StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

# split data without ndvi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.20, random_state=42, shuffle=True) # stratify = y_data.ravel()

# parameter searching


# K-NNC
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
# Predict the labels of test data
knn_pred = knn.predict(X_test)
print(f"Accuracy with K-NNC: {accuracy_score(y_test, knn_pred)*100}")
print(classification_report(y_test, knn_pred))


# SVM
svm = SVC(C=3.0, kernel='rbf', degree=6, cache_size=1024)
# Fit Data
svm.fit(X_train, y_train)
# Predict labels for test data
svm_pred = svm.predict(X_test)
# Accuracy and Classification Reeport
print(f"Accuracy with SVM: {accuracy_score(y_test, svm_pred)*100}")
print(classification_report(y_test, svm_pred))


# LightGBM
"""
-> Figure out parameters
"""
d_train = lgb.Dataset(X_train, label=y_train)
# Parameters
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=15
params['num_class']=6 #no.of unique values in the target class not inclusive of the end value

clf = lgb.train(params, d_train, 100)

# prediction
lgb_predictions = clf.predict(X_test)
lgb_pred = np.argmax(lgb_predictions, axis=1)

# Accuracy and Classification Report
print(f"Accuracy: {accuracy_score(y_test, lgb_pred)*100}")
print(classification_report(y_test, lgb_pred))

# decision tree


# random forest


# xgboost



