

import os
from os.path import dirname as up

# data processing
import geopandas as gpd

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


root_dir = up(os.getcwd())
points_data = os.path.join(root_dir, 'data/processing_data/vectors/points_planet_composite.geojson')

gdf = gpd.read_file(points_data)

# Initial anlaysis without data normalization and NDVI
X_data = gdf[['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']]
y_data = gdf['type_class']

# scaler without ndvi
scaler = StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

# split data without ndvi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.30, stratify = y_data.ravel(), random_state=42)

# K-NNC without NDVI
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

