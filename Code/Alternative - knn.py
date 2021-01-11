# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:45:24 2020
K-Nearest Neighbours
@author: Chen-An Fan
"""

# %% Data prepare
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing


# Load Feature Data
train_and_test_features = pd.read_csv("new_new_feature/new_features_train.csv", header=None)
test_public_features = pd.read_csv("new_new_feature/new_features_test_public.csv", header=None)

features_list=['source_node', 'destination_node','indicator_link', 'common_neighbours', 'shortest_path',
       'ccpa_front', 'ccpa_back', 'ccpa_0.5', 'ccpa_0.6', 'ccpa_0.7',
       'ccpa_0.8', 'ccpa_0.9', 'cnd']
features_list2=['source_node', 'destination_node', 'common_neighbours', 'shortest_path',
       'ccpa_front', 'ccpa_back', 'ccpa_0.5', 'ccpa_0.6', 'ccpa_0.7',
       'ccpa_0.8', 'ccpa_0.9', 'cnd']

train_and_test_features.columns = features_list
test_public_features.columns = features_list2

# %% Treatment on shortest_path

train_and_test_features.loc[train_and_test_features['shortest_path'] == -1, 'shortest_path'] = 100
test_public_features.loc[test_public_features['shortest_path'] == -1, 'shortest_path'] = 100


# %%Split the train and test dataset
from sklearn.model_selection import train_test_split
train_features, test_features = train_test_split(train_and_test_features, test_size=0.2, random_state=9)


# %% Drop source, sink, label

train_labels = train_features['indicator_link']
test_labels = test_features['indicator_link']
train_features = train_features.drop(['source_node', 'destination_node','indicator_link'], axis = 1)
test_features = test_features.drop(['source_node', 'destination_node','indicator_link'], axis = 1)
test_public_features = test_public_features.drop(['source_node', 'destination_node'], axis = 1)

# %% Normalize by StandardScaler
### With or without normalization, makes no difference......

s_scaler = preprocessing.StandardScaler()

train_features = s_scaler.fit_transform(train_features)
train_features = pd.DataFrame(train_features)


test_features = s_scaler.fit_transform(test_features)
test_features = pd.DataFrame(test_features)

test_public_features = s_scaler.fit_transform(test_public_features)
test_public_features = pd.DataFrame(test_public_features)


'''
train_features = ( train_features-train_features.min() )/( train_features.max()-train_features.min() )
test_features = ( test_features-test_features.min() )/( test_features.max()-test_features.min() )
test_public_features = ( test_public_features-test_public_features.min() )/( test_public_features.max()-test_public_features.min() )
'''
# %% Find the parameters for MLP
'''
from sklearn.model_selection import GridSearchCV

MLPClassifier().get_params()
mlp_param_grid={
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }

search = GridSearchCV(MLPClassifier(max_iter=100), param_grid=mlp_param_grid, n_jobs=-1, cv=5)
search.fit(train_features, train_labels)

print('Best parameters found:\n', search.best_params_)
'''
#%% Train KNN model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
model.fit(train_features, train_labels)
test_probability = model.predict_proba(test_features)
accuracy = model.score(test_features, test_labels)
print(accuracy)
#%%
print("Start predicting...")
test_public_label = model.predict(test_public_features)
test_public_probability = model.predict_proba(test_public_features)

count_true_link = 0
count_fake_link = 0
for instance in test_public_label:
    if instance == 0:
        count_fake_link = count_fake_link + 1
    if instance == 1:
        count_true_link = count_true_link + 1
print("#True = " + str(count_true_link))
print("#Fake = " + str(count_fake_link))

df_test_public_result = pd.DataFrame(columns=['Id', 'Predicted'])
id_list = []
for i in range(1,2001):
    id_list.append(i)

df_test_public_result['Id'] = id_list
df_test_public_result['Predicted'] = test_public_probability[:,1]

# save to csv
df_test_public_result.to_csv('result.csv', index=False)


# %% ROC Curve & AUC
from sklearn.metrics import roc_curve, auc
fpr,tpr, thresholds = roc_curve(test_labels, test_probability[:,1])
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()

