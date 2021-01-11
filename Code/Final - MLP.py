# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:45:24 2020

Combine The Feature and train model. Using Hailey's new features. 
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
from sklearn.metrics import roc_curve, auc
#%% Load data for training and test

train_and_test_pos = pd.read_csv("pos50k.csv")
train_and_test_neg = pd.read_csv("neg50k.csv")
pos = []
neg = []


for i in range(0,len(train_and_test_pos)):
    pos.append(1)

for i in range(0,len(train_and_test_neg)):
    neg.append(0)

train_and_test_pos['indicator_link'] = pos
train_and_test_neg['indicator_link'] = neg

test_public_features = pd.read_csv("test2k.csv")

# Merge the pos and neg
train_and_test_features = train_and_test_pos.append(train_and_test_neg,ignore_index=True)

# Sort the column name
train_and_test_features = train_and_test_features.reindex(sorted(train_and_test_features.columns), axis=1)
test_public_features = test_public_features.reindex(sorted(test_public_features.columns), axis=1)

# Split the train and test dataset
from sklearn.model_selection import train_test_split
train_features, test_features = train_test_split(train_and_test_features, test_size=0.2)


# %% Drop source, sink, label

train_labels = train_features['indicator_link']
test_labels = test_features['indicator_link']
train_features = train_features.drop(['indicator_link'], axis = 1)
test_features = test_features.drop(['indicator_link'], axis = 1)
#test_public_features = test_public_features.drop(['source_node', 'destination_node'], axis = 1)

# %% Normalize by StandardScaler


#s_scaler = preprocessing.StandardScaler()
s_scaler = preprocessing.RobustScaler() #This could remove the infect of outlier


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
# %% Train MLP model
trainStartTime = time.time()
#mlp = MLPClassifier(hidden_layer_sizes=(20,10), activation = 'logistic' ,solver='adam', learning_rate_init=0.005, learning_rate = 'constant')
#mlp = MLPClassifier(hidden_layer_sizes=(100,50,25,10,5), activation = 'logistic' ,solver='adam', learning_rate_init=0.000005, learning_rate = 'constant', alpha=1)
#mlp = MLPClassifier(hidden_layer_sizes=(20,10,4,2), activation = 'tanh' ,solver='sgd', learning_rate_init=0.0005, learning_rate = 'constant', alpha=0.005)
#mlp = MLPClassifier(hidden_layer_sizes=(10,20,10), activation = 'identity' ,solver='sgd', learning_rate_init=0.000005, learning_rate = 'constant', alpha=0.0001)
mlp = MLPClassifier(hidden_layer_sizes=(20,10), activation = 'relu' ,solver='adam', learning_rate_init=0.005, learning_rate = 'constant', alpha=0.005)
y_class = [0,1] #the possible outcome

epochs = 1000

train_score_curve = []
test_score_curve = []
train_predicted_probability = []
# %%
for i in range(epochs):
    mlp.partial_fit(train_features, train_labels, y_class)
    print("========== Step " + str(i) + " ==========")
    #train_predicted_probability = mlp.predict_proba(train_features)
    
    train_score = mlp.score(train_features, train_labels) 
    train_score_curve.append(train_score)
    print("Train score = " + str(train_score))
    # See the score on unseen valid dataset
    test_score = mlp.score(test_features, test_labels)
    test_score_curve.append(test_score)
    print("Valid score = " + str(test_score))
    test_predicted_probability = mlp.predict_proba(test_features)
    fpr,tpr, thresholds = roc_curve(test_labels, test_predicted_probability[:,1])
    auc_sc = auc(fpr, tpr)
    print("Valid AUC = " + str(auc_sc))
    
    #test_predicted_probability = mlp.predict_proba(test_features)
    #print("predict = " + str(test_predicted_probability))

trainFinishTime = time.time()
print("Time spent on training is: " + str(trainFinishTime - trainStartTime) + " sec")

# %% Predict on the test set and get the probability
test_predicted_probability = mlp.predict_proba(test_features)
print("predict = " + str(test_predicted_probability))

# %% Predict on test-public features
test_public_predicted_probability = mlp.predict_proba(test_public_features)
test_public_predicted_label = mlp.predict(test_public_features)


count_true_link = 0
count_fake_link = 0
for instance in test_public_predicted_label:
    if instance == 0:
        count_fake_link = count_fake_link + 1
    if instance == 1:
        count_true_link = count_true_link + 1
print("#True = " + str(count_true_link))
print("#Fake = " + str(count_fake_link))

    
#a = mlp.predict(test_public_features)

df_test_public_result = pd.DataFrame(columns=['Id', 'Predicted'])
id_list = []
for i in range(1,2001):
        id_list.append(i)

df_test_public_result['Id'] = id_list
df_test_public_result['Predicted'] = test_public_predicted_probability[:,1]

# save to csv
df_test_public_result.to_csv('resultMLP3.csv', index=False)


# %% ROC Curve & AUC
fpr,tpr, thresholds = roc_curve(test_labels, test_predicted_probability[:,1])
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()


# %% Plot training and valid curve

plt.close('all')
x_axis = []
for i in range(epochs):
    x_axis.append(i+1)

plt.figure()
plt.plot(x_axis, train_score_curve, "r", x_axis, test_score_curve, "b")
plt.legend(labels=['train','valid'],loc='best')
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
#plt.plot(maxValidAccStep, maxValidAcc, marker='x', markersize=5, color="Green")
#plt.annotate(str(maxValidAccP)+"%",xy=(maxValidAccStep, maxValidAcc))

#plt.savefig('train and valid curve.png', dpi=600)

#%% see the weight of mlp model
# print(mlp.coefs_)