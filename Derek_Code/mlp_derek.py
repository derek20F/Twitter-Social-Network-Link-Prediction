# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:45:24 2020

Combine The Feature and train model

@author: Derek
"""

# %% Data prepare
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report




train_features = pd.read_csv("feature/2nd_features_train.csv", header=None)

test_features = pd.read_csv("feature/2nd_features_test.csv", header=None)

test_public_features = pd.read_csv("feature/features_test_public.csv", header=None)

test_public_features = test_public_features.drop([0,1], axis = 1)

train_labels = train_features[2]
test_labels = test_features[2]
train_features = train_features.drop([0,1,2], axis = 1)
test_features = test_features.drop([0,1,2], axis = 1)
# %% Normalize


train_features = ( train_features-train_features.min() )/( train_features.max()-train_features.min() )
test_features = ( test_features-test_features.min() )/( test_features.max()-test_features.min() )
test_public_features = ( test_public_features-test_public_features.min() )/( test_public_features.max()-test_public_features.min() )




# %% Train MLP model
trainStartTime = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(10,4,2), activation = 'logistic' ,solver='adam', learning_rate_init=0.005, learning_rate = 'constant')

y_class = [0,1] #the possible outcome

epochs = 100

train_score_curve = []
test_score_curve = []
train_predicted_probability = []

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
    
    #test_predicted_probability = mlp.predict_proba(test_features)
    #print("predict = " + str(test_predicted_probability))

trainFinishTime = time.time()
print("Time spent on training is: " + str(trainFinishTime - trainStartTime) + " sec")

# %% Predict on the test set and get the probability
test_predicted_probability = mlp.predict_proba(test_features)
print("predict = " + str(test_predicted_probability))

# %% Predict on test-public features
test_public_predicted_probability = mlp.predict_proba(test_public_features)
#a = mlp.predict(test_public_features)

df_test_public_result = pd.DataFrame(columns=['Id', 'Predicted'])
id_list = []
for i in range(1,2001):
        id_list.append(i)

df_test_public_result['Id'] = id_list
df_test_public_result['Predicted'] = test_public_predicted_probability[:,1]

# save to csv
df_test_public_result.to_csv('result.csv', index=False)


# %% ROC Curve & AUC
from sklearn.metrics import roc_curve, auc
fpr,tpr, thresholds = roc_curve(test_labels, test_predicted_probability[:,1])
auc_sc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='navy',label='ROC curve (area = %0.2f)' % auc_sc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic with test data')
plt.legend()
plt.show()

#%% Test on the unlabled testing dataset
