# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 02:06:27 2020

@author: Chen-An Fan
Feature Inspection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load data for training and test
data_columns_list = ['source_node', 'destination_node','indicator_link','common_neighbours', \
               'shortest_path', 'ccpa_front', 'ccpa_back', 'ccpa_0.5', 'ccpa_0.6', 'ccpa_0.7', \
               'ccpa_0.8', 'ccpa_0.9', 'num_followers_s', 'num_followers_d','num_followees_s', \
                'num_followees_d', 'inter_followers', 'inter_followees', 'cnd']

test_columns_list = ['source_node', 'destination_node','common_neighbours', \
               'shortest_path', 'ccpa_front', 'ccpa_back', 'ccpa_0.5', 'ccpa_0.6', 'ccpa_0.7', \
               'ccpa_0.8', 'ccpa_0.9', 'num_followers_s', 'num_followers_d','num_followees_s', \
                'num_followees_d', 'inter_followers', 'inter_followees', 'cnd']



train_and_test_pos1 = pd.read_csv('features_train200k.csv', names = data_columns_list)
train_and_test_pos2 = pd.read_csv('features_train500k.csv', names = data_columns_list)
train_and_test_neg = pd.read_csv("features_train500k-neg.csv", names = data_columns_list)

train_and_test_pos = pd.concat([train_and_test_pos1, train_and_test_pos2])

test_public_features = pd.read_csv("features_test_public_500k.csv", names = test_columns_list)




train_and_test_pos = train_and_test_pos.drop(['source_node', 'destination_node','indicator_link', \
                'ccpa_front', 'ccpa_back', 'ccpa_0.6', 'ccpa_0.7', \
               'ccpa_0.8', 'ccpa_0.9', 'num_followers_s', 'num_followers_d','num_followees_s', \
                'num_followees_d', 'inter_followers', 'inter_followees'], axis = 1)

train_and_test_neg = train_and_test_neg.drop(['source_node', 'destination_node','indicator_link', \
               'ccpa_front', 'ccpa_back', 'ccpa_0.6', 'ccpa_0.7', \
               'ccpa_0.8', 'ccpa_0.9', 'num_followers_s', 'num_followers_d','num_followees_s', \
                'num_followees_d', 'inter_followers', 'inter_followees'], axis = 1)

train_and_test_pos = train_and_test_pos.replace([np.inf],1000000)
train_and_test_neg = train_and_test_neg.replace([np.inf],1000000)

#%% Plot the density distribution of common neighbours
df_cn = pd.DataFrame(columns = ['pos_CN', 'neg_CN'])
df_cn['pos_CN'] = train_and_test_pos['common_neighbours']
df_cn['neg_CN'] = train_and_test_neg['common_neighbours']

df_cn.plot.kde(title='Common Neighbours', logy=True)

#%% Plot the density distribution of shortest path
df_sp = pd.DataFrame(columns = ['pos_SP', 'neg_SP'])
df_sp['pos_SP'] = train_and_test_pos['shortest_path']
df_sp['neg_SP'] = train_and_test_neg['shortest_path']

df_sp.plot.kde(title='Shortest Path',logy=True)

#%% Plot the density distribution of ccpa_0.5
df_ccpa = pd.DataFrame(columns = ['pos_ccpa', 'neg_ccpa'])
df_ccpa['pos_ccpa'] = train_and_test_pos['ccpa_0.5']
df_ccpa['neg_ccpa'] = train_and_test_neg['ccpa_0.5']

df_ccpa.plot.kde(title='ccpa', logy=True)

#%% Plot the density distribution of cnd
df_cnd = pd.DataFrame(columns = ['pos_cnd', 'neg_cnd'])
df_cnd['pos_cnd'] = train_and_test_pos['cnd']
df_cnd['neg_cnd'] = train_and_test_neg['cnd']

df_cnd.plot.kde(title='cnd', logy=True)

#df_cnd.plot(kind='area',title='cnd')

