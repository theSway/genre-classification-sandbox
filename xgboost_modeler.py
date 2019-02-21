import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.externals import joblib
from FeatureEngineer import create_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


# Load data
data = create_dataset()

# Split dataset into train and test
train, test = train_test_split(data, test_size=0.3, random_state=0)

x_train = train.drop(labels=0, axis='columns')
y_train = np.ravel(train[[0]])

x_test = test.drop(labels=0, axis='columns')
y_test = np.ravel(test[[0]])

'''
xgb1 = XGBClassifier(learning_rate=0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='multi:softmax',
                     nthread=4,
                     scale_pos_weight=1,
                     num_class=10,
                     seed=27)

xgb1_fit = xgb1.fit(x_train, y_train)
xgb1_score = xgb1_fit.score(x_test, y_test)
'''
# Test 1
'''
param_test1 = {'max_depth': range(3, 10, 2),
               'min_child_weight': range(1, 6, 2)
               }
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test1, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch1.fit(x_train, y_train)
print(gsearch1.best_params_, gsearch1.best_score_)
'''
# Test 2 (see results of Test 1)
'''
param_test2 = {'max_depth': [6, 7, 8],
               'min_child_weight': [4, 5, 6]
               }
gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test2, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch2.fit(x_train, y_train)
print(gsearch2.best_params_, gsearch2.best_score_)
'''
# Test 3
'''
param_test3 = {'gamma': [i/10.0 for i in range(0, 5)]}
gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=7,
                                                min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test3, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch3.fit(x_train, y_train)
print(gsearch3.best_params_, gsearch3.best_score_)
'''
# See performance
'''
xgb2 = XGBClassifier(learning_rate=0.1,
                     n_estimators=1000,
                     max_depth=7,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective='multi:softmax',
                     nthread=4,
                     scale_pos_weight=1,
                     num_class=10,
                     seed=27)

xgb2_fit = xgb2.fit(x_train, y_train)
xgb2_score = xgb2_fit.score(x_test, y_test)
'''
# Test 4
'''
param_test4 = {'subsample': [i/10.0 for i in range(6, 10)],
               'colsample_bytree': [i/10.0 for i in range(6, 10)]}
gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=7,
                                                min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test4, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch4.fit(x_train, y_train)
print(gsearch4.best_params_, gsearch4.best_score_)
'''
# Test 5
'''
param_test5 = {'subsample': [i/100.0 for i in range(65, 80, 5)],
               'colsample_bytree': [i/100.0 for i in range(55, 70, 5)]}
gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=7,
                                                min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test5, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch5.fit(x_train, y_train)
print(gsearch5.best_params_, gsearch5.best_score_)
'''
# Test 6
'''
param_test6 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=7,
                                                min_child_weight=5, gamma=0.1, subsample=0.65, colsample_bytree=0.6,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test6, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch6.fit(x_train, y_train)
print(gsearch6.best_params_, gsearch6.best_score_)
'''
# Test 7
'''
param_test7 = {'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]}
gsearch7 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=7,
                                                min_child_weight=5, gamma=0.1, subsample=0.65, colsample_bytree=0.6,
                                                objective='multi:softmax', num_class=10, nthread=4, scale_pos_weight=1,
                                                seed=27, reg_alpha=0.00001),
                        param_grid=param_test7, scoring='accuracy', n_jobs=-1, iid=False, cv=5)

gsearch7.fit(x_train, y_train)
print(gsearch7.best_params_, gsearch7.best_score_)
'''
# See performance
'''
xgb3 = XGBClassifier(learning_rate=0.1,
                     n_estimators=1000,
                     max_depth=7,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.65,
                     colsample_bytree=0.6,
                     objective='multi:softmax',
                     nthread=4,
                     scale_pos_weight=1,
                     num_class=10,
                     seed=27,
                     reg_alpha=0.00001,
                     reg_lambda=1)

xgb3_fit = xgb3.fit(x_train, y_train)
xgb3_score = xgb3_fit.score(x_test, y_test)
'''
# Add trees
xgb4 = XGBClassifier(learning_rate=0.01,
                     n_estimators=5000,
                     max_depth=7,
                     min_child_weight=5,
                     gamma=0.1,
                     subsample=0.65,
                     colsample_bytree=0.6,
                     objective='multi:softmax',
                     nthread=4,
                     scale_pos_weight=1,
                     num_class=10,
                     seed=27,
                     reg_alpha=0.00001,
                     reg_lambda=1)

xgb4_fit = xgb4.fit(x_train, y_train)
xgb4_score = xgb4_fit.score(x_test, y_test)
# Aaaaand save
joblib.dump(xgb4_fit, './models/optimized_xgb.joblib')

