'''

Run everything here.

'''
from FeatureEngineer import create_dataset
from find_best import support_vector_classifier, decision_tree_classifier, naive_bayes, knn_classifier, neural_network
from ensembles import onevsrest_classifier, voter, xgboosted
from ensemble_of_ensembles import ensemblevoter
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import numpy as np

# Search best estimators from the entire feature space

data = create_dataset()

# Split dataset into train and test
train, test = train_test_split(data, test_size=0.3, random_state=0)

x_train = train.drop(labels=0, axis='columns')
y_train = np.ravel(train[[0]])

x_test = test.drop(labels=0, axis='columns')
y_test = np.ravel(test[[0]])

# Find out how good individual classifiers are
print('Finding best SVC...')
best_svc, svc_score = support_vector_classifier(x_train, y_train, x_test, y_test)
print('Finding best DT...')
best_decision_tree, tree_score = decision_tree_classifier(x_train, y_train, x_test, y_test)
print('Finding best Naive Bayes...')
best_bayes, bayes_score = naive_bayes(x_train, y_train, x_test, y_test)
'''
print('Finding best Random Forest...')
best_forest, forest_score = random_forest(x_train, y_train, x_test, y_test)
'''
print('Finding best KNN...')
best_knn, knn_score = knn_classifier(data, x_train, y_train, x_test, y_test)
print('Finding best Neural Network...')
best_nn, nn_score = neural_network(x_train, y_train, x_test, y_test)

print('SVC: ', svc_score,
      '\nDT: ', tree_score,
      '\nBayes: ', bayes_score,
      '\nKNN: ', knn_score,
      '\nNN: ', nn_score)

# Save these to files for later use
joblib.dump(best_svc, './models/ALL_svc.joblib')
joblib.dump(best_decision_tree, './models/ALL_tree.joblib')
joblib.dump(best_bayes, './models/ALL_bayes.joblib')
joblib.dump(best_knn, './models/ALL_knn.joblib')
joblib.dump(best_nn, './models/ALL_nn.joblib')

# Find out how ensembles of these are (plus one XGB)
print('Finding best OVR...')
best_ovr, ovr_score = onevsrest_classifier(best_svc, best_decision_tree, best_bayes, best_nn, x_train, y_train, x_test, y_test)
print('Finding best Voter...')
best_voter, voter_score = voter(best_svc, best_decision_tree, best_bayes, best_nn, x_train, y_train, x_test, y_test)
print('Finding best XGB...')
best_xgb, xgb_score = xgboosted(x_train, y_train, x_test, y_test)

print('OVR: ', ovr_score,
      '\nVoter: ', voter_score,
      '\nXGB: ', xgb_score)

# Save ensembles to files for later use
joblib.dump(best_ovr, './models/ALL_ENS_ovr.joblib')
joblib.dump(best_voter, './models/ALL_ENS_oter.joblib')
joblib.dump(best_xgb, './models/ALL_ENS_xgb.joblib')

# Find out if the performance can be improved by voting on ensembles
print('Finding METAENSEMBLE...')
best_metaensemble, meta_score = ensemblevoter(best_xgb, best_voter, best_ovr, x_train, y_train, x_test, y_test)

print('Meta Ensemble score: ', meta_score)

# Save metaensemble to file for later use
joblib.dump(best_ovr, './models/ALL_META_voter.joblib')