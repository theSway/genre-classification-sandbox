'''

Algorithms based on bagging and boosting

'''

# Importing needed packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn

# Read datasets
df_data = pd.read_csv('train_data.csv', header=None)
df_labels = pd.read_csv('train_labels.csv', header=None)
data = pd.concat([df_labels, df_data], axis='columns', ignore_index=True)

# Rhythm (for inspection)
rhythm = data.iloc[:, 1:169].copy()

# Chroma (for inspection)
chroma = data.iloc[:, 169:217].copy()
chroma_cleaned = data.iloc[:, 169:193]

# MFCCs (for inspection)
mfcc = data.iloc[:, 217:265].copy()
mfcc_cleaned = data.iloc[:, 221:265].copy()

# Seems we want to strip 4 first columns of MFCC and 24 last of chroma
cleaned_x = pd.concat([rhythm, chroma_cleaned, mfcc_cleaned],
                      axis='columns',
                      ignore_index=True)

# Outlier detection
threshold = 3
for col in range(cleaned_x.shape[1]):
    mean = np.mean(cleaned_x.iloc[:, col])
    z = np.abs(stats.zscore(cleaned_x.iloc[:, col]))
    rows = np.where(z > threshold)
    for row in rows:
        cleaned_x.at[row, col] = mean

# Scaling
scaler = PowerTransformer()
scaled_data = scaler.fit_transform(cleaned_x)
scaled_df = pd.DataFrame(scaled_data)

cleaned_data = pd.concat([df_labels, scaled_df], axis='columns', ignore_index=True)

# Split dataset into train and test
train, test = train_test_split(cleaned_data, test_size=0.3, random_state=0)

x_train = train.drop(labels=0, axis='columns')
y_train = np.ravel(train[[0]])

x_test = test.drop(labels=0, axis='columns')
y_test = np.ravel(test[[0]])

# Build a bagging meta-estimator: best result was ~57 % with 1000 estimators and 10 max features
bagged_tree_model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1),
                                      n_estimators=1000,
                                      max_features=10)
bagfit = bagged_tree_model.fit(x_train, y_train)
bagscore = bagged_tree_model.score(x_test, y_test)
print('Bagged tree model DONE: ', bagscore)


# Build a random forest: best result was ~61.8 % with 1000 estimators
forest_model = RandomForestClassifier(random_state=1, n_estimators=1000, n_jobs=-1)
fit_forest = forest_model.fit(x_train, y_train)
forest_score = forest_model.score(x_test, y_test, probability=True)
print('Random Forest Classifier DONE: ', forest_score)

# Build an XGBoosted classifier: 60.9 % with learning rate 0.01
xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
fit_xgb = xgb_model.fit(x_train, y_train)
xgb_score = xgb_model.score(x_test, y_test)
print('XGBoost DONE: ', xgb_score)

# Voting Classifier
svm_clf = SVC()

voting_classifier = VotingClassifier(estimators=[('svc', svm_clf),
                                                 ('Bagged trees', bagged_tree_model),
                                                 ('Random Forest', forest_model),
                                                 ('XGBoost', xgb_model)],
                                     voting='soft')

voted_fit = voting_classifier.fit(x_train, y_train)
voted_score = voting_classifier.score(x_test, y_test)
print('Voting DONE: ', voted_score)