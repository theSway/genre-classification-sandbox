import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def produce_smoted():

    sample_map = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500,
                  6: 500, 7: 500, 8: 500, 9: 500, 10: 500}

    # Read data
    X = pd.read_csv('train_data.csv', header=None)
    y = pd.read_csv('train_labels.csv', header=None)

    # Combine for shuffling and partitioning
    data = pd.concat([y, X], axis='columns', ignore_index=True)

    # Shuffle for more reliable validation later
    data = data.sample(frac=1).reset_index(drop=True)

    # Let's partition. 1st part is used to train with SMOTE, 2nd (smaller) part is used to validate
    train, test = train_test_split(data, test_size=0.3, random_state=0)

    # Find x & y
    x_train = train.drop(labels=0, axis='columns')
    y_train = train[[0]]

    x_test = test.drop(labels=0, axis='columns')
    y_test = test[[0]]

    # Let's try SMOTE
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(x_train, y_train)
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)

    training_data = pd.concat([y_resampled, X_resampled],
                              axis='columns', ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Rhythm patterns
    rhythm = training_data.iloc[:, 1:169].copy()

    # Chroma
    chroma_cleaned = training_data.iloc[:, 169:205]

    # MFCCs
    mfcc_cleaned = training_data.iloc[:, 221:265].copy()

    cleaned_x_training = pd.concat([rhythm, chroma_cleaned, mfcc_cleaned],
                                   axis='columns',
                                   ignore_index=True)

    # Outlier detection
    threshold = 3
    for col in range(cleaned_x_training.shape[1]):
        mean = np.mean(cleaned_x_training.iloc[:, col])
        z = np.abs(stats.zscore(cleaned_x_training.iloc[:, col]))
        rows = np.where(z > threshold)
        for row in rows:
            cleaned_x_training.at[row, col] = mean

    # Scaling
    scaler = PowerTransformer()
    scaled_data = scaler.fit_transform(cleaned_x_training)
    scaled_x_training = pd.DataFrame(scaled_data)

    # NOW SAME OPERATIONS FOR VALIDATION DATA
    validation_data = pd.concat([y_test, x_test],
                                axis='columns', ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Rhythm patterns
    rhythm = validation_data.iloc[:, 1:169].copy()

    # Chroma
    chroma_cleaned = validation_data.iloc[:, 169:193]

    # MFCCs
    mfcc_cleaned = validation_data.iloc[:, 221:265].copy()

    cleaned_x_validation = pd.concat([rhythm, chroma_cleaned, mfcc_cleaned],
                                     axis='columns',
                                     ignore_index=True)

    # Outlier detection
    threshold = 3
    for col_val in range(cleaned_x_validation.shape[1]):
        mean_val = np.mean(cleaned_x_validation.iloc[:, col_val])
        z_val = np.abs(stats.zscore(cleaned_x_validation.iloc[:, col_val]))
        rows_val = np.where(z_val > threshold)
        for row_val in rows_val:
            cleaned_x_validation.at[row_val, col_val] = mean_val

    # Scaling
    scaler = PowerTransformer()
    scaled_data = scaler.fit_transform(cleaned_x_validation)
    scaled_x_validation = pd.DataFrame(scaled_data)


    return scaled_x_training, scaled_x_validation, training_data[[0]], validation_data[[0]]