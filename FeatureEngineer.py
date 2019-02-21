'''

For improving features

'''

import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


# ID | Rhythm patterns |   Chroma   |   MFCCs   |
# 1         2 - 169       170 - 217   218 - 265
#
#
#

def create_dataset(partitioning=None):
    # Read datasets
    df_data = pd.read_csv('train_data.csv', header=None)
    df_labels = pd.read_csv('train_labels.csv', header=None)
    data = pd.concat([df_labels, df_data], axis='columns', ignore_index=True)

    # Rhythm patterns summary
    rhythm = data.iloc[:, 1:169].copy()

    # Chroma summary
    chroma = data.iloc[:, 169:217].copy()
    chroma_cleaned = data.iloc[:, 169:193]

    # MFCCs summary
    mfcc = data.iloc[:, 217:265].copy()
    mfcc_cleaned = data.iloc[:, 221:265].copy()

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
    #scaler = MinMaxScaler()
    scaler = PowerTransformer()
    scaled_data = scaler.fit_transform(cleaned_x)
    scaled_df = pd.DataFrame(scaled_data)

    if partitioning == 'classes':

        rhythm_set = pd.concat([df_labels, scaled_df.iloc[:, 0:168]], axis='columns', ignore_index=True)
        chroma_set = pd.concat([df_labels, scaled_df.iloc[:, 168:192]], axis='columns', ignore_index=True)
        mfcc_set = pd.concat([df_labels, scaled_df.iloc[:, 192:236]], axis='columns', ignore_index=True)

        return [rhythm_set, chroma_set, mfcc_set]

    elif partitioning == 'statistics':

        # Rhythm set includes 7 statistics
        set1 = pd.concat([df_labels, scaled_df.iloc[:, 0:24]], axis='columns', ignore_index=True)
        set2 = pd.concat([df_labels, scaled_df.iloc[:, 24:48]], axis='columns', ignore_index=True)
        set3 = pd.concat([df_labels, scaled_df.iloc[:, 48:72]], axis='columns', ignore_index=True)
        set4 = pd.concat([df_labels, scaled_df.iloc[:, 72:96]], axis='columns', ignore_index=True)
        set5 = pd.concat([df_labels, scaled_df.iloc[:, 96:120]], axis='columns', ignore_index=True)
        set6 = pd.concat([df_labels, scaled_df.iloc[:, 120:144]], axis='columns', ignore_index=True)
        set7 = pd.concat([df_labels, scaled_df.iloc[:, 144:168]], axis='columns', ignore_index=True)

        # Chroma set includes 4 statistics but 2 are cleaned away
        set8 = pd.concat([df_labels, scaled_df.iloc[:, 168:180]], axis='columns', ignore_index=True)
        set9 = pd.concat([df_labels, scaled_df.iloc[:, 180:192]], axis='columns', ignore_index=True)

        # MFCC set includes 4 statistics. Last feature set is shorter due to cleaning
        set10 = pd.concat([df_labels, scaled_df.iloc[:, 192:200]], axis='columns', ignore_index=True)
        set11 = pd.concat([df_labels, scaled_df.iloc[:, 200:212]], axis='columns', ignore_index=True)
        set12 = pd.concat([df_labels, scaled_df.iloc[:, 212:224]], axis='columns', ignore_index=True)
        set13 = pd.concat([df_labels, scaled_df.iloc[:, 224:236]], axis='columns', ignore_index=True)

        return [set1, set2, set3, set4, set5,
                set6, set7, set8, set9, set10,
                set11, set12, set13]
    else:
        return pd.concat([df_labels, scaled_df], axis='columns', ignore_index=True)