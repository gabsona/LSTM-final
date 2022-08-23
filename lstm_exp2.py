import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()
tf.compat.v1.random.set_random_seed(1234)

from data_engineering import *
from feature_engineering import *


df = download_data('NFLX',start_date = '2018-01-01',end_date ='2022-01-01', interval='1d')

# print(np.dtype(df.loc[:, 'Close']))
# print(np.dtype(df.iloc[:, 4:5]))


minmax = MinMaxScaler().fit(df.iloc[:, 4:5].values.reshape(-1,1)) # Close index
df_log = minmax.transform(df.iloc[:, 4:5]) # Close index
df_log = pd.DataFrame(df_log)
print(df_log.head())