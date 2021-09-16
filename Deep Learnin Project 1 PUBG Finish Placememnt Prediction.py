# Load modules
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import gc
import time

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


#feature engineering - finding min, max and average grouped by match ID and group ID
def feature_engineering(train):
    drop_features_temp = ['Id', 'groupId', 'matchId', 'matchType']

    features = list(set(train.columns) - set(drop_features_temp))
    
    #min
    min_group = train.groupby(by=["matchId","groupId"])[features].min()
    min_group_rank = min_group.groupby('matchId')[features].rank(pct=True)
    min_group = min_group.add_suffix('_min')
    min_group_rank = min_group_rank.add_suffix('_min_rank')
    print("added min features")
    #max
    max_group = train.groupby(by=["matchId","groupId"])[features].max()
    max_group_rank = max_group.groupby('matchId')[features].rank(pct=True)
    max_group = max_group.add_suffix('_max')
    max_group_rank = max_group_rank.add_suffix('_max_rank')
    print("added max features")
    #mean
    mean_group = train.groupby(by=["matchId","groupId"])[features].mean()
    mean_group_rank = mean_group.groupby('matchId')[features].rank(pct=True)
    mean_group = mean_group.add_suffix('_mean')
    mean_group_rank = mean_group_rank.add_suffix('_mean_rank')
    print("added mean features")
    
    grouped_train = pd.concat([min_group, min_group_rank, max_group, max_group_rank, mean_group, mean_group_rank], axis=1)
    print("concatenated")
    del min_group, min_group_rank
    del max_group, max_group_rank
    del mean_group, mean_group_rank
    gc.collect()
    
    return grouped_train


#load train data
train_actual = pd.read_csv("../input/train_V2.csv")
train_actual = train_actual.dropna()
print("Number of train data: ", len(train_actual))

#load test data
test_actual = pd.read_csv("../input/test_V2.csv")
print("Number of test data: ", len(test_actual))

#add extra features
train_actual['headshotrate'] = train_actual['kills']/train_actual['headshotKills']
train_actual['headshotrate'].fillna(0, inplace=True)
train_actual['headshotrate'].replace(np.inf, 0, inplace=True)
test_actual['headshotrate'] = test_actual['kills']/test_actual['headshotKills']
test_actual['headshotrate'].fillna(0, inplace=True)
test_actual['headshotrate'].replace(np.inf, 0, inplace=True)

train_actual['killStreakrate'] = train_actual['killStreaks']/train_actual['kills']
train_actual['killStreakrate'].fillna(0, inplace=True)
train_actual['killStreakrate'].replace(np.inf, 0, inplace=True)
test_actual['killStreakrate'] = test_actual['killStreaks']/test_actual['kills']
test_actual['killStreakrate'].fillna(0, inplace=True)
test_actual['killStreakrate'].replace(np.inf, 0, inplace=True)

train_actual['healthitems'] = train_actual['heals'] + train_actual['boosts']
train_actual['healthitems'].fillna(0, inplace=True)
train_actual['healthitems'].replace(np.inf, 0, inplace=True)
test_actual['healthitems'] = test_actual['heals'] + test_actual['boosts']
test_actual['healthitems'].fillna(0, inplace=True)
test_actual['healthitems'].replace(np.inf, 0, inplace=True)

train_actual['totalDistance'] = train_actual['rideDistance'] + train_actual["walkDistance"] + train_actual["swimDistance"]
test_actual['totalDistance'] = test_actual['rideDistance'] + test_actual["walkDistance"] + test_actual["swimDistance"]

train_actual['killPlace_over_maxPlace'] = train_actual['killPlace'] / train_actual['maxPlace']
train_actual['killPlace_over_maxPlace'].fillna(0, inplace=True)
train_actual['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)
test_actual['killPlace_over_maxPlace'] = test_actual['killPlace'] / test_actual['maxPlace']
test_actual['killPlace_over_maxPlace'].fillna(0, inplace=True)
test_actual['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)

train_actual['headshotKills_over_kills'] = train_actual['headshotKills'] / train_actual['kills']
train_actual['headshotKills_over_kills'].fillna(0, inplace=True)
train_actual['headshotKills_over_kills'].replace(np.inf, 0, inplace=True)
test_actual['headshotKills_over_kills'] = test_actual['headshotKills'] / test_actual['kills']
test_actual['headshotKills_over_kills'].fillna(0, inplace=True)
test_actual['headshotKills_over_kills'].replace(np.inf, 0, inplace=True)

train_actual['distance_over_weapons'] = train_actual['totalDistance'] / train_actual['weaponsAcquired']
train_actual['distance_over_weapons'].fillna(0, inplace=True)
train_actual['distance_over_weapons'].replace(np.inf, 0, inplace=True)
test_actual['distance_over_weapons'] = test_actual['totalDistance'] / test_actual['weaponsAcquired']
test_actual['distance_over_weapons'].fillna(0, inplace=True)
test_actual['distance_over_weapons'].replace(np.inf, 0, inplace=True)

train_actual['walkDistance_over_heals'] = train_actual['walkDistance'] / train_actual['heals']
train_actual['walkDistance_over_heals'].fillna(0, inplace=True)
train_actual['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)
test_actual['walkDistance_over_heals'] = test_actual['walkDistance'] / test_actual['heals']
test_actual['walkDistance_over_heals'].fillna(0, inplace=True)
test_actual['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)

train_actual['walkDistance_over_kills'] = train_actual['walkDistance'] / train_actual['kills']
train_actual['walkDistance_over_kills'].fillna(0, inplace=True)
train_actual['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
test_actual['walkDistance_over_kills'] = test_actual['walkDistance'] / test_actual['kills']
test_actual['walkDistance_over_kills'].fillna(0, inplace=True)
test_actual['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
 
train_actual['killsPerWalkDistance'] = train_actual['kills'] / train_actual['walkDistance']
train_actual['killsPerWalkDistance'].fillna(0, inplace=True)
train_actual['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)
test_actual['killsPerWalkDistance'] = test_actual['kills'] / test_actual['walkDistance']
test_actual['killsPerWalkDistance'].fillna(0, inplace=True)
test_actual['killsPerWalkDistance'].replace(np.inf, 0, inplace=True)

train_actual["skill"] = train_actual["headshotKills"] + train_actual["roadKills"]
test_actual["skill"] = test_actual["headshotKills"] + test_actual["roadKills"]

#delete not required df
del train_actual['heals'], test_actual['heals']
gc.collect()

#feature engineering
train_actual_grouped = feature_engineering(train_actual)
print("Number of train data grouped: ", len(train_actual_grouped))

test_actual_grouped = feature_engineering(test_actual)
print("Number of test data grouped: ", len(test_actual_grouped))

# train, test split
grouped_train, grouped_test = train_test_split(train_actual_grouped, test_size=0.2, random_state=0)
print("Number of pseudo train data: ", len(grouped_train))
print("Number of pseudo test data: ", len(grouped_test))

# split features and target
train_y = grouped_train['winPlacePerc_mean']
train_x = grouped_train.drop(['winPlacePerc_min', 'winPlacePerc_min_rank','winPlacePerc_max','winPlacePerc_max_rank','winPlacePerc_mean', 'winPlacePerc_mean_rank'], axis=1)
test_y = grouped_test[['winPlacePerc_mean']]
test_x = grouped_test.drop(['winPlacePerc_min', 'winPlacePerc_min_rank','winPlacePerc_max','winPlacePerc_max_rank','winPlacePerc_mean', 'winPlacePerc_mean_rank'], axis=1)
print("Number of train features: ", len(train_x.columns))
print("Number of test features: ", len(test_x.columns))

#delete not required df
del train_actual_grouped, grouped_train, grouped_test
gc.collect()

#reduce memory
train_x = reduce_mem_usage(train_x)
test_x = reduce_mem_usage(test_x)

#Light Gradient Boost Method
train_data = lgb.Dataset(data=train_x, label=train_y)
valid_data = lgb.Dataset(data=test_x, label=test_y)   
params = {"objective" : "regression", "metric" : "mae", 'n_estimators':15000, 'early_stopping_rounds':100,
          "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.9,
           "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
         }
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100) 

#predict for actual test grouped data
test_actual_grouped_predict = lgb_model.predict(test_actual_grouped, num_iteration=lgb_model.best_iteration)

#reduce memory
test_actual_grouped = reduce_mem_usage(test_actual_grouped)
test_actual = reduce_mem_usage(test_actual)

#assign predictions
test_actual_grouped["predict"] = test_actual_grouped_predict

#join predictions
test_actual = test_actual.merge(test_actual_grouped["predict"].reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])

#re-assign predictions
test_actual['winPlacePerc'] = test_actual['predict']

#first submission
submission1 = test_actual[['Id', 'winPlacePerc']]
submission1.loc[submission1['winPlacePerc'] > 1.0, "winPlacePerc"] = 1.0
submission1.loc[submission1['winPlacePerc'] < 0.0, "winPlacePerc"] = 0.0
submission1.to_csv('submission1.csv', index=False)

#copy first submission
test_actual['winPlacePerc1'] = test_actual['winPlacePerc']

#adjust predictions based on maxPlace
test_actual['gap'] = 1.0 / (test_actual['maxPlace'] - 1)
test_actual['winPlacePerc'] = round(test_actual['winPlacePerc']/ test_actual['gap']) * test_actual['gap']
test_actual.loc[test_actual['winPlacePerc'] > 1.0, "winPlacePerc"] = 1.0
test_actual.loc[test_actual['winPlacePerc'] < 0.0, "winPlacePerc"] = 0.0

#second submission
submission2 = test_actual[['Id', 'winPlacePerc']]
submission2.to_csv('submission2.csv', index=False)

#third submission
df_sub = pd.read_csv("../input/sample_submission_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_sub['winPlacePerc'] = test_actual['winPlacePerc1']

# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub[["Id", "winPlacePerc"]].to_csv("submission3.csv", index=False)
