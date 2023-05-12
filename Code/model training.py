import pandas as pd
df = read_csv('train.csv')

# Target 
target = ['target']
features = ['home_team_pre_match_ppg__home','away_team_pre_match_ppg__home','home_ppg__home','away_ppg__home','home_team_goal_count__home','away_team_goal_count__home','total_goal_count__home','total_goals_at_half_time__home','home_team_goal_count_half_time__home','away_team_goal_count_half_time__home','home_team_goal_timings__home','away_team_goal_timings__home','home_team_corner_count__home','away_team_corner_count__home','home_team_yellow_cards__home','home_team_red_cards__home','away_team_yellow_cards__home','away_team_red_cards__home','home_team_first_half_cards__home','home_team_second_half_cards__home','away_team_first_half_cards__home','away_team_second_half_cards__home','home_team_shots__home','away_team_shots__home','home_team_shots_on_target__home','away_team_shots_on_target__home','home_team_shots_off_target__home','away_team_shots_off_target__home','home_team_fouls__home','away_team_fouls__home','home_team_possession__home','away_team_possession__home','home_team_xg__home','away_team_xg__home','home_team_pre_match_ppg__away','away_team_pre_match_ppg__away','home_ppg__away','away_ppg__away','home_team_goal_count__away','away_team_goal_count__away','total_goal_count__away','total_goals_at_half_time__away','home_team_goal_count_half_time__away','away_team_goal_count_half_time__away','home_team_goal_timings__away','away_team_goal_timings__away','home_team_corner_count__away','away_team_corner_count__away','home_team_yellow_cards__away','home_team_red_cards__away','away_team_yellow_cards__away','away_team_red_cards__away','home_team_first_half_cards__away','home_team_second_half_cards__away','away_team_first_half_cards__away','away_team_second_half_cards__away','home_team_shots__away','away_team_shots__away','home_team_shots_on_target__away','away_team_shots_on_target__away','home_team_shots_off_target__away','away_team_shots_off_target__away','home_team_fouls__away','away_team_fouls__away','home_team_possession__away','away_team_possession__away','home_team_xg__away','away_team_xg__away']


# Train Test Split
train = df.loc[:2000,target+features].values
test = df.loc[2000:,target+features].values


# Train model
import lightgbm as lgb
import numpy as np

params = {
    'device': 'cpu',
    'max_bin': 255,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'min_data_in_leaf': 5,
    #'num_leaves': 15,
    'max_depth':17,
    'n_estimators': 5000,
    'lambda_l1': 0.01,
    'learning_rate': 0.01,
    'feature_fraction': 0.95,
    'bagging_freq': 3,
    'metric': 'softmax'}

gbm = lgb.LGBMRegressor(**params)
gbm = gbm.fit(X=train[:,1:],y=train[:,0],eval_set=[(test[:,1:],test[:,0])],eval_metric=params['metric'],early_stopping_rounds=10)
preds = gbm.predict(test[:,1:])

