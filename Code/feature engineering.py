# Feature engineering in SQL
sql = '''
with hometeamlast10history as

(select s1.*,
s2.timestamp as timestamp2,
s2.home_team_pre_match_ppg as home_team_pre_match_ppg__home,
s2.away_team_pre_match_ppg as away_team_pre_match_ppg__home,
s2.home_ppg as home_ppg__home,
s2.away_ppg as away_ppg__home,
s2.home_team_goal_count as home_team_goal_count__home,
s2.away_team_goal_count as away_team_goal_count__home,
s2.total_goal_count as total_goal_count__home,
s2.total_goals_at_half_time as total_goals_at_half_time__home,
s2.home_team_goal_count_half_time as home_team_goal_count_half_time__home,
s2.away_team_goal_count_half_time as away_team_goal_count_half_time__home,
s2.home_team_goal_timings as home_team_goal_timings__home,
s2.away_team_goal_timings as away_team_goal_timings__home,
s2.home_team_corner_count as home_team_corner_count__home,
s2.away_team_corner_count as away_team_corner_count__home,
s2.home_team_yellow_cards as home_team_yellow_cards__home,
s2.home_team_red_cards as home_team_red_cards__home,
s2.away_team_yellow_cards as away_team_yellow_cards__home,
s2.away_team_red_cards as away_team_red_cards__home,
s2.home_team_first_half_cards as home_team_first_half_cards__home,
s2.home_team_second_half_cards as home_team_second_half_cards__home,
s2.away_team_first_half_cards as away_team_first_half_cards__home,
s2.away_team_second_half_cards as away_team_second_half_cards__home,
s2.home_team_shots as home_team_shots__home,
s2.away_team_shots as away_team_shots__home,
s2.home_team_shots_on_target as home_team_shots_on_target__home,
s2.away_team_shots_on_target as away_team_shots_on_target__home,
s2.home_team_shots_off_target as home_team_shots_off_target__home,
s2.away_team_shots_off_target as away_team_shots_off_target__home,
s2.home_team_fouls as home_team_fouls__home,
s2.away_team_fouls as away_team_fouls__home,
s2.home_team_possession as home_team_possession__home,
s2.away_team_possession as away_team_possession__home,
s2.home_team_xg as home_team_xg__home,
s2.away_team_xg as away_team_xg__home,
rank() over(partition by s1.timestamp,s1.home_team_name order by s2.timestamp desc) as rnk

from soccer s1
join soccer s2
  on s1.home_team_name = s2.home_team_name
  and s1.timestamp > s2.timestamp),

awayteamlast10history as

(select s1.*,
s2.timestamp as timestamp2,
s2.home_team_pre_match_ppg as home_team_pre_match_ppg__away,
s2.away_team_pre_match_ppg as away_team_pre_match_ppg__away,
s2.home_ppg as home_ppg__away,
s2.away_ppg as away_ppg__away,
s2.home_team_goal_count as home_team_goal_count__away,
s2.away_team_goal_count as away_team_goal_count__away,
s2.total_goal_count as total_goal_count__away,
s2.total_goals_at_half_time as total_goals_at_half_time__away,
s2.home_team_goal_count_half_time as home_team_goal_count_half_time__away,
s2.away_team_goal_count_half_time as away_team_goal_count_half_time__away,
s2.home_team_goal_timings as home_team_goal_timings__away,
s2.away_team_goal_timings as away_team_goal_timings__away,
s2.home_team_corner_count as home_team_corner_count__away,
s2.away_team_corner_count as away_team_corner_count__away,
s2.home_team_yellow_cards as home_team_yellow_cards__away,
s2.home_team_red_cards as home_team_red_cards__away,
s2.away_team_yellow_cards as away_team_yellow_cards__away,
s2.away_team_red_cards as away_team_red_cards__away,
s2.home_team_first_half_cards as home_team_first_half_cards__away,
s2.home_team_second_half_cards as home_team_second_half_cards__away,
s2.away_team_first_half_cards as away_team_first_half_cards__away,
s2.away_team_second_half_cards as away_team_second_half_cards__away,
s2.home_team_shots as home_team_shots__away,
s2.away_team_shots as away_team_shots__away,
s2.home_team_shots_on_target as home_team_shots_on_target__away,
s2.away_team_shots_on_target as away_team_shots_on_target__away,
s2.home_team_shots_off_target as home_team_shots_off_target__away,
s2.away_team_shots_off_target as away_team_shots_off_target__away,
s2.home_team_fouls as home_team_fouls__away,
s2.away_team_fouls as away_team_fouls__away,
s2.home_team_possession as home_team_possession__away,
s2.away_team_possession as away_team_possession__away,
s2.home_team_xg as home_team_xg__away,
s2.away_team_xg as away_team_xg__away,
rank() over(partition by s1.timestamp,s1.away_team_name order by s2.timestamp desc) as rnk

from soccer s1
join soccer s2
  on s1.away_team_name = s2.away_team_name
  and s1.timestamp > s2.timestamp)

select h1.home_team_name,h1.away_team_name,
case 
    when h1.home_team_goal_count > h1.away_team_goal_count then 0
    when h1.home_team_goal_count = h1.away_team_goal_count then 1
    else 2
  end as target,
h1.timestamp,

avg(home_team_pre_match_ppg__home) as home_team_pre_match_ppg__home,
avg(away_team_pre_match_ppg__home) as away_team_pre_match_ppg__home,
avg(home_ppg__home) as home_ppg__home,
avg(away_ppg__home) as away_ppg__home,
avg(home_team_goal_count__home) as home_team_goal_count__home,
avg(away_team_goal_count__home) as away_team_goal_count__home,
avg(total_goal_count__home) as total_goal_count__home,
avg(total_goals_at_half_time__home) as total_goals_at_half_time__home,
avg(home_team_goal_count_half_time__home) as home_team_goal_count_half_time__home,
avg(away_team_goal_count_half_time__home) as away_team_goal_count_half_time__home,
avg(home_team_goal_timings__home) as home_team_goal_timings__home,
avg(away_team_goal_timings__home) as away_team_goal_timings__home,
avg(home_team_corner_count__home) as home_team_corner_count__home,
avg(away_team_corner_count__home) as away_team_corner_count__home,
avg(home_team_yellow_cards__home) as home_team_yellow_cards__home,
avg(home_team_red_cards__home) as home_team_red_cards__home,
avg(away_team_yellow_cards__home) as away_team_yellow_cards__home,
avg(away_team_red_cards__home) as away_team_red_cards__home,
avg(home_team_first_half_cards__home) as home_team_first_half_cards__home,
avg(home_team_second_half_cards__home) as home_team_second_half_cards__home,
avg(away_team_first_half_cards__home) as away_team_first_half_cards__home,
avg(away_team_second_half_cards__home) as away_team_second_half_cards__home,
avg(home_team_shots__home) as home_team_shots__home,
avg(away_team_shots__home) as away_team_shots__home,
avg(home_team_shots_on_target__home) as home_team_shots_on_target__home,
avg(away_team_shots_on_target__home) as away_team_shots_on_target__home,
avg(home_team_shots_off_target__home) as home_team_shots_off_target__home,
avg(away_team_shots_off_target__home) as away_team_shots_off_target__home,
avg(home_team_fouls__home) as home_team_fouls__home,
avg(away_team_fouls__home) as away_team_fouls__home,
avg(home_team_possession__home) as home_team_possession__home,
avg(away_team_possession__home) as away_team_possession__home,
avg(home_team_xg__home) as home_team_xg__home,
avg(away_team_xg__home) as away_team_xg__home,
avg(home_team_pre_match_ppg__away) as home_team_pre_match_ppg__away,
avg(away_team_pre_match_ppg__away) as away_team_pre_match_ppg__away,
avg(home_ppg__away) as home_ppg__away,
avg(away_ppg__away) as away_ppg__away,
avg(home_team_goal_count__away) as home_team_goal_count__away,
avg(away_team_goal_count__away) as away_team_goal_count__away,
avg(total_goal_count__away) as total_goal_count__away,
avg(total_goals_at_half_time__away) as total_goals_at_half_time__away,
avg(home_team_goal_count_half_time__away) as home_team_goal_count_half_time__away,
avg(away_team_goal_count_half_time__away) as away_team_goal_count_half_time__away,
avg(home_team_goal_timings__away) as home_team_goal_timings__away,
avg(away_team_goal_timings__away) as away_team_goal_timings__away,
avg(home_team_corner_count__away) as home_team_corner_count__away,
avg(away_team_corner_count__away) as away_team_corner_count__away,
avg(home_team_yellow_cards__away) as home_team_yellow_cards__away,
avg(home_team_red_cards__away) as home_team_red_cards__away,
avg(away_team_yellow_cards__away) as away_team_yellow_cards__away,
avg(away_team_red_cards__away) as away_team_red_cards__away,
avg(home_team_first_half_cards__away) as home_team_first_half_cards__away,
avg(home_team_second_half_cards__away) as home_team_second_half_cards__away,
avg(away_team_first_half_cards__away) as away_team_first_half_cards__away,
avg(away_team_second_half_cards__away) as away_team_second_half_cards__away,
avg(home_team_shots__away) as home_team_shots__away,
avg(away_team_shots__away) as away_team_shots__away,
avg(home_team_shots_on_target__away) as home_team_shots_on_target__away,
avg(away_team_shots_on_target__away) as away_team_shots_on_target__away,
avg(home_team_shots_off_target__away) as home_team_shots_off_target__away,
avg(away_team_shots_off_target__away) as away_team_shots_off_target__away,
avg(home_team_fouls__away) as home_team_fouls__away,
avg(away_team_fouls__away) as away_team_fouls__away,
avg(home_team_possession__away) as home_team_possession__away,
avg(away_team_possession__away) as away_team_possession__away,
avg(home_team_xg__away) as home_team_xg__away,
avg(away_team_xg__away) as away_team_xg__away

from hometeamlast10history h1
join awayteamlast10history h2
  on h1.home_team_name = h2.home_team_name
  and h1.away_team_name = h2.away_team_name
  and h1.timestamp = h2.timestamp

where h1.rnk<10 and h2.rnk<10
group by 
  h1.home_team_name,h1.away_team_name,
  case 
      when h1.home_team_goal_count > h1.away_team_goal_count then 0
      when h1.home_team_goal_count = h1.away_team_goal_count then 1
      else 2
    end,
  h1.timestamp
'''

cols = ['home_team_name','away_team_name','target','timestamp','home_team_pre_match_ppg__home','away_team_pre_match_ppg__home','home_ppg__home','away_ppg__home','home_team_goal_count__home','away_team_goal_count__home','total_goal_count__home','total_goals_at_half_time__home','home_team_goal_count_half_time__home','away_team_goal_count_half_time__home','home_team_goal_timings__home','away_team_goal_timings__home','home_team_corner_count__home','away_team_corner_count__home','home_team_yellow_cards__home','home_team_red_cards__home','away_team_yellow_cards__home','away_team_red_cards__home','home_team_first_half_cards__home','home_team_second_half_cards__home','away_team_first_half_cards__home','away_team_second_half_cards__home','home_team_shots__home','away_team_shots__home','home_team_shots_on_target__home','away_team_shots_on_target__home','home_team_shots_off_target__home','away_team_shots_off_target__home','home_team_fouls__home','away_team_fouls__home','home_team_possession__home','away_team_possession__home','home_team_xg__home','away_team_xg__home','home_team_pre_match_ppg__away','away_team_pre_match_ppg__away','home_ppg__away','away_ppg__away','home_team_goal_count__away','away_team_goal_count__away','total_goal_count__away','total_goals_at_half_time__away','home_team_goal_count_half_time__away','away_team_goal_count_half_time__away','home_team_goal_timings__away','away_team_goal_timings__away','home_team_corner_count__away','away_team_corner_count__away','home_team_yellow_cards__away','home_team_red_cards__away','away_team_yellow_cards__away','away_team_red_cards__away','home_team_first_half_cards__away','home_team_second_half_cards__away','away_team_first_half_cards__away','away_team_second_half_cards__away','home_team_shots__away','away_team_shots__away','home_team_shots_on_target__away','away_team_shots_on_target__away','home_team_shots_off_target__away','away_team_shots_off_target__away','home_team_fouls__away','away_team_fouls__away','home_team_possession__away','away_team_possession__away','home_team_xg__away','away_team_xg__away']
data = c.execute(sql).fetchall()
df = pd.DataFrame(data,columns = cols)
df.to_csv('train.csv',index=False)


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




# Get Latest Stats for Prediction Model Inputs
copa = pd.read_csv('./Data/dummy_submission_file_copa_america.csv')

data = []
for row in range(len(copa)):
  hometeam = copa.iloc[row].team1_name
  awayteam = copa.iloc[row].team2_name
  print(hometeam,awayteam)
  team_df = df.loc[(df.home_team_name==hometeam),:]
  vec1 = team_df.loc[team_df.timestamp == team_df.timestamp.max(),features[:34]].values
  team_df = df.loc[(df.away_team_name==awayteam),:]
  vec2 = team_df.loc[team_df.timestamp == team_df.timestamp.max(),features[34:]].values
  vec = np.concatenate([vec1[0],vec2[0]])
  data.append(vec)

f1 = pd.DataFrame(gbm.predict(np.array(data)))

copa.p_team1_win = f1[:,0]
copa.p_draw = f1[:,1]
copa.p_team2_win = f1[:,2]








# Get Latest Stats for Prediction Model Inputs
euro = pd.read_csv('./Data/dummy_submission_file_euro.csv')

data = []
for row in range(len(euro)):
  hometeam = euro.iloc[row].team1_name
  awayteam = euro.iloc[row].team2_name
  print(hometeam,awayteam)
  team_df = df.loc[(df.home_team_name==hometeam),:]
  vec1 = team_df.loc[team_df.timestamp == team_df.timestamp.max(),features[:34]].values
  team_df = df.loc[(df.away_team_name==awayteam),:]
  vec2 = team_df.loc[team_df.timestamp == team_df.timestamp.max(),features[34:]].values
  vec = np.concatenate([vec1[0],vec2[0]])
  data.append(vec)

f1 = pd.DataFrame(gbm.predict(np.array(data)))

euro.p_team1_win = f1.loc[:,0]
euro.p_draw = f1.loc[:,1]
euro.p_team2_win = f1.loc[:,2]

euro.to_csv('preds_euro.csv')
