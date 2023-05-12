import pandas as pd
df = read_csv('train.csv')

# Target 
target = ['target']
features = ['home_team_pre_match_ppg__home','away_team_pre_match_ppg__home','home_ppg__home','away_ppg__home','home_team_goal_count__home','away_team_goal_count__home','total_goal_count__home','total_goals_at_half_time__home','home_team_goal_count_half_time__home','away_team_goal_count_half_time__home','home_team_goal_timings__home','away_team_goal_timings__home','home_team_corner_count__home','away_team_corner_count__home','home_team_yellow_cards__home','home_team_red_cards__home','away_team_yellow_cards__home','away_team_red_cards__home','home_team_first_half_cards__home','home_team_second_half_cards__home','away_team_first_half_cards__home','away_team_second_half_cards__home','home_team_shots__home','away_team_shots__home','home_team_shots_on_target__home','away_team_shots_on_target__home','home_team_shots_off_target__home','away_team_shots_off_target__home','home_team_fouls__home','away_team_fouls__home','home_team_possession__home','away_team_possession__home','home_team_xg__home','away_team_xg__home','home_team_pre_match_ppg__away','away_team_pre_match_ppg__away','home_ppg__away','away_ppg__away','home_team_goal_count__away','away_team_goal_count__away','total_goal_count__away','total_goals_at_half_time__away','home_team_goal_count_half_time__away','away_team_goal_count_half_time__away','home_team_goal_timings__away','away_team_goal_timings__away','home_team_corner_count__away','away_team_corner_count__away','home_team_yellow_cards__away','home_team_red_cards__away','away_team_yellow_cards__away','away_team_red_cards__away','home_team_first_half_cards__away','home_team_second_half_cards__away','away_team_first_half_cards__away','away_team_second_half_cards__away','home_team_shots__away','away_team_shots__away','home_team_shots_on_target__away','away_team_shots_on_target__away','home_team_shots_off_target__away','away_team_shots_off_target__away','home_team_fouls__away','away_team_fouls__away','home_team_possession__away','away_team_possession__away','home_team_xg__away','away_team_xg__away']


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
