# Create DB
from pathlib import Path
Path('./Data/form.db').touch()

# Create Table
import sqlite3
conn = sqlite3.connect('./Data/form.db')
c = conn.cursor()
c.execute('''CREATE TABLE soccer (
              tournament TEXT,
              timestamp TEXT,
              date_GMT TEXT,
              status TEXT,
              stadium_name TEXT,
              attendance NUMERIC,
              home_team_name TEXT,
              away_team_name TEXT,
              referee NUMERIC,
              GameWeek NUMERIC,
              home_team_pre_match_ppg NUMERIC,
              away_team_pre_match_ppg NUMERIC,
              home_ppg NUMERIC,
              away_ppg NUMERIC,
              home_team_goal_count INT,
              away_team_goal_count INT,
              total_goal_count INT,
              total_goals_at_half_time NUMERIC,
              home_team_goal_count_half_time NUMERIC,
              away_team_goal_count_half_time NUMERIC,
              home_team_goal_timings TEXT,
              away_team_goal_timings TEXT,
              home_team_corner_count NUMERIC,
              away_team_corner_count NUMERIC,
              home_team_yellow_cards NUMERIC,
              home_team_red_cards NUMERIC,
              away_team_yellow_cards NUMERIC,
              away_team_red_cards NUMERIC,
              home_team_first_half_cards NUMERIC,
              home_team_second_half_cards NUMERIC,
              away_team_first_half_cards NUMERIC,
              away_team_second_half_cards NUMERIC,
              home_team_shots NUMERIC,
              away_team_shots NUMERIC,
              home_team_shots_on_target NUMERIC,
              away_team_shots_on_target NUMERIC,
              home_team_shots_off_target NUMERIC,
              away_team_shots_off_target NUMERIC,
              home_team_fouls NUMERIC,
              away_team_fouls NUMERIC,
              home_team_possession NUMERIC,
              away_team_possession NUMERIC,
              home_team_xg NUMERIC,
              away_team_xg NUMERIC)''')


# Load Form into SQL
import pandas as pd
form = pd.read_csv('./Data/datathon_updated_form_data.csv')
form = form.fillna(0)
form.to_sql('soccer', conn, if_exists='append', index = False)