# shot_prediction
This repository contains midterm challenge for the DS701 class.


# 🏀 Basketball Shot Prediction - DS701 Midterm Challenge

## 📌 Project Overview
This project focuses on predicting **shot success (SHOT_MADE)** in basketball using machine learning techniques **without deep learning**. The dataset includes various player, game, and shot-related features, and the goal is to engineer meaningful features to improve predictive accuracy.

## 📂 Dataset
The dataset for this project is provided as part of the **DS701 Fall 2024 Midterm Competition** on Kaggle. It includes shot-level data with the following key features:
- **Spatial Features** (`LOC_X`, `LOC_Y`, `SHOT_DISTANCE`)
- **Game Context** (`GAME_DATE`, `QUARTER`, `MINS_LEFT`, `SECS_LEFT`)
- **Player Information** (`PLAYER_NAME`, `TEAM_NAME`, `POSITION`)
- **Shot Type & Location** (`SHOT_TYPE`, `ZONE_NAME`, `BASIC_ZONE`)

🔗 **Dataset Source**: [Kaggle Competition](https://www.kaggle.com/competitions/ds-701-midterm-competition/data)  
📜 **Citation**:  
Chandrahas Aroori, Farid Karimli, Scott Ladenheim, Shreyas Sudarsan, and Thomas Gardos. *DS701 Fall 2024 Midterm Competition*. [Kaggle](https://kaggle.com/competitions/ds-701-midterm-competition), 2024.

---

## 🚀 Methodology

### 🔍 Feature Engineering
To enhance prediction accuracy, the following new features were created:

1. **Player Success Rate**  
   - `game_success_rate`: Average shot success rate for each player in a given game.  
   - `AVG_SUCCESS_RATE`: Overall average shot success rate per player across all games.  
   ```python
   game_success_rate = data.groupby(['PLAYER_NAME', 'GAME_ID'])['SHOT_MADE'].mean().reset_index()
   average_success_rate_per_player = game_success_rate.groupby('PLAYER_NAME')['SHOT_MADE'].mean().reset_index()
   average_success_rate_per_player.rename(columns={'SHOT_MADE': 'AVG_SUCCESS_RATE'}, inplace=True)
   data = data.merge(average_success_rate_per_player[['PLAYER_NAME', 'AVG_SUCCESS_RATE']], on='PLAYER_NAME', how='left')


2. **Spatial Features**

- `SHOT_ANGLE`: Angle of the shot using atan2(LOC_Y, LOC_X).
- `SHOT_DISTANCE`: Distance from the basket using the Euclidean distance formula.

  ```python
   data['SHOT_ANGLE'] = np.arctan2(data['LOC_Y'], data['LOC_X'])
   data['SHOT_DISTANCE'] = np.sqrt(data['LOC_X']**2 + data['LOC_Y']**2)

3. **Fatigue Estimation**

- `FATIGUE`: Estimated using time left in the game.
  
   ```python
   data['FATIGUE'] = (data['MINS_LEFT'] / 48) + (data['SECS_LEFT'] / 2880)

4. **Game Progression**

- `DAY_SINCE_SEASON_START`: Number of days since the start of the season.
  
   ```python
   data['SEASON_YEAR'] = data['SEASON_2'].str.split('-').str[0].astype(int)
   def get_season_start_date(season_year):
       return pd.to_datetime(f'{season_year}-10-01')
   data['SEASON_START_DATE'] = data['SEASON_YEAR'].apply(get_season_start_date)
   data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
   data['DAY_SINCE_SEASON_START'] = (data['GAME_DATE'] - data['SEASON_START_DATE']).dt.days
   data.drop(columns=['SEASON_START_DATE', 'SEASON_YEAR'], inplace=True)

5. **Home Court Advantage**

- `HOME_ADVANTAGE`: 1 if the team was playing at home, 0 otherwise.

   ```python
   data['TEAM_ABBREVIATION'] = data['TEAM_NAME'].map(team_name_to_abbreviation)
   data['HOME_ADVANTAGE'] = (data['TEAM_ABBREVIATION'] == data['HOME_TEAM']).astype(int)
   data.drop(columns=['TEAM_ABBREVIATION'], inplace=True)

6. **Position-Zone Advantage**

 - `POSITION_ZONE_ADVANTAGE`: Average shot success rate for each position-zone combination.
 - 
   ```python
   position_zone_success = data.groupby(['POSITION', 'ZONE_NAME'])['SHOT_MADE'].mean().reset_index()
   position_zone_success.rename(columns={'SHOT_MADE': 'POSITION_ZONE_ADVANTAGE'}, inplace=True)
   data = data.merge(position_zone_success, on=['POSITION', 'ZONE_NAME'], how='left')

### 🎯 Model & Evaluation

The model selection focused on machine learning algorithms (excluding deep learning).
 - Model Used: XGBoost decision trees.
- Preprocessing: One-hot encoding for categorical variables, scaling numerical features.
- Hyperparameter Tuning: Grid search and cross-validation for optimal performance.
- Evaluation Metrics: Accuracy, F1-score, AUC-ROC.


📉 Detailed model selection process and performance results will be included in the report.
