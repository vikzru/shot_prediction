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
