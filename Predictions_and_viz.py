import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import calibration

#Loading and Prepping Data for predictions
os.chdir('C:/Users/Andrew Moss/PycharmProjects/Tennis_Live_WP')
model_xgb = pickle.load(open('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\XGB_live_odds.sav', 'rb'))
print()
model_xgb_features = model_xgb.get_booster().feature_names

model_log = pickle.load(open('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\Logistic_Regressions_live_odds.sav', 'rb'))
print()
model_log_features = model_log.coef_

model_inputs_outputs = pd.read_csv('model_inputs_outputs.csv')
original_data = pd.read_csv('tennis_data.csv').reset_index()

model_inputs_outputs = pd.get_dummies(data=model_inputs_outputs, columns=['Points_Away_player_0', 'Points_Away_player_1',
                                                                          'Games_Away_player_0', 'Games_Away_player_1',
                                                                          'Sets_Away_player_0', 'Sets_Away_player_1'])
# Making actual percentage predictions and appending to original dataframe
model_inputs_outputs[['player_1_WP_XGB', 'player_2_WP_XGB']] = model_xgb.predict_proba(model_inputs_outputs.drop(
    ['Unnamed: 0', 'match_id', 'match_winning_player'], axis=1))

model_inputs_outputs[['player_1_WP_Log', 'player_2_WP_Log']] = model_log.predict_proba(model_inputs_outputs.drop(
    ['Unnamed: 0', 'match_id', 'match_winning_player', 'player_1_WP_XGB', 'player_2_WP_XGB'], axis=1))


original_data = pd.merge(original_data, model_inputs_outputs[['Unnamed: 0', 'player_1_WP_XGB', 'player_2_WP_XGB']],
                         left_on='index', right_on='Unnamed: 0', how='left')

original_data = pd.merge(original_data, model_inputs_outputs[['Unnamed: 0', 'player_1_WP_Log', 'player_2_WP_Log']],
                         left_on='index', right_on='Unnamed: 0', how='left').drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
## code that generates feature importance and calibration
features = pd.DataFrame({'features': model_xgb_features, 'importances': model_xgb.feature_importances_})
features = features.sort_values('importances', ascending=False).iloc[0:10, :]
plt.bar(features['features'].values, features['importances'].values)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.title('Top 10 Feature Importances')
plt.savefig('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\feature_importances.png', bbox_inches='tight')
plt.clf()

calibration_bins_XGB = calibration.calibration_curve(original_data['match_winning_player'], original_data['player_2_WP_XGB'].values, normalize=False,
                                     n_bins=20, strategy='uniform')
calibration_bins_Log = calibration.calibration_curve(original_data['match_winning_player'], original_data['player_2_WP_Log'].values, normalize=False,
                                     n_bins=20, strategy='uniform')
plt.plot(np.arange(0, 1, .05), calibration_bins_XGB[0])
plt.plot(np.arange(0, 1, .05), calibration_bins_Log[0])
plt.plot(np.arange(0, 1, .05), np.arange(0, 1, .05))


print('Calibration SSE Log: ', sum(i*i for i in calibration_bins_Log[0]-np.arange(0, 1, .05)))
print('Calibration SSE XGB: ', sum(i*i for i in calibration_bins_XGB[0]-np.arange(0, 1, .05)))

plt.legend(labels=['XGBoost', 'Logistic Regression','y=x (Perfect calibration)'])
plt.xticks(np.arange(0, 1, step=0.05), rotation=45, ha='right')
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Calibration Plot for Tennis WP Model')
plt.xlabel('Level of Model confidence for each situation')
plt.ylabel('Fraction of Player_2 Victories in Each Bin')
plt.tight_layout()
plt.savefig('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\Calibration.png', bbox_inches='tight')

original_data.to_csv('tennis_data_with_preds.csv')

#Calibration SSE Log:  0.02519
#Calibration SSE XGB:  0.01346


