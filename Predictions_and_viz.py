import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import calibration

#Loading and Prepping Data for predictions
os.chdir('C:/Users/Andrew Moss/PycharmProjects/Tennis_Live_WP')
model = pickle.load(open('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\XGB_live_odds.sav', 'rb'))
model_features = model.get_booster().feature_names

model_inputs_outputs = pd.read_csv('model_inputs_outputs.csv')
original_data = pd.read_csv('tennis_data.csv').reset_index()

model_inputs_outputs = pd.get_dummies(data=model_inputs_outputs, columns=['Points_Away_player_0', 'Points_Away_player_1',
                                                                          'Games_Away_player_0', 'Games_Away_player_1',
                                                                          'Sets_Away_player_0', 'Sets_Away_player_1'])
# Making actual percentage predictions and appending to original dataframe
model_inputs_outputs[['player_1_WP', 'player_2_WP']] = model.predict_proba(model_inputs_outputs.drop(
    ['Unnamed: 0', 'match_id', 'match_winning_player'], axis=1))


original_data = pd.merge(original_data, model_inputs_outputs[['Unnamed: 0', 'player_1_WP', 'player_2_WP']],
                         left_on='index', right_on='Unnamed: 0', how='left').drop('Unnamed: 0', axis=1)
## code that generates feature importance and calibration
features = pd.DataFrame({'features': model_features, 'importances': model.feature_importances_})
features = features.sort_values('importances', ascending=False).iloc[0:10, :]
plt.bar(features['features'].values, features['importances'].values)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.title('Top 10 Feature Importances')
plt.savefig('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\feature_importances.png', bbox_inches='tight')
plt.clf()

calibration_bins = calibration.calibration_curve(original_data['match_winning_player'], original_data['player_2_WP'].values, normalize=False,
                                     n_bins=20, strategy='uniform')
plt.plot(np.arange(0, 1, .05), calibration_bins[0])
plt.xticks(np.arange(0, 1, step=0.05), rotation=45, ha='right')
plt.yticks(np.arange(0, 1, step=0.05))
plt.title('Calibration Plot for Tennis WP Model')
plt.xlabel('Level of Model confidence for each situation')
plt.ylabel('Fraction of Player_2 Victories in Each Bin')
plt.tight_layout()
plt.savefig('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\Calibration.png', bbox_inches='tight')

original_data.to_csv('tennis_data_with_preds.csv')


