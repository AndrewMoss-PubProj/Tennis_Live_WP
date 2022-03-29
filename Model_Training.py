import pandas as pd
import os
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pickle
import timer

os.chdir('C:/Users/Andrew Moss/PycharmProjects/Tennis_Live_WP')
data = pd.read_csv('model_inputs_outputs.csv').drop('Unnamed: 0', axis=1)

# Preprocessing with one-hot encoding for Games_Away and Points for both players.


data = pd.get_dummies(data=data, columns=['Points_Away_player_0', 'Points_Away_player_1',
                                          'Games_Away_player_0', 'Games_Away_player_1',
                                          'Sets_Away_player_0', 'Sets_Away_player_1'])
inputs = data.drop(['match_id', 'match_winning_player'], axis=1)
outputs = data['match_winning_player']

## I should have mirrored the data here to remove bias between priors over independant p1 and p2, but my computer is too
## slow to retrain again, but something I thought about when looking through the results. A lot of the considerations
## I've made for this particular model seems like they could have been solved with better priors but it seems that was
## a part of this particular assignment

model_kfold_log = LogisticRegression()

model_kfold_log.fit(inputs, outputs)
pickle.dump(model_kfold_log, open('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\'
                              'Logistic_Regressions_live_odds.sav', 'wb'))


kfold = StratifiedKFold(n_splits=10)
results_kfold_log = model_selection.cross_val_score(model_kfold_log, inputs, outputs,
                                                scoring='neg_log_loss', cv=kfold, n_jobs=-1)
print("Log_Loss: '%.3g'" % (results_kfold_log.mean()))



params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 500, 1000]
        }
model_kfold_XGB = XGBClassifier(objective='binary:logistic')
gridsearch = GridSearchCV(model_kfold_XGB, params, scoring='neg_log_loss', n_jobs=-1,
                          cv=kfold.split(inputs, outputs), verbose=3)
start_time = timer
gridsearch.fit(inputs, outputs)
print(start_time)

results_kfold_XGB = model_selection.cross_val_score(model_kfold_XGB, inputs, outputs,
                                                scoring='neg_log_loss', cv=kfold, n_jobs=-1)
print("Log_Loss: '%.3g'" % (results_kfold_XGB.mean()))

model_kfold_XGB.fit(inputs, outputs)
pickle.dump(model_kfold_XGB, open('C:\\Users\\Andrew Moss\\PycharmProjects\\Tennis_Live_WP\\XGB_live_odds.sav', 'wb'))






