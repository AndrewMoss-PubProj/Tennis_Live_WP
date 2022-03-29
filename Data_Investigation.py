import pandas as pd
import numpy as np
import Utils

##Reading in data and applying basic transformations
data = pd.read_csv('tennis_data.csv')
print(data.head(5))
# renaming everything player_0 or player_1 to player_0 and player_0. thinking in 0 index in python means less to keep track of
data['serving_player'] = np.where(data['serving_player'] == 1, 0, 1)
data['match_winning_player'] = np.where(data['match_winning_player'] == 1, 0, 1)

data = data.rename(columns={'player_1_points': 'player_0_points', 'player_2_points': 'player_1_points',
                            'player_1_games': 'player_0_games', 'player_2_games': 'player_1_games',
                            'player_1_sets': 'player_0_sets', 'player_2_sets': 'player_1_sets'})


## AD-40 is the same effective score as 40-30, so applying np.where to standardize
## Likewise, deuce is the same number of points away from winning the game as 30-30, so same transformation applied

player_0_ads = np.where((data['player_0_points'] == 'AD') & (data['player_1_points'] == '40'))[0]
player_1_ads = np.where((data['player_1_points'] == 'AD') & (data['player_0_points'] == '40'))[0]
deuces = np.where((data['player_1_points'] == '40') & (data['player_0_points'] == '40'))[0]

## Potential Bias in Dataset: This dataset is structured such that 6-5 always becomes 7-5. That means that P(6-5 -> 7-5)
## in the dataset > P(6-5 -> 7-5) in the population. As a result I want the model to look at 6-5 and 5-4 the same
## because they are the same number of games away from winning the set. Future work would include P(Set goes to tiebreak)
## model and a tiebreak winner model but they are outside of the scope of this 8 hour project

six_five_p1 = np.where((data['player_0_games'] == 6) & (data['player_1_games'] == 5))[0]
six_five_p2 = np.where((data['player_0_games'] == 5) & (data['player_1_games'] == 6))[0]


data.loc[player_0_ads, ['player_0_points', 'player_1_points']] = [[40, 30]] * len(player_0_ads)
data.loc[player_1_ads, ['player_0_points', 'player_1_points']] = [[30, 40]] * len(player_1_ads)
data.loc[deuces, ['player_0_points', 'player_1_points']] = [[30, 30]] * len(deuces)
data.loc[six_five_p1, ['player_0_games', 'player_1_games']] = [[5, 4]] * len(six_five_p1)
data.loc[six_five_p2, ['player_0_games', 'player_1_games']] = [[4, 5]] * len(six_five_p2)



data['player_0_points'] = data['player_0_points'].astype(int)
data['player_1_points'] = data['player_1_points'].astype(int)



## Creating Points_Away_from_game and Games_Away_from_set, and sets_away from match variables for player 1 and 2
data['Points_Away_player_0'] = np.where(data['player_0_points'] == 0, 4,
                                        np.where(data['player_0_points'] == 15, 3,
                                        np.where(data['player_0_points'] == 30, 2, 1)))
data['Points_Away_player_1'] = np.where(data['player_1_points'] == 0, 4,
                                        np.where(data['player_1_points'] == 15, 3,
                                        np.where(data['player_1_points'] == 30, 2, 1)))
data['Games_Away_player_0'] = np.where(data['player_1_games'] == 5,
                                       7-data['player_0_games'],
                                       6-data['player_0_games'])
data['Games_Away_player_1'] = np.where(data['player_0_games'] == 5,
                                       7-data['player_1_games'],
                                       6-data['player_1_games'])
data['Sets_Away_player_0'] = 2 - data['player_0_sets']
data['Sets_Away_player_1'] = 2 - data['player_1_sets']
## Considered creating a function to determine running total of % of all points by a single player but decided
## sample size would be too small/it would create too much noise for a tree-based model that I want to implement later

##Some basic EDA, like figuring out % of points won by serving player and winning percentage for each game/set score combo

game_set_score_groups = data.groupby(['Sets_Away_player_0', 'Sets_Away_player_1',
                                      'Games_Away_player_0', 'Games_Away_player_1']).mean().sort_values('match_winning_player')
## This is a little croweded/difficult to read table, so simplifying to get something a little easier to read
data['p0_set_diff'] = data['Sets_Away_player_1'] - data['Sets_Away_player_0']
data['p0_game_diff'] = data['Games_Away_player_1'] - data['Games_Away_player_0']
game_set_diff_groups = data.groupby(['p0_game_diff', 'p0_set_diff']).mean().sort_values('match_winning_player')['match_winning_player']
## This is table of probabilities by game and set score
game_set_diff_groups = abs(game_set_diff_groups.reset_index().pivot(index='p0_game_diff',columns='p0_set_diff').subtract(2))



##adding the winner of the next point and current game as a variable to help with EDA
## I'm adding a match_game ID so I'm renaming what you gave me as game ID to match ID and creating my own game_ID
data = data.rename(columns={'game_id': 'match_id'})
data['game_id'] = 0
data = Utils.game_IDs(data)

data['point_winner'] = -1
game_groups = Utils.game_winners(data.groupby('game_id').mean())


for match_id in data['match_id'].unique():
    print(match_id)
    temp_game = data[data['match_id'] == match_id]
    starting_index = temp_game.iloc[0].name
    ending_index = temp_game.iloc[-1].name
    temp_game = Utils.point_winner(temp_game)
    data.loc[starting_index:ending_index, :] = temp_game
data['point_won_by_server'] = np.where(data['point_winner'] == data['serving_player'], 1, 0)
game_groups['game_won_by_server'] = np.where(game_groups['game_winner'] == game_groups['serving_player'], 1, 0)
model_inputs_outputs = data.loc[:, ['match_id', 'serving_player', 'Points_Away_player_0', 'Points_Away_player_1',
                                   'Games_Away_player_0', 'Games_Away_player_1', 'Sets_Away_player_0','Sets_Away_player_1',
                                    'match_winning_player']]
data.to_csv('investigated_data.csv')
model_inputs_outputs.to_csv('model_inputs_outputs.csv')



# sum(data['point_won_by_server'])/len(data) shows us that 56.7 percent of points were won by the serving player and
# sum(game_groups['game_won_by_server'])/len(game groups) shows us that 65.0 percent of games were won by the server





