import pandas as pd
import numpy as np

## For a given dataset that spans a game, this method adds a column with who won the next point
def point_winner(game):
    for index, row in game.iterrows():
        if index == game.iloc[-1].name:
            game.loc[index, 'point_winner'] = game.loc[index, 'match_winning_player'].copy()
        else:
            if game.loc[index + 1, 'player_0_sets'] > row['player_0_sets'] or \
                    game.loc[index + 1, 'player_0_games'] > row['player_0_games'] or \
                    game.loc[index + 1, 'player_0_points'] > row['player_0_points']:
                game.loc[index, 'point_winner'] = 0
            elif game.loc[index + 1, 'player_1_points'] < row['player_1_points'] and game.loc[index + 1, 'player_1_games'] == row['player_1_games']:
                game.loc[index, 'point_winner'] = 0
            else:
                game.loc[index, 'point_winner'] = 1
    return game

## Gives each game its own ID
def game_IDs(data):
    for index, row in data.iterrows():
        print(index)
        if index == 0:
            continue
        elif row['serving_player'] == data.loc[index - 1, 'serving_player'] and \
                row['set_num'] == data.loc[index - 1, 'set_num']:
            data.loc[index, 'game_id'] = data.loc[index-1, 'game_id'].copy()
        else:
            data.loc[index, 'game_id'] = data.loc[index - 1, 'game_id'].copy() + 1
    return data

## For a given dataset that spans a match, this method adds a column with who won the game the row is apart of
def game_winners(game_groups):
    game_groups['game_winner'] = -1
    for index, row in game_groups.iterrows():
        print(index)
        if index == len(game_groups) - 1:
            game_groups.loc[index, 'game_winner'] = game_groups.loc[index, 'match_winning_player'].copy()
        elif row['match_id'] == game_groups.loc[index + 1, 'match_id']:
            if row['set_num'] == game_groups.loc[index + 1, 'set_num'] and \
                    row['player_0_games'] < game_groups.loc[index + 1, 'player_0_games']:
                game_groups.loc[index, 'game_winner'] = 0
            elif row['player_0_sets'] > game_groups.loc[index + 1, 'player_0_sets']:
                game_groups.loc[index, 'game_winner'] = 0
            else:
                game_groups.loc[index, 'game_winner'] = 1
        else:
            game_groups.loc[index, 'game_winner'] = game_groups.loc[index, 'match_winning_player'].copy()
    return game_groups