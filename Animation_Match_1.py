import plotly.graph_objects as go
import pandas as pd
import os

os.chdir('C:/Users/Andrew Moss/PycharmProjects/Tennis_Live_WP')
data = pd.read_csv('tennis_data_with_preds.csv').reset_index()
data = data[data['game_id'] == 1].loc[:, ['index', 'serving_player', 'player_1_points', 'player_2_points',
                                          'player_1_games', 'player_2_games', 'player_1_sets', 'player_2_sets',
                                          'player_1_WP_XGB']]



## Making the dots green when p1 win probability >= .5 and red when it is less
trace1 = go.Scatter(x=data.index[:2],
                    y=data.player_1_WP_XGB[:2],
                    mode='lines + markers',
                    marker=dict(
                    color=((.5 < data.player_1_WP_XGB).astype('int')), colorscale=[[0, 'red'], [1, 'green']]
                    ),
                    line=dict(color='black')
                )


frames = [dict(data=dict(type='scatter',
                           x=data.index[:k+1],
                           y=data.player_1_WP_XGB[:k+1]),
               traces=[0, 1],
              ) for k in range(1, len(data)-1)]

layout = go.Layout(width=800,
                   height=500,
                   title='Live Win Probabilities and Odds for Match 1',
                   showlegend=False,
                   hovermode='closest',
                   updatemenus=[dict(type='buttons', showactive=False,
                                y=1.05,
                                x=1.15,
                                xanchor='right',
                                yanchor='top',
                                pad=dict(t=0, r=10),
                                buttons=[dict(label='Play',
                                              method='animate',
                                              args=[None,
                                                    dict(frame=dict(duration=3,
                                                                    redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True,
                                                         mode='immediate')])])])


layout.update(xaxis=dict(range=[data.index[0], data.index[len(data)-1]], autorange=False),
              yaxis=dict(range=[0, 1], autorange=False))

fig = go.Figure(data=trace1, frames=frames, layout=layout)

fig.update_yaxes(
    ticktext=['0', '.25/+300', '.5/-100', '.75/-300', '1'],
    tickvals=['0', '.25', '.5', '.75', '1'],
    title='Player_1_WP'
)
fig.update_xaxes(title='Point Number in Match'
)


fig.show()