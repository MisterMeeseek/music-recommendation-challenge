#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:33:53 2019

@author: Joel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs_data = pd.read_csv('songs.csv')
songs_extra = pd.read_csv('song_extra_info.csv')
members = pd.read_csv('members.csv')

train.head(10)
train.columns
train.shape

# Number of unique users
train['msno'].nunique()                         # 30,755 unique users
train['msno'].isnull().sum()

# Number of unique songs
train['song_id'].nunique()                      # 359,966 unique songs
train['song_id'].isnull().sum()

songs_data['song_id'].nunique()
songs_data['song_id'].isnull().sum()

test['song_id'].nunique()
test['song_id'].isnull().sum()

''' 
    There appears to be a significant difference between the number of unique 
    song ID's in the train and songs data files. Not sure what impact this may 
    have or what may be driving the difference right now. First thought is 
    mislabeling the song_id's. Or, re-publishing the same song under different ID's. 
'''

# Source_system_tab column
train['source_system_tab'].nunique()
train['source_system_tab'].isnull().sum()
train['source_system_tab'].value_counts().plot(kind = 'bar')

sst_by_target = train.groupby('target')['source_system_tab'].value_counts() # clear differences in target values based on the SST

ind = np.arange(len(sst_by_target[1].index))
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.35

p1 = ax.bar(ind, sst_by_target[0], width, color = 'red')
p2 = ax.bar(ind, sst_by_target[1], width, color = 'green', bottom = sst_by_target[1])

ax.set_ylabel('Count')
ax.set_xlabel('SST Groups')
ax.set_title('Target Classes by SST Groups')

ax.set_xticks(ind + width/2.)
ax.set_yticks(np.arange(sst_by_target.min(), sst_by_target.max(), 100000))
ax.set_xticklabels(sst_by_target[0].index.tolist())

plotly_fig = tls.mpl_to_plotly(fig)

plotly_fig['layout']['showlegend'] = True
plotly_fig['data'][0]['name'] = 'Not Repeated'
plotly_fig['data'][1]['name'] = 'Repeated'
plt.plot(plotly_fig, filename = 'stacked-bar-chart')

# Source_screen_name column
train['source_screen_name'].nunique()
train['source_screen_name'].value_counts()
train['source_screen_name'].isnull().sum()

# Source_type column
train['source_type'].nunique()
train['source_type'].value_counts()
train['source_type'].isnull().sum()

### Songs data

# Merge the two songs data files
songs_data.shape
songs_extra.shape
songs = pd.merge(songs_data,
                 songs_extra,
                 how = 'outer', 
                 on = 'song_id',
                 sort = False)
songs.shape
songs.columns

# songs_id
songs['song_id'].nunique()
songs['song_id'].isnull().sum()

# song_length
songs['song_length'].describe()
(songs['song_length'].min() / 1000) / 60
(songs['song_length'].mean() / 1000) / 60
(songs['song_length'].median() / 1000) / 60
(songs['song_length'].max() / 1000) / 60

songs[songs['artist_name'] == 'Bill Evans']    # just getting idea of genre id,
                                               # ex. 2122 probably = jazz

# genres
songs['genre_ids'].nunique()                # 1,045 unique genre ID's
songs['genre_ids'].isnull().sum()           # % of nulls to total is approx. 4.0%
songs['genre_ids']