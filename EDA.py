#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:33:53 2019

@author: Joel
"""
# Import needed libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Load in data files
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
train['source_system_tab'].unique()
train['source_system_tab'].isnull().sum()
train['source_system_tab'].value_counts().plot(kind = 'bar')

# Fill NaN's with blank string
train['source_system_tab'] = train['source_system_tab'].fillna('')

train.groupby(['source_system_tab', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

# Source_screen_name column
train['source_screen_name'].nunique()
train['source_screen_name'].value_counts()
train['source_screen_name'].isnull().sum()

train.groupby(['source_screen_name', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

# Source_type column
train['source_type'].nunique()
train['source_type'].value_counts()
train['source_type'].isnull().sum()

train.groupby(['source_type', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

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