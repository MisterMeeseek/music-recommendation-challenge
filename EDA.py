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
import seaborn as sns

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

# Source_system_tab column - name of the system tab (group of system functions) where the replay event was triggered
train['source_system_tab'].unique()
train['source_system_tab'].isnull().sum()
train['source_system_tab'].value_counts().plot(kind = 'bar')

# Fill NaN's with blank string
train['source_system_tab'] = train['source_system_tab'].fillna('')

train.groupby(['source_system_tab', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

# Source_screen_name column - name of the layout a user sees
train['source_screen_name'].nunique()
train['source_screen_name'].value_counts()
train['source_screen_name'].isnull().sum()

train.groupby(['source_screen_name', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

# Source_type column - entry point a user first plays music on mobile apps
train['source_type'].nunique()
train['source_type'].value_counts()
train['source_type'].isnull().sum()

train.groupby(['source_type', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45 )

### Songs data ###

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
songs.head()
songs.isnull().sum()

# songs_id
songs['song_id'].nunique() # 2M+ unique Song ID's
songs['song_id'].isnull().sum() # 0 nulls

# song_length - in milliseconds
songs['song_length'].describe()
(songs['song_length'].min() / 1000) / 60
(songs['song_length'].mean() / 1000) / 60
(songs['song_length'].median() / 1000) / 60
(songs['song_length'].max() / 1000) / 60

songs[songs['artist_name'] == 'Bill Evans']    # just getting idea of genre id,
                                               # ex. 2122 probably = jazz
# genres - songs with more than 1 are separated with '|'
songs['genre_ids'].nunique()                # 1,045 unique genre ID's
songs['genre_ids'].isnull().sum()           # % of nulls to total is approx. 4.0%
songs['genre_ids']

# artist name
songs['artist_name'].nunique()  # 222,363 unique artist names
songs['artist_name'].isnull().sum() # 549 null values
temp_df = songs['artist_name'].drop_duplicates()
temp_df.size                        # only 1 duplicate

# how many songs does each artist have on the app?
songs.groupby('artist_name')['song_id'].count().nlargest(10) # Various Artists is a very common artist name
                                                             # Top 10 are almost all jazz and rock musicians

## Note: I'm skipping over the other variables for now and focusing on the ISRC feature
songs['isrc'].isnull().value_counts()       # not as many nulls as expected
songs['isrc'].nunique()                     # 1.8M unique codes


### Members data ###
members = pd.read_csv('members.csv')
members.head(10)
members.shape
members.size
members.columns
members.isnull().sum()

# msno - a subscribers' unique ID
members['msno'].nunique()                   # no duplicates
members['msno'].isnull().sum()              # no nulls

# City
members['city'].nunique()                   # 21 cities, integer based feature
members['city'].isnull().sum()              # no nulls
members['city'].value_counts()
#? What does the grouping on cities look like with respect to repeat plays or
#? some of the other features

# bd - age, outliers will be present
members['bd'].describe()
members['bd'].isnull().sum()
members[members['bd'] == 0]['bd'].count()   # lot of 0's so it may be worthwhile to impute this value based on the other features, will have to se if there's any relationships with the other features

members_bd = members[(members['bd'] > 0) & (members['bd'] <= 100)]['bd']

plt.xticks(rotation = 30)
sns.countplot(members_bd) # need to rotate axis and figure out index ordering
sns.kdeplot(members_bd)

# gender
members['gender'].isnull().value_counts()
members['gender'].fillna('unknown')
members['gender'].value_counts()        # another potential candidate for imputation
sns.countplot(members['gender'])

# registered_via
members['registered_via'].head()
members['registered_via'].isnull().sum()                            # no nulls
members[members['registered_via'] == 0]['registered_via'].count()   # no zeroes
members['registered_via'].nunique()                                 # 6 unique registration points
members['registered_via'].value_counts()                            
sns.countplot(members['registered_via'])         # 4, 7, and 9 are by far the top registration points

# registration_init_time - format is %Y%m%d
members['registration_init_time']
year_registered = members['registration_init_time'].astype(str).str[:4]
year_registered.nunique()
year_registered.value_counts()                  # 2016 saw biggest increase in registrations
sns.countplot(year_registered)

month_registered = members['registration_init_time'].astype(str).str[4:6]
month_registered.nunique()
month_registered.value_counts()
sns.countplot(month_registered)

day_registered = members['registration_init_time'].astype(str).str[6:8]
day_registered.nunique()
day_registered.value_counts()
sns.countplot(day_registered)

# expiration_date
members['expiration_date']
year_expired = members['expiration_date'].astype(str).str[:4]
year_expired.nunique()
year_expired.value_counts()
sns.countplot(year_expired)


### EDA on merged datasets to evaluate relationships between features and target ###

# merging the datasets into one
total_data = pd.merge(train,
                      members,
                      how = 'inner',
                      on = 'msno',
                      sort = False)

total_data = pd.merge(total_data,
                      songs,
                      how = 'inner',
                      on = 'song_id',
                      sort = False)

# Cross joint distributions/groupings of features vs. "target"

# column check
total_data.columns

# Examining differences in behavior and by regional (city) feature
total_data.groupby(['city', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45) # almost 50/50 split across all cities
total_data.groupby(['city', 'source_system_tab']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45)
total_data.groupby(['city', 'source_screen_name']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45)
total_data.groupby(['city', 'gender']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45)

# Examining differences in gender wrt target
total_data.groupby(['gender', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 45)
total_data['target'].groupby(total_data['gender']).value_counts() / total_data['target'].groupby(total_data['gender']).value_counts().sum()

# Source screen name wrt target
total_data.groupby(['source_screen_name', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 30)

# Source system tab wrt target
total_data.groupby(['source_system_tab', 'target']).size().unstack().plot(kind = 'bar', stacked = True, rot = 30)
total_data.groupby(['source_screen_name', 'source_system_tab']).size().unstack().plot(kind = 'bar', stacked = True, rot = 30)

# Song lengths and target
# first, I'll transform song_length to minutes
total_data['song_length_minutes'] = (total_data['song_length'] / 1000) / 60
print(total_data['song_length_minutes'])

sns.stripplot(x = 'target',
            y = 'song_length_minutes',
            jitter = True,
            data = total_data)

sns.swarmplot(x = 'target',
              y = 'song_length_minutes',
              data = total_data)

''' There a couple differences in the distribution of target variables across song length:
    1.) Clear break at 125 minutes above which repeat plays are consistent
    2.) Interesting concentration in both classes at around 60 minutes. I wonder
        if these are specific content formats like a podcast?
    3.) Most songs are at length 30 minutes or less '''
    
''' Let's try the above data again with a swarmplot instead, see if we can get
more insight to the distribution '''    