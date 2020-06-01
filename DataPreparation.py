import re
import datetime
from datetime import date
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer


class DataPreparation:


    def __init__(self, 
                 df):   

        '''
        Intake the event data; print out summary and build the first level cleaned data
        '''

        print("Numbers of rows in the data: {:,}".format(df.count()))
        print("Numbers of rows with N/A user Ids: {:,}".format(df.filter("userId is Null").count()))
        print("Numbers of unique customers: {:,}".format(df.select('userId').distinct().count()))

        self.df = df
        self.clean_userAgent = udf(self.clean_userAgent, ArrayType(StringType()))
        self.clean_ts = udf(self.clean_ts, StringType())
        self.clean_ts_hour = udf(self.clean_ts_hour, StringType())
        self.extract_state = udf(self.extract_state, ArrayType(StringType()))

        self.user_activities = self.df\
                                .filter("auth = 'Logged In'")\
                                .select('userId', 
                                        'registration',
                                        self.clean_ts('registration').alias("registration_ts"),
                                        'level',
                                        F.col('ts').alias('time'),
                                        self.clean_ts('ts').alias("timestamp"),
                                        self.clean_ts_hour('ts').alias("time_in_the_day"),
                                        'sessionId',
                                        'itemInSession',
                                        'page',
                                        'artist',
                                        F.concat(F.col('artist'), F.lit(" - ") , F.col('song')).alias("singer-song"),
                                        'length'
                                        )\
                                .withColumn('days_since_registration', 
                                            F.datediff(F.to_date('timestamp',"yyyy-MM-dd"), 
                                          F.to_date('registration_ts',"yyyy-MM-dd")))
        self.max_time = self.user_activities\
                        .groupby()\
                        .max('time')\
                        .select(self.clean_ts('max(time)').alias('max_time'))\
                        .collect()[0]['max_time']

        self.user_activities = self.user_activities.withColumn('days_before_today', 
                                                               F.datediff(F.to_date(F.lit(self.max_time),"yyyy-MM-dd"), 
                                                                          F.to_date('timestamp',"yyyy-MM-dd")))

    @staticmethod
    def clean_userAgent(x):
      '''
      Extract agents from the userAgent column;
      Example:
      "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103 Safari/537.36"
      --> ['Mozilla','AppleWebKit','Chrome','Safari']
      '''
      try:
        x = re.sub("[\(\[].*?[\)\]]", "", x)
        x = x.replace("\"", "").split()
        x = [i.split('/')[0] for i in x if i != 'like']
        return x
      except:
        return None


    @staticmethod
    def one_hot_encode(df, col):
      '''
      one-hot encode a categorical column (long table to wide table);
      the function will add the one hot encoding columns to the original dataframe and will drop the original categorical column;
      Input:
      df: the dataframe; should be deduped
      col: the categorical column
      '''
      col_unique_values = [i[col] for i in df.select(col).distinct().collect()]
      other_cols = [i for i in df.columns if i != col]
      for i in col_unique_values:
        df = df.withColumn(col + '_' + i, F.when(F.col(col)==i, 1).otherwise(0))
      df = df.drop(col).groupby(other_cols).sum()
      for c in df.columns:
        df = df.withColumnRenamed(c, c.replace("sum(", "").replace(")", ""))
      return df


    @staticmethod
    def clean_ts(timestamp):
        '''
        Convert unix timestamp to "YYYY-mm-dd" format
        '''
        try:
          return datetime.datetime.fromtimestamp(int(timestamp)/1000).strftime("%Y-%m-%d")
        except:
          return None 


    @staticmethod
    def clean_ts_hour(timestamp):
        '''
        Get the bucket hour:
        1am ~ 6am
        6am ~ 12pm
        12pm ~ 6pm
        6pm ~ 1am
        '''
        try:
          hour = int(datetime.datetime.fromtimestamp(int(timestamp)/1000).strftime("%H"))
          if hour >= 1 and hour <= 6:
            return "1am ~ 6am"
          elif hour > 6 and hour <= 12:
            return "6am ~ 12pm"
          elif hour > 12 and hour <= 18:
            return "12pm ~ 6pm"
          else:
            return "6pm ~ 1am"
        except:
          return None


    @staticmethod
    def extract_state(x):
      '''
      Extract states from location column
      '''
      try:
        state_list = x.split(",")[1].split("-")
        return list(set([i.replace(' ','') for i in state_list]))
      except:
        return None


    def build_static_features(self):
      '''
      Features:

      gender
      state
      useragent
      customer_age (for how long the customer has joined)
      '''
      gender = self.df.select('userId', F.when(F.col('gender')=="F", 1).otherwise(0).alias('female')).distinct()

      state = self.one_hot_encode(self.df.select('userId', F.explode(self.extract_state('location')).alias('state')).distinct(), 'state')

      customer_age = self.df\
                     .select('userId',self.clean_ts('registration').alias("registration"))\
                     .select('userId',
                             F.datediff(F.to_date(F.lit(self.max_time),"yyyy-MM-dd"), 
                                        F.to_date('registration',"yyyy-MM-dd")).alias('customer_age'))\
                     .distinct()

      agent_one_hot = self.one_hot_encode(self.df\
                                          .select('userId', F.explode(self.clean_userAgent('userAgent')))\
                                          .distinct()\
                                          .withColumnRenamed('col','agent'), 
                                          'agent')
      return gender\
            # .join(state, on="userId", how='outer')\
            .join(customer_age, on="userId", how='outer')\
            .join(agent_one_hot, on="userId", how='outer')


    def build_dynamic_featues(self,
                              max_days_before_today=None,
                              min_days_before_today=None):
      '''
      Features:
      
      total sessions
      total items
      active days
      total active time (milliseconds)

      average sessions per day
      average time per day (milliseconds)
      distributions of time in the day (distribution of time spent in these 4 time blocks: 
      1am ~ 6am, 6am ~ 12pm, 12pm ~ 6pm, 6pm ~ 1am)

      average items per session
      average active time per session

      numbers of unique songs
      numbers of unique singers
      largest song time percentage (calculate the time distribution among the songs that the 
      customer had listened to and pick the largest percentage)
      largest singer percentage

      numbers thumbsup
      numbers thumbsdown
      numbers add playlist
      numbers add friend
      numbers error
      perc thumbsup
      perc thumbsdown
      perc add playlist
      perc add friend

      perc popular songs (percentage of time spent on popular songs; popular songs: ranked by 
      the numbers of unique listeners and pick the ones ranked in the top 50%)

      last_status (whether the customer is in the 'paid' level on the last day)
      perc_paid_days
      paid_days
      '''
      if max_days_before_today is None:
        user_activities = self.user_activities
      else:
        user_activities = self.user_activities.filter("days_before_today <= {}".format(max_days_before_today))

      if min_days_before_today is None:
        user_activities = user_activities
      else: 
        user_activities = user_activities.filter("days_before_today >= {}".format(min_days_before_today))

      #Use time
      total_sessions = user_activities\
                       .select('userId','sessionId')\
                       .distinct()\
                       .groupby('userId')\
                       .count()\
                       .withColumnRenamed('count','sessions')

      total_items = user_activities\
                    .groupby('userId')\
                    .count()\
                    .withColumnRenamed('count','items')

      active_days = user_activities\
                    .select('userId','timestamp')\
                    .distinct()\
                    .groupby('userId')\
                    .count()\
                    .withColumnRenamed('count','active_days')

      total_active_time = user_activities\
                          .groupby(['userId','sessionId'])\
                          .agg(F.max('time').alias('max'), F.min('time').alias('min'))\
                          .selectExpr('userId','max - min as active_time')\
                          .groupby('userId')\
                          .agg(F.sum('active_time').alias('active_time'))

      sessions_per_day = user_activities\
                         .select('userId','timestamp','sessionId')\
                         .distinct()\
                         .groupby('userId','timestamp')\
                         .count()\
                         .groupby('userId').agg(F.avg('count').alias('sessions_per_day'))

      time_per_day = user_activities\
                    .groupby(['userId','timestamp','sessionId'])\
                    .agg(F.max('time').alias('max'), F.min('time').alias('min'))\
                    .selectExpr('userId','timestamp','max - min as active_time')\
                    .groupby(['userId','timestamp'])\
                    .agg(F.sum('active_time'))\
                    .groupby('userId')\
                    .agg(F.mean('sum(active_time)').alias('time_per_day'))

      avg_items_per_session = user_activities\
                             .groupby(['userId','sessionId'])\
                             .count()\
                             .groupby('userId')\
                             .agg(F.avg('count').alias('avg_items_per_session'))

      average_time_per_session = user_activities\
                                 .groupby(['userId','sessionId'])\
                                 .agg(F.max('time').alias('max'), F.min('time').alias('min'))\
                                 .selectExpr('userId','max - min as active_time')\
                                 .groupby('userId')\
                                 .agg(F.avg('active_time').alias('time_per_session'))

      time_distribution = user_activities\
                          .groupby('userId','sessionId','time_in_the_day')\
                          .agg(F.max('time').alias('max'), F.min('time').alias('min'))\
                          .selectExpr('userId','time_in_the_day','max - min as active_time')\
                          .groupby(['userId','time_in_the_day'])\
                          .agg(F.sum('active_time'))\
                          .withColumn('1am ~ 6am', 
                                      F.when(F.col('time_in_the_day') == '1am ~ 6am', F.col('`sum(active_time)`')).otherwise(0))\
                          .withColumn('6am ~ 12pm', 
                                      F.when(F.col('time_in_the_day') == '6am ~ 12pm', F.col('`sum(active_time)`')).otherwise(0))\
                          .withColumn('12pm ~ 6pm', 
                                      F.when(F.col('time_in_the_day') == '12pm ~ 6pm', F.col('`sum(active_time)`')).otherwise(0))\
                          .withColumn('6pm ~ 1am', 
                                      F.when(F.col('time_in_the_day') == '6pm ~ 1am', F.col('`sum(active_time)`')).otherwise(0))\
                          .withColumn('x', 
                                      F.col('1am ~ 6am')+F.col('6am ~ 12pm')+F.col('12pm ~ 6pm')+F.col('6pm ~ 1am'))\
                          .groupby('userId')\
                          .sum()\
                          .selectExpr('userId', 
                                      '`sum(1am ~ 6am)`/ `sum(x)` as perc_1_6',
                                      '`sum(6am ~ 12pm)` / `sum(x)` as perc_6_12',
                                      '`sum(12pm ~ 6pm)` / `sum(x)` as perc_12_18',
                                      '`sum(6pm ~ 1am)` / `sum(x)` as perc_18_1')

      unique_songs = user_activities\
                     .select('userId','singer-song')\
                     .distinct()\
                     .groupby('userId')\
                     .agg(F.count('singer-song').alias('unique_songs'))

      unique_singers = user_activities\
                       .select('userId','artist')\
                       .distinct()\
                       .groupby('userId')\
                       .agg(F.count('artist').alias('unique_artists'))

      song_perc =  user_activities\
                    .groupby(['userId','singer-song'])\
                    .agg(F.sum('length').alias('song_time'))\
                    .select('userId',
                            'song_time',
                            F.sum('song_time').over(Window.partitionBy('userId')).alias('total_time'))\
                    .selectExpr('userId',
                                'song_time/total_time as perc')\
                    .groupby('userId')\
                    .agg(F.max('perc').alias('max_song_perc'))

      singer_perc = user_activities\
                    .filter("`singer-song` is not Null and page == 'NextSong'")\
                    .groupby(['userId','artist'])\
                    .agg(F.sum('length').alias('artist_time'))\
                    .select('userId',
                            'artist_time',
                            F.sum('artist_time').over(Window.partitionBy('userId')).alias('total_time'))\
                    .selectExpr('userId',
                                'artist_time/total_time as perc')\
                    .groupby('userId')\
                    .agg(F.max('perc').alias('max_artist_perc'))

      numbers_actions = user_activities\
                        .select('userId',
                                F.when(F.col('page') == 'Thumbs Up', 1).otherwise(0).alias('thumbsup'),
                                F.when(F.col('page') == 'Thumbs Down', 1).otherwise(0).alias('thumbsdown'),
                                F.when(F.col('page') == 'Add to Playlist', 1).otherwise(0).alias('add_playlist'),
                                F.when(F.col('page') == 'Add Friend', 1).otherwise(0).alias('add_friend'),
                                F.when(F.col('page') == 'Error', 1).otherwise(0).alias('error'),
                                F.lit(1).alias('x'))\
                        .groupby('userId')\
                        .sum()\
                        .selectExpr('userId',
                                    '`sum(thumbsup)` as numbers_thup',
                                    '`sum(thumbsdown)` as numbers_thdn',
                                    '`sum(add_playlist)` as numbers_addlist',
                                    '`sum(add_friend)` as numbers_addfrd',
                                    '`sum(error)` as numbers_error',
                                    '`sum(thumbsup)`/`sum(x)` as perc_thup',
                                    '`sum(thumbsdown)`/`sum(x)` as perc_thdn',
                                    '`sum(add_playlist)`/`sum(x)` as perc_addlist',
                                    '`sum(add_friend)`/`sum(x)` as perc_addfrd')

      song_popularity = user_activities\
                        .filter("`singer-song` is not Null and page == 'NextSong'")\
                        .select('userId','singer-song')\
                        .distinct()\
                        .groupby('singer-song')\
                        .count()

      most_popular_bar = song_popularity.approxQuantile('count', [0.5], 0.25)[0]

      popular_songs = song_popularity.filter("count >= {}".format(most_popular_bar))

      popular_songs_perc = user_activities\
                          .filter("`singer-song` is not Null and page == 'NextSong'")\
                          .join(popular_songs, on="singer-song", how="left")\
                          .select('userId',
                                  F.when(F.col('count').isNotNull(), 1).otherwise(0).alias('popular'),
                                  F.lit(1).alias('x'))\
                          .groupby('userId')\
                          .sum()\
                          .selectExpr('userId','`sum(popular)`/`sum(x)` as perc_popular_songs')

      last_status = user_activities\
                    .withColumn('rownum', F.row_number().over(Window.partitionBy("userId").orderBy(F.col('time').desc())))\
                    .filter('rownum==1')\
                    .select('userId',
                            F.when(F.col('level')=='paid', 1).otherwise(0).alias('last_status_paid'))

      perc_paid_days = user_activities\
                       .select('userId',
                               'timestamp',
                               F.when(F.col('level')=='paid', 1).otherwise(0).alias('paid'),
                               F.lit(1).alias('x'))\
                       .distinct()\
                       .groupby('userId')\
                       .agg(F.sum('paid').alias('paid_days'), F.sum('x').alias('total_days'))\
                       .selectExpr('userId', 'paid_days', 'paid_days/total_days as perc_paid_days')

      return total_sessions\
              .join(total_items, on="userId", how='outer')\
              .join(active_days, on="userId", how='outer')\
              .join(total_active_time, on="userId", how='outer')\
              .join(sessions_per_day, on="userId", how='outer')\
              .join(time_per_day, on="userId", how='outer')\
              .join(avg_items_per_session, on="userId", how='outer')\
              .join(average_time_per_session, on="userId", how='outer')\
              .join(time_distribution, on="userId", how='outer')\
              .join(unique_songs, on="userId", how='outer')\
              .join(unique_singers, on="userId", how='outer')\
              .join(song_perc, on="userId", how='outer')\
              .join(numbers_actions, on="userId", how='outer')\
              .join(popular_songs_perc, on="userId", how='outer')\
              .join(last_status, on="userId", how='outer')\
              .join(perc_paid_days, on="userId", how='outer')


    def run(self,
            max_days_before_today=None,
            min_days_before_today=None):

        static = self.build_static_features()
        dynamic = self.build_dynamic_featues(max_days_before_today, min_days_before_today)
        return static.join(dynamic, on="userId", how='outer')