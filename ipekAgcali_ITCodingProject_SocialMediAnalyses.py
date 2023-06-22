#For Gucci Instagram page, I want to compare last week and this week comments and likes counts for feeds and reels.
#Before starting my code, I want to explain how to gather my code from META API
#I created facebook account and instagram account.
#By using the facebook code, I opened the new page and by using Facebook business I connected this page with my facebook acoount.
#I went to developers.facebook.com then I created new project. During creation, I adjuested everything for Instagram.
#After creation, I clicked Tools-->Graph API Explorer. Left hand side, I gave permissions and I generated Acsses Token.
#In the submit area, I put "me?fields=id,name,accounts{instagram_business_account}", and I obtained my instagram page id number.The second id is used and labelled "instagram_business_account"
#Both Accses Token and instagram page id number are used for code.
#Also I found query from https://developers.facebook.com/docs/instagram-api/guides/business-discovery.

#First two is used gathering data from API
import requests
import json

#This is used for distinguishing time(this week and last week)
from datetime import datetime, timedelta, timezone

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


class Brand_Selection:
  def __init__(self, access_token, instagram_id, brand):
    self.access_token = access_token
    self.instagram_id = instagram_id
    self.brand = brand

  def brands_calling(self):
    api_version = 'v17.0'
    url = f'https://graph.facebook.com/{api_version}/{self.instagram_id}'

    params = {
        'access_token': self.access_token,
        'fields': 'business_discovery.username('+ self.brand +'){username,name,ig_id,id,follows_count,followers_count,media_count,media{like_count,comments_count,media_product_type,timestamp}}'
            }

    response = requests.get(url, params=params)
    data = response.json()
    #print(data)
    
    media_data = data['business_discovery']['media']['data']
    current_time = datetime.now(timezone.utc)
    one_week_ago = current_time - timedelta(weeks=1)

    #they are created for likes
    this_week_likes_feeds = []
    last_week_likes_feeds = []
    this_week_likes_reels = []
    last_week_likes_reels = [] 

    #they are created for comments
    this_week_comments_feeds=[]
    this_week_comments_reels=[]
    last_week_comments_feeds=[]
    last_week_comments_reels=[]

    
    for media in media_data:
      like_count = media['like_count']
      comment_count = media['comments_count']
      timestamp = datetime.strptime(media['timestamp'], '%Y-%m-%dT%H:%M:%S%z')
      media_type = media['media_product_type']
  
    #First if used for checking the week
    #Second if used for cheking feed(post) or reels
      
      if timestamp >= one_week_ago:   

        if media_type == 'FEED':
          this_week_likes_feeds.append(like_count)
          this_week_comments_feeds.append(comment_count)         
        elif media_type == 'REELS':
          this_week_likes_reels.append(like_count)
          this_week_comments_reels.append(comment_count)
         
      else:
        if media_type == 'FEED':
          last_week_likes_feeds.append(like_count)
          last_week_comments_feeds.append(comment_count)     
        elif media_type == 'REELS':
          last_week_likes_reels.append(like_count)
          last_week_comments_reels.append(comment_count)
          

    # Equalize the lengths of the lists by using 0. I added 0 because If there is no data, dataframe stop to take data. This creates a problem.Because if second data is none, my dataframe has only one data. 
    max_len = max(len(last_week_likes_feeds), len(last_week_likes_reels), len(this_week_likes_feeds), len(this_week_likes_reels),
                  len(last_week_comments_feeds), len(last_week_comments_reels), len(this_week_comments_feeds), len(this_week_comments_reels))
    last_week_likes_feeds += [0] * (max_len - len(last_week_likes_feeds))
    last_week_likes_reels += [0] * (max_len - len(last_week_likes_reels))
    this_week_likes_feeds += [0] * (max_len - len(this_week_likes_feeds))
    this_week_likes_reels += [0] * (max_len - len(this_week_likes_reels))
    last_week_comments_feeds += [0] * (max_len - len(last_week_comments_feeds))
    last_week_comments_reels += [0] * (max_len - len(last_week_comments_reels))
    this_week_comments_feeds += [0] * (max_len - len(this_week_comments_feeds))
    this_week_comments_reels += [0] * (max_len - len(this_week_comments_reels))


    #My aim is that I created one data frame which includes lastweek and this week likes and comments count because I want to make clustering.     

     #creation dataFrames for Likes
    this_weekLikes_feed_df = pd.DataFrame(this_week_likes_feeds)
    last_weekLikes_feed_df = pd.DataFrame(last_week_likes_feeds)
    this_weekLikes_reel_df=pd.DataFrame(this_week_likes_reels)
    last_weekLikes_reel_df = pd.DataFrame(last_week_likes_reels)
   
    #creation dataFrames for Comments
    this_weekComments_feed_df = pd.DataFrame(this_week_comments_feeds)
    last_weekComments_feed_df = pd.DataFrame(last_week_comments_feeds)
    this_weekComments_reel_df=pd.DataFrame(this_week_comments_reels)
    last_weekComments_reel_df = pd.DataFrame(last_week_comments_reels)

    
    #combination of Data Frames this week LIKES for reels&feeds(tw=This Week)
    thisWeekLikes= pd.concat([this_weekLikes_feed_df,this_weekLikes_reel_df],axis=1,join="inner")
    thisWeekLikes.columns=["Feed Likes Count(tw)","Reel Likes Count(tw)"]

    #combination of Data Frames this week COMMENTS for reels&feeds(tw=This Week)
    thisWeekComments=pd.concat([this_weekComments_feed_df,this_weekComments_reel_df],axis=1,join="inner")
    thisWeekComments.columns=["Feed Comments Count(tw)","Reel Comments Count(tw)"]
    
    #combination of Data Frames last week LIKES for reels&feeds(lw=Last Week)
    lastWeekLikes=pd.concat([last_weekLikes_feed_df,last_weekLikes_reel_df],axis=1,join="inner")
    lastWeekLikes.columns=["Feed Likes Count(lw)","Reel Likes Count(lw)"]
    
    #combination of Data Frames this week COMMENTS for reels&feeds(lw=This Week)
    lastWeekComments=pd.concat([last_weekComments_feed_df,last_weekComments_reel_df],axis=1,join="inner")
    lastWeekComments.columns=["Feed Comments Count(lw)","Reel Comments Count(lw)"]
    
    #combination of this weeks 
    thisWeek=pd.concat([thisWeekLikes,thisWeekComments],axis=1,join="inner")
  
    #combination of Last Week
    lastWeek=pd.concat([lastWeekLikes,lastWeekComments],axis=1,join="inner")

    #combination of thisWeek and Lastweek
    df=pd.concat([thisWeek, lastWeek],axis=1,join="inner")
    print(df) 

    
    #Firstly I create DBSCAN clustering for Reels
    reels_data = df[['Reel Comments Count(lw)', 'Reel Likes Count(lw)', 'Reel Comments Count(tw)', 'Reel Likes Count(tw)']]

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Applying DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    clusters = dbscan.fit_predict(scaled_data)

    # Plotting the graph for Reels
    plt.scatter(df['Reel Comments Count(tw)'], df['Reel Likes Count(tw)'], c=clusters, cmap='viridis', label='This Week')
    plt.scatter(df['Reel Comments Count(lw)'], df['Reel Likes Count(lw)'], c=clusters, cmap='plasma', marker='x',label='Last Week')
    plt.xlabel('Reel Comments Count')
    plt.ylabel('Reel Likes Count')
    plt.title("Comparison for Reel with DBScan")
    plt.legend()
    plt.show()

    # Plotting the graph for Reels
    plt.scatter(df['Feed Comments Count(tw)'], df['Feed Likes Count(tw)'], c=clusters, cmap='viridis', label='This Week')
    plt.scatter(df['Feed Comments Count(lw)'], df['Feed Likes Count(lw)'], c=clusters, cmap='plasma', marker='x',label='Last Week')
    plt.xlabel('Feed Comments Count')
    plt.ylabel('Feed Likes Count')
    plt.title("Comparison for Feed with DBScan")
    plt.legend()
    plt.show()

    


  

my_brands=Brand_Selection("Accses Token","instagram_business_account ID"","gucci") #I choose gucci insagram page. You can use which public page do you want. Just enter the page name. 
my_brands.brands_calling()