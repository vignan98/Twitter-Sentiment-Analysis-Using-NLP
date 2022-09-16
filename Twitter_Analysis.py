#!/usr/bin/env python
# coding: utf-8

# In[118]:


import twint
import pandas as pd
import nest_asyncio  
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import re
from textblob import TextBlob


# In[119]:


nest_asyncio.apply()
airline_search = {"American Airlines":"AmericanAirlines",
                  "American Airlines":"americanairlines",
                  "American Airlines":"Americanairlines",
                  "American Airlines":"americanAirlines",
                "Southwest Airlines" : "SouthwestAirlines",
                "Southwest Airlines":"southwestairlines",
                "Southwest Airlines" : "Southwestairlines",
                "Southwest Airlines" : "southwestAirlines",
                 "JetBlue Airlines": "jetblueairlines",
                 "JetBlue Airlines":"JetBlueAirlines",
                  "JetBlue Airlines": "jetblueAirlines",
                  "JetBlue Airlines": "Jetblueairlines",
                  "United Airlines": "UnitedAirlines",
                  "United Airlines":"unitedairlines",
                  "United Airlines": "Unitedairlines",
                  "United Airlines": "unitedAirlines",
                  "United Airlines": "unitedAIRLINES",
                  "Delta Airlines" : "DeltaAirlines",
                   "Delta Airlines" : "deltaairlines",
                 "Delta Airlines" : "Deltaairlines",
                  "Delta Airlines" : "deltaAirlines",}
def twintConfig(since,until, search_string):
    c = twint.Config()
    c.Search = search_string[1]
    c.Since = since
    c.Until = until
    c.Pandas = True
    twint.run.Search(c)


# In[120]:


since = input("Input a start date eg 2021-09-17: ")
until = input("Input an end date eg 2021-09-18: ")
def Run_Twint(search_vals):
    
    #set empty dataframe for join
    out_df= pd.DataFrame()
    
    for airline in search_vals.items():
        print ("running for search item: "+airline[0]+"\n")
        print ("Search string: "+airline[1]+"\n")
        
        #run twint
        twintConfig(since,until, airline)
        
        #get dataframe from twint output
        tweets_df = twint.storage.panda.Tweets_df
        
        #join Dataframes and create 'Bank' column
        tweets_df["Bank"]= airline[0]
        out_df = pd.concat([out_df,tweets_df])
        
    return out_df
tweets_df= Run_Twint(airline_search)


# In[121]:


len(tweets_df)


# In[122]:


tweets_df.columns


# In[123]:


df1=tweets_df[['id','username','date','tweet','Bank']]


# In[124]:


def clean_text(text):  
    pat1 = r'@[^ ]+'                   
    pat2 = r'https?://[A-Za-z0-9./]+'  
    pat3 = r'\'s'                      
    pat4 = r'\#\w+'                     
    pat5 = r'&amp '                     
    pat6 = r'[^A-Za-z\s]'               
    combined_pat = r'|'.join((pat1, pat2,pat3,pat4,pat5, pat6))
    text = re.sub(combined_pat,"",text).lower()
    return text.strip()


# In[125]:


df1["tweet"] = df1["tweet"].apply(clean_text)


# In[126]:


df2 = df1.loc[df1["tweet"] !=""]


# In[127]:


df3=df2


# In[128]:


df3.head()


# In[129]:


for i in range(len(df3)):
    tweet = df3.iloc[i,2]
    analysis= TextBlob(tweet)
    print(analysis.sentiment)


# In[130]:


l1=[]
l2=[]
for i in range(len(df3)):
    tweet = df3.iloc[i,2]
    analysis= TextBlob(tweet)
    l1.append(analysis.sentiment[0])
    l2.append(analysis.sentiment[1])


# In[131]:


df4=df3


# In[132]:


df3['Polarity']=np.array(l1)
df3['Subjectivity']=np.array(l2)


# In[133]:


df4.head()


# In[134]:


for i in range(len(df4)):
 print(df4['Polarity'].iloc[i])


# In[135]:


df4['Sentiment']=""


# In[136]:


i=0
for i in range(len(df4)):
    if df4['Polarity'].iloc[i]>0:
        df4['Sentiment'].iloc[i] = "Positive"
    elif df4['Polarity'].iloc[i]<0:
        df4['Sentiment'].iloc[i] = "Negative"
    else:
        df4['Sentiment'].iloc[i] = "Neutral"


# In[137]:


df4.head()


# In[138]:


negative_tweets= df4[df4['Sentiment']=='Negative']


# In[139]:


len(negative_tweets)


# In[140]:


import nltk


bad_words=[]
for i in range(len(negative_tweets)):
    dictt={}
    l1=nltk.word_tokenize(negative_tweets['tweet'].iloc[i])
    for j in range(len(l1)):
        a=TextBlob(l1[j])
        dictt[l1[j]]= a .sentiment[0]
    
    print(dictt)
    for k in dictt:
        if dictt[k] <0:
            bad_words.append(k)
    


# In[25]:


bad_words
from collections import Counter
Counter(bad_words) 


# In[26]:


negative_tweets  


# In[136]:


from nltk.stem import WordNetLemmatizer

#Apply tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
negative_tweets['tokenized'] = negative_tweets['tweet'].apply(lambda x: tokenization(x.lower()))
#Removing Stop words
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
negative_tweets['nonstop'] = negative_tweets['tokenized'].apply(lambda x:remove_stopwords(x))
#Stemmer
#ps = nltk.PorterStemmer()
#def stemming(text):
 #   text = [ps.stem(word) for word in text]
  #  return text
#negative_tweets['stemmed'] = negative_tweets['nonstop'].apply(lambda x: stemming(x))
lemmatizer = WordNetLemmatizer()
def lematizing(text):
    text= [lemmatizer.lemmatize(word,pos='v') for word in text]
    return text
negative_tweets['lematized'] = negative_tweets['nonstop'].apply(lambda x: lematizing(x))
#join all the words to make a final text field
negative_tweets['final'] = negative_tweets['lematized'].apply(lambda x: ' '.join(x))
negative_tweets.head()


# In[137]:


pd.options.display.max_colwidth=1000
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
negative_tweets[['tweet','final']]


# In[138]:


negative_tweetss=negative_tweets.copy(deep=True)


# In[139]:


negative_tweetss


# In[140]:


tags=['delay','service','layover','cancel','luggage','bag','booking','damage','seat','refund','cleaning','website','pilot','plane','clean','wait','staff']


# In[141]:


negative_tweetss['Negative_reason']=""


# In[142]:


def isWordPresent(sentence, word):
     
    # To break the sentence in words
    s = sentence.split(" ")
 
    for i in s:
 
        # Comparing the current word
        # with the word to be searched
        if (i == word):
            return True
    return False


# In[143]:


j=0
for word in tags:
    for k in range(len(negative_tweetss)):
        if (isWordPresent(negative_tweetss['final'].iloc[k],word)== True):
            negative_tweetss['Negative_reason'].iloc[k]=word


# In[144]:


negative_tweetss


# In[145]:


for i in range(len(negative_tweetss)):
    if negative_tweetss['Negative_reason'].iloc[i]=="" :
        negative_tweetss['Negative_reason'].iloc[i]='unknown'


# In[146]:


len(df4)


# In[147]:


df1=df4[['id','username','date','tweet','Bank','Sentiment']]


# In[148]:


dfp=df1[df1['Sentiment'].isin(['Positive','Neutral'])]


# In[149]:


len(dfp)


# In[150]:


dfp['Negative_reason']=""


# In[151]:


dfp


# In[152]:


dfn=negative_tweetss[['id','username','date','tweet','Bank','Negative_reason']]


# In[153]:


dfn['Sentiment']='Negative'


# In[154]:


df_n=dfn.iloc[:,[0,1,2,3,5,4]]


df_n




# In[155]:


frames = [dfp, df_n]
  
result = pd.concat(frames)


# In[156]:


result.sort_index(inplace=True)


# In[157]:


len(result)


# In[158]:


result.index = range(7465)


# In[159]:


result.loc[(result['Negative_reason'] =='bag'), 'Negative_reason'] = "luggage"
result.loc[(result['Negative_reason'] =='delay'), 'Negative_reason'] = "delayed flight"
result.loc[(result['Negative_reason'] =='service'), 'Negative_reason'] = "bad service"
result.loc[(result['Negative_reason'] =='cancel'), 'Negative_reason'] = "cancelled flight"
result.loc[(result['Negative_reason'] =='booking'), 'Negative_reason'] = "flight booking"
result.loc[(result['Negative_reason'] =='seat'), 'Negative_reason'] = "seat issue"
result.loc[(result['Negative_reason'] =='refund'), 'Negative_reason'] = "refund issue"
result.loc[(result['Negative_reason'] =='website'), 'Negative_reason'] = "website issue"
result.loc[(result['Negative_reason'] =='plane'), 'Negative_reason'] = "plane condition"
result.loc[(result['Negative_reason'] =='wait'), 'Negative_reason'] = "waiting time"
result.loc[(result['Negative_reason'] =='staff'), 'Negative_reason'] = "delayed flight"


# In[160]:


result


# In[161]:


result.groupby('Negative_reason').count()


# In[162]:


result.to_csv("tweets11.csv")


# In[166]:


len(result)


# In[1]:


df3 = pd.merge(tweets_df, result, left_on=['id', 'date'], right_on=['id', 'date'], how='inner')


# In[ ]:




