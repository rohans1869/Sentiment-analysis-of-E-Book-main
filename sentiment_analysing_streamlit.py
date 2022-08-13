
# Libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
lm= WordNetLemmatizer()
from nltk.tokenize import word_tokenize, sent_tokenize
import string
nltk.download("stopwords")
from nltk.probability import FreqDist
import string 
stopwords_extra = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
stopwords_all = stopwords.words("english")
stopwords_all.extend(stopwords_extra)
from tika import parser 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer=SentimentIntensityAnalyzer()

#title
st.title("Book Summary Extraction")

# Book data
book = st.sidebar.file_uploader("Upload PDF file")
if book != None:
    raw = parser.from_file(book)
    book_content = (raw['content'])
    sentence=nltk.sent_tokenize(book_content)

    list1=[]

    for i in range(0, len(sentence)):
        review=re.sub('[^a-zA-Z]',' ',sentence[i]).lower().split()
        review=[lm.lemmatize(word) for word in review if not word in stopwords_all]
        review=' '.join(review)
        list1.append(review)

    #Analyse sentiment
    df1=pd.DataFrame(list1)
    df1.columns = ["text"]

    scores=[]
    compound_list=[]
    positive_list=[]
    negative_list=[]
    neutral_list=[]

    for i in range(0, len(df1)):
        compound=analyzer.polarity_scores(df1['text'][i])['compound']
        pos=analyzer.polarity_scores(df1['text'][i])['pos']
        neu=analyzer.polarity_scores(df1['text'][i])['neu']
        neg=analyzer.polarity_scores(df1['text'][i])['neg']
        scores.append({'Compound':compound,'positive':pos,'neutral':neu,'negative':neg})

    sentiment_scores=pd.DataFrame.from_dict(scores)
    df1=df1.join(sentiment_scores)

    df1['Sentiment']=df1['Compound'].apply(lambda c:'neu' if c==0 else 'pos' if c > 0 else 'neg')

    value = [df1["Sentiment"].value_counts()[2],  
         df1["Sentiment"].value_counts()[1],  
         df1["Sentiment"].value_counts()[0]]



    negative = value[0]
    neutral  = value[1]
    positive = value[2]
        

    col1, col2, col3 = st.columns(3)
    
    col1.metric("Positive Sentences", '{:.1%}'.format(positive/(negative+positive+neutral))     , "")
    col2.metric("Negative Sentences", '{:.1%}'.format(negative/(negative+positive+neutral)), "")
    col3.metric("neutral Sentences",  '{:.1%}'.format(neutral/(negative+positive+neutral)), "")


    fig2 = plt.figure(figsize = (10, 5))
    plt.axis("equal")
    plt.title("Sentiment")
    plt.pie(value,labels=["Negative","neutral","Positive"], shadow=False, autopct='%2.1f%%',radius=1.2,explode=[0,0,0],counterclock=True, startangle=45)
    st.pyplot(fig2)