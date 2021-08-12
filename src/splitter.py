# create train test and valid
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import re, string, unicodedata

import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import Counter
import datetime
from nltk.stem import SnowballStemmer

from pandarallel import pandarallel
pandarallel.initialize()


def get_data_explained_percentage(df, tags_df, K):
  tags = list(tags_df['class'])[:K]
  new_df = df[df['BROWSE_NODE_ID'].isin(tags)]
  print("Percentage Explained by top ", K, " tags is :", len(new_df)/len(df)*100)
  return new_df


"""Text Cleaning"""
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')
total_processed = 0
def process_questions(row):
    global total_processed
    article = str(row['BULLET_POINTS'])
    title =  str(row['TITLE'])
    total_processed += 1
    if(total_processed%10000 == 0):
      print("processed : ", total_processed)
    if '[' in article:
        article=article.lstrip('[').rstrip(']')
    
    
    question=3*(str(title)+" ")+str(article)  
    question=re.sub(r'[^A-Za-z,.]+',' ',question) 
    words_in_questions=word_tokenize(str(question.lower()))
    #Removing stopwords, 1 letter words except 'c'
    question_cleaned=" ".join(stemmer.stem(word) for word in words_in_questions if not word in stop_words)
    return question_cleaned




if __name__ == "__main__":
    df = pd.read_csv("data/dataset/train.csv", escapechar = "\\", quoting = csv.QUOTE_NONE)
    df = df[["TITLE", "BULLET_POINTS", "BROWSE_NODE_ID"]]

    freq_dict = Counter(df['BROWSE_NODE_ID'])
    tags_df=pd.DataFrame(list(freq_dict.items()),columns=['class','Frequency'])
    tags_df.head()
    tags_df=tags_df.sort_values(ascending=False,by='Frequency') #sorted by no. of occurances

    topK = 2000
    new_df = get_data_explained_percentage(df, tags_df, topK)

    processed_data=pd.DataFrame()
    processed_data['desc']=new_df.parallel_apply(process_questions, axis=1)
    
    print("Total no. of descriptions processed from each title and bullet: ",len(processed_data['desc']))
           
    processed_data['BROWSE_NODE_ID']=new_df['BROWSE_NODE_ID']
    df = processed_data
    train, test = train_test_split(df, test_size=0.2)
    train , val = train_test_split(train, test_size=0.25)

    os.makedirs("custum-data", exist_ok=True)
    train.to_csv("custum-data/train.csv", index=False)
    test.to_csv("custum-data/test.csv", index=False)
    val.to_csv("custum-data/val.csv", index=False)
    print("done")