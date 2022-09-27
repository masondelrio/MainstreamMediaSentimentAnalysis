#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import os


# In[3]:


os.chdir("C://Users//Cindy//Dropbox//Spring Quarter 2022//STA 160//ScrapedData")


# In[114]:


FOX = pd.read_csv('FOX.csv')
ABC = pd.read_csv('ABC.csv')
CBS = pd.read_csv('CBS.csv')
CNN = pd.read_csv('CNN.csv')
MSNBC = pd.read_csv('MSNBC.csv')


# In[58]:


import nltk
nltk.download('punkt')


# The below code obtains overall sentiment analysis for each headline, followed by sentiment analysis on each word of each headline.

# In[115]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, RegexpTokenizer
sia = SIA()
FOXresults = []
ABCresults = []
CBSresults = []
CNNresults = []
MSNBCresults = []
posFOX = []
negFOX = []
neuFOX = []
posABC = []
negABC = []
neuABC = []
posCBS = []
negCBS = []
neuCBS = []
posCNN = []
negCNN = []
neuCNN = []
posMSNBC = []
negMSNBC = []
neuMSNBC = []
              


for lineFOX in FOX.Title:
    tokenized_sentence = nltk.word_tokenize(lineFOX)
    pol_score = sia.polarity_scores(lineFOX)
    pol_score['headline'] = lineFOX
    FOXresults.append(pol_score)
    for i in tokenized_sentence:
        if (sia.polarity_scores(i)['compound']) >= 0.2:
            posFOX.append(i)
        elif (sia.polarity_scores(i)['compound']) <= -0.2:
            negFOX.append(i)
        else:
            neuFOX.append(i)   

    
for lineABC in ABC.Title:
    tokenized_sentence = nltk.word_tokenize(lineABC)
    pol_score = sia.polarity_scores(lineABC)
    pol_score['headline'] = lineABC
    ABCresults.append(pol_score)
    for i in tokenized_sentence:
        if (sia.polarity_scores(i)['compound']) >= 0.2:
            posABC.append(i)
        elif (sia.polarity_scores(i)['compound']) <= -0.2:
            negABC.append(i)
        else:
            neuABC.append(i)  

for lineCBS in CBS.Title:
    tokenized_sentence = nltk.word_tokenize(lineCBS)
    pol_score = sia.polarity_scores(lineCBS)
    pol_score['headline'] = lineCBS
    CBSresults.append(pol_score)
    for i in tokenized_sentence:
        if (sia.polarity_scores(i)['compound']) >= 0.2:
            posCBS.append(i)
        elif (sia.polarity_scores(i)['compound']) <= -0.2:
            negCBS.append(i)
        else:
            neuCBS.append(i)   
for lineCNN in CNN.Title:
    tokenized_sentence = nltk.word_tokenize(lineCNN)
    pol_score = sia.polarity_scores(lineCNN)
    pol_score['headline'] = lineCNN
    CNNresults.append(pol_score)
    for i in tokenized_sentence:
        if (sia.polarity_scores(i)['compound']) >= 0.2:
            posCNN.append(i)
        elif (sia.polarity_scores(i)['compound']) <= -0.2:
            negCNN.append(i)
        else:
            neuCNN.append(i) 
for lineMSNBC in MSNBC.Title:
    tokenized_sentence = nltk.word_tokenize(lineMSNBC)
    pol_score = sia.polarity_scores(lineMSNBC)
    pol_score['headline'] = lineMSNBC
    MSNBCresults.append(pol_score)
    for i in tokenized_sentence:
        if (sia.polarity_scores(i)['compound']) >= 0.2:
            posMSNBC.append(i)
        elif (sia.polarity_scores(i)['compound']) <= -0.2:
            negMSNBC.append(i)
        else:
            neuMSNBC.append(i)


# Convert from dictionary to dataframe

# In[116]:


sentFOX = pd.DataFrame.from_dict(FOXresults)
sentABC = pd.DataFrame.from_dict(ABCresults)
sentCBS = pd.DataFrame.from_dict(CBSresults)
sentCNN = pd.DataFrame.from_dict(CNNresults)
sentMSNBC = pd.DataFrame.from_dict(MSNBCresults)


# Label each headline based on their compound values, with above 0.2 as positive, below -0.2 as negative, and in between as neutral.

# In[121]:


sentFOX['label'] = 0
sentFOX.loc[sentFOX['compound'] > 0.2, 'label'] = 1
sentFOX.loc[sentFOX['compound'] < -0.2, 'label'] = -1
sentFOX.head()


# In[122]:


sentABC['label'] = 0
sentABC.loc[sentABC['compound'] > 0.2, 'label'] = 1
sentABC.loc[sentABC['compound'] < -0.2, 'label'] = -1
sentABC.head()


# In[123]:


sentCBS['label'] = 0
sentCBS.loc[sentCBS['compound'] > 0.2, 'label'] = 1
sentCBS.loc[sentCBS['compound'] < -0.2, 'label'] = -1
sentCBS.head()


# In[124]:


sentCNN['label'] = 0
sentCNN.loc[sentCNN['compound'] > 0.2, 'label'] = 1
sentCNN.loc[sentCNN['compound'] < -0.2, 'label'] = -1
sentCNN.head()


# In[125]:


sentMSNBC['label'] = 0
sentMSNBC.loc[sentMSNBC['compound'] > 0.2, 'label'] = 1
sentMSNBC.loc[sentMSNBC['compound'] < -0.2, 'label'] = -1
sentMSNBC.head()


# Obtain frequencies of each neutral, negative and positive headline corresponding to sia model. 

# In[126]:


sentFOX['label'].value_counts()


# In[127]:


sentABC['label'].value_counts()


# In[128]:


sentCBS['label'].value_counts()


# In[129]:


sentCNN['label'].value_counts()


# In[130]:


sentMSNBC['label'].value_counts()


# Combine each mean of each sentiment value, along with their label and compound metrics. 

# In[131]:


totalSent = pd.concat([sentFOX.mean(), sentCNN.mean(), sentABC.mean(),sentCBS.mean(),sentMSNBC.mean()], axis=1)
totalSent = totalSent.transpose()
totalSent = totalSent.reset_index()
totalSent['News Outlet'] = ['FOX', 'CNN', 'ABC', 'CBS', 'MSNBC']


# In[132]:


totalSent


# Plot the sentiment value averages

# In[133]:


import plotly.express as px
fig = px.histogram(totalSent,
                   x='News Outlet',
                   title='Sentiment Score by News Outlet | Youtube Headlines',
                   y= ['neg', 'neu','pos', 'compound', 'label'],
                   barmode='group',
                  color_discrete_sequence=["red", "blue", "green"])
fig.update_xaxes( title='News Outlets').update_yaxes(title='Sentiment score')
fig.show()


# Overall wordcloud for each news outlet, disregarding sentiment

# # Fox Wordcloud 

# In[141]:


from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

text = " ".join(i for i in sentFOX.headline)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[142]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # ABC Wordcloud

# In[143]:


text = " ".join(i for i in sentABC.headline)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[144]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # CBS Wordcloud

# In[146]:


text = " ".join(i for i in sentCBS.headline)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[147]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[10:20]))
plt.show()


# In[148]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # CNN Wordcloud

# In[149]:


text = " ".join(i for i in sentCNN.headline)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[150]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # MSNBC Wordcloud

# In[151]:


text = " ".join(i for i in sentMSNBC.headline)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[152]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Counts the frequency of different sentiment values for each word in different news outlet headlines. (For Sentiment wordclouds below)

# In[70]:


def CountFreq(li):
    freq = {}
    for item in li:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq


# # FOX Sentiment Word Clouds

# Positive Sentiment Words

# In[80]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(posFOX))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[81]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Negative Sentiment Words

# In[82]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(negFOX))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[83]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# In[106]:


FOX['Title']


# In[110]:


crisisList = []
for i in FOX["Title"]:
    if "crisis" in i:
        crisisList.append(i)
    else:
        continue
        


# In[113]:


crisisList


# # CNN Sentiment Word Clouds

# Positive Sentiment Words

# In[76]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(posCNN))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[77]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Negative Sentiment Words

# In[78]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(negCNN))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[79]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # ABC Sentiment Word Clouds

# Positive Sentiment Words

# In[84]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(posABC))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[85]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Negative Sentiment Words

# In[86]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(negABC))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[87]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # CBS Sentiment Word Clouds

# Positive Sentiment Words

# In[88]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(posCBS))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[92]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(22,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Negative Sentiment Words

# In[93]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(negCBS))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[94]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# # MNSBC Sentiment Word Clouds

# Positive Sentiment Words

# In[137]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(posMSNBC))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[138]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(22,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()


# Negative Sentiment Words

# In[139]:


stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(frequencies=CountFreq(negCBS))
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[140]:


import matplotlib.pyplot as plt
names= []
values = []

f, ax = plt.subplots(figsize=(18,5))

#list(dictionaryName.items())[:N]
plt.bar(*zip(*list(wordcloud.words_.items())[:10]))
plt.show()

