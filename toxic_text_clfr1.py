#!/usr/bin/env python
# coding: utf-8

# # Zummit Africa Project

# # Problem Statement #

# ***Given a group of sentences or paragraphs, used as a comment by a user in an online platform<br>classify it to belong to one or more of the following categories — toxic, severe-toxic, obscene, threat,<br>insult or identity-hate with either approximate probabilities or discrete values (0/1).***

# In[1]:


#import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# load the dataset
df = pd.read_csv(r'C:\Users\HP\Desktop\Data Science\toxic_comments\train.csv')


# In[3]:


# check the dataframe
df.head()


# In[4]:


# check data general infomation
df.info()


# In[5]:


# check for null or missing values
df.isnull().sum()


# In[6]:


# print the statistical analysis
df.describe().T


# ### Exploratory Data Analysis

# In[7]:


# perform a value count for each column feature
# locate or define a numeric function for plotting
sentence_col=df.iloc[:,2:].sum()
sentence_col


# In[8]:


# plot the graph of value count
sns.set_style("darkgrid")
ls=sentence_col.sort_values(ascending=False)
plt.figure(figsize=(15,8))
temp =sns.barplot(ls.index, ls.values, alpha=0.8) 
plt.title('Comments')
plt.ylabel('COUNT', fontsize=14)
plt.xlabel('Toxic Comment Types', fontsize=15)
temp.set_xticklabels(rotation=90,labels=ls.index,fontsize=10)
plt.show()


# In[9]:


# count plot of each feature
sns.countplot(df['toxic'])


# In[10]:


# count plot of each feature
sns.countplot(df['obscene'])


# In[11]:


# count plot of each feature
sns.countplot(df['insult'])


# In[12]:


# count plot of each feature
sns.countplot(df['severe_toxic'])


# In[13]:


# count plot of each feature
sns.countplot(df['identity_hate'])


# In[14]:


# count plot of each feature
sns.countplot(df['threat'])


# ***Obviously we have maximum values of zero and minimum values of 1, based on the analysis above.*** 

# In[15]:


# visualize comment percentage
plt.pie(sentence_col.values,labels=sentence_col.index,autopct='%.2f',counterclock=False)
plt.axis('equal')
plt.show()


# In[16]:


# plot comment_text length distribution
plt.figure(figsize = (24,12))
sns.distplot(df["comment_text"].apply(lambda x : len(x)))
plt.title("Length Of Comments")
plt.show()


# ***The distribution curve is skewed to the right, which indicte the uneven spread of data***

# ### Data Pre-processing

# In[17]:


# clean the text by removing punctuation and stopword
# convert abbreviations to full text
# remove special character
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
df['comment_text'] = df['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


# In[18]:


df['comment_text']


# In[19]:


# plot a wordcloud for comment text
words = ' '.join([k for k in df['comment_text']])
W_C = WordCloud(width=1000, height=250, random_state=31, max_font_size=200,background_color='pink').generate(words)
plt.figure(figsize=(20,8))
plt.imshow(W_C, interpolation="bilinear")
plt.axis('off')
plt.show()


# ### Feature Selection and Model Building

# In[20]:


# used tf_idf vectorizing technique to transform text to numerals
# here i am considering max-features as 40000 and passing bi-gram model
tf_idf = TfidfVectorizer(analyzer='word', max_features=40000, ngram_range=(1,2), stop_words='english')
# declare an independent variable
X = tf_idf.fit_transform(df['comment_text'])
X


# In[21]:


# get column names
name_of_features=tf_idf.get_feature_names()
# getting list of features
name_of_features[0:10]


# In[22]:


# declaring the dependent variable
y=df.drop(['id', 'comment_text'], axis=1)
y.drop(['toxic'], axis=1, inplace=True)
y


# In[23]:


# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)


# ### Support Vector Classifier in OneVsRest Classifier

# In[24]:


# preparing object for linear SVC
svc = LinearSVC()


# In[25]:


# fixing the model in onevsrest classifier
# use one-vs-rest classifier because of multiclass classification
clf = OneVsRestClassifier(svc)


# In[26]:


# fit the model to the dataset
clf.fit(X_train,y_train)


# In[27]:


# make prediction with the model
svc_pred= clf.predict(X_test)
svc_pred


# ### Model Validation and Evaluation

# In[28]:


# print confusion matrix
cm_svc=confusion_matrix(y_test.values.argmax(axis=1),svc_pred.argmax(axis=1))
cm_svc


# In[29]:


# print classification report
print(classification_report(svc_pred.argmax(axis=1),y_test.values.argmax(axis=1)))


# In[30]:


# check model accuracy
accuracy_svc_pred=accuracy_score(y_test.values.argmax(axis=1),svc_pred.argmax(axis=1))
accuracy_svc_pred


# In[31]:


# check the f1-score of the model
print("f1_score:",f1_score(y_test,svc_pred, average="micro"))


# ### Model Testing

# In[32]:


# Declare a comment list for testing the model
comment = ['i hate you', "you are a good boy", "you are an asshole"]


# In[33]:


# tranform the comments to vectors
xt = tf_idf.transform(comment)
xt


# In[34]:


# make the prediction
category=clf.predict(xt)
category


# In[35]:


# create a df for the prediction class
col=[ 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
cat = pd.DataFrame(category,columns=col)
cat


# In[36]:


# print prediction
for i in range(len(cat)):
    if cat.columns[(cat == 1).iloc[i]].notna().all():
        print(cat.columns[(cat == 1).iloc[i]].values)


# ### Create a pipeline for the model

# In[37]:


#import the library
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


# In[38]:


pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
                                                      max_features=1000,
                                                      stop_words= ENGLISH_STOP_WORDS)),
                            ('model', OneVsRestClassifier(LinearSVC()))])


# In[39]:


# fit the model into the pipeline
pipeline.fit(df.comment_text, y)


# In[40]:


# make prediction with the pipeline
pipeline.predict(df.comment_text)


# In[41]:


# create a word list for prediction test
word=["eat shit and die"]


# In[42]:


# check prediction
pipeline.predict(word)


# ### Save or Dump the Model Using Joblib

# In[43]:


# import the library
from joblib import dump


# In[44]:


# save now
dump(pipeline, filename="toxic_comment_classification.joblib")


# In[ ]:




