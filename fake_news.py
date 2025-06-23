
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import kagglehub
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re # regular expression library
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
path = kagglehub.dataset_download("mahdimashayekhi/fake-news-detection-dataset")
df = pd.read_csv(path+'/fake_news_dataset.csv')
print(df.isnull().sum())
print(df['source'].mode())
plot = df['source'].value_counts().plot(kind='bar')
plt.title("Source Distribution")
plt.xlabel("Source")
plt.ylabel("Count")
plt.show()
df.fillna({'source': 'Daily News'}, inplace=True)
print(df['author'].value_counts())
df.fillna({'author': 'Michael Smith'}, inplace=True)
print(df.isnull().sum())
print(df.duplicated().sum())

print(stopwords.words('english'))
print(df.head())