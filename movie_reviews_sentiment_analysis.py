## Importing modules ##
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import joblib
import nltk  # Import nltk library

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# 1 | Data Preprocessing
dataset = pd.read_csv('Dataset/IMDB.csv')
print(f"Dataset shape : {dataset.shape}\n")
print(f"Dataset head : \n{dataset.head()}\n")

print(f"Dataset output counts:\n{dataset.sentiment.value_counts()}\n")

dataset.sentiment.replace('positive', 1, inplace=True)
dataset.sentiment.replace('negative', 0, inplace=True)
print(f"Dataset head after encoding :\n{dataset.head(10)}\n")

## 2 | Data cleaning ##
"""Clean dataset reviews as following:
1. Remove HTML tags
2. Remove special characters
3. Convert everything to lowercase
4. Remove stopwords
5. Stemming
"""

# 2.1 Remove HTML tags
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

dataset.review = dataset.review.apply(clean)
print(f"Review sample after removing HTML tags : \n{dataset.review[0]}\n")

# 2.2 Remove special characters
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

dataset.review = dataset.review.apply(is_special)
print(f"Review sample after removing special characters : \n{dataset.review[0]}\n")

# 2.3 Convert everything to lowercase
def to_lower(text):
    return text.lower()

dataset.review = dataset.review.apply(to_lower)
print(f"Review sample after converting everything to lowercase : \n{dataset.review[0]}\n")

# 2.4 Remove stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

dataset.review = dataset.review.apply(rem_stopwords)
print(f"Review sample after removing stopwords : \n{dataset.review[0]}\n")

# 2.5 Stem the words
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

dataset.review = dataset.review.apply(stem_txt)
print(f"Review sample after stemming the words : \n{dataset.review[0]}\n")

## 3 | Model Creation ##
# Fit data into model
X = np.array(dataset.iloc[:, 0].values)
y = np.array(dataset.sentiment.values)
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(dataset.review).toarray()
print(f"=== Bag of words ===\n")
print(f"BOW X shape : {X.shape}")
print(f"BOW y shape : {y.shape}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
print(f"Train shapes : X = {X_train.shape}, y = {y_train.shape}")
print(f"Test shapes  : X = {X_test.shape},  y = {y_test.shape}\n")

# 4 | Extract Negative Reviews
negative_reviews = dataset[dataset['sentiment'] == 0]
print(negative_reviews['review'].head())
