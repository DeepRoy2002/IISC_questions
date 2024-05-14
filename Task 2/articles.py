import os
import csv
import zipfile
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Unzip the data.zip file
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Step 2: Create a CSV file and write headers
csv_filename = 'bbc_articles.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['article_id', 'text', 'category'])

# Step 3: Read text files and write to CSV
articles_folder = 'BBC_articles'
for filename in os.listdir(articles_folder):
    if filename.endswith('.txt'):
        article_id, category = filename.split('_')
        with open(os.path.join(articles_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([article_id, text, category])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('bbc_articles.csv')


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing, removing punctuation and stopwords, and stemming
    tokens = [ps.stem(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Step 3: Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])

# Step 4: Combine numerical features with category labels
features_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names())
features_df['category'] = df['category']

# Step 5: Write the new CSV file
features_df.to_csv('bbc_articles_numerical_features.csv', index=False)

