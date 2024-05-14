import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np

# Step 1: Read the CSV file into a DataFrame and sort by article_id
df = pd.read_csv('bbc_articles.csv')
df = df.sort_values(by='article_id')

# Step 2: Tokenize the text data and remove stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return filtered_tokens

df['tokenized_text'] = df['text'].apply(remove_stopwords)

# Step 3: Train Word2Vec model on the tokenized text
word2vec_model = Word2Vec(df['tokenized_text'], vector_size=100, window=5, min_count=1, workers=4)

# Step 4: Create word embeddings for each article
def get_embedding(tokens):
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv.key_to_index:
            embeddings.append(word2vec_model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

df['embeddings'] = df['tokenized_text'].apply(get_embedding)

# Step 5: Write the new CSV file
df.to_csv('bbc_articles_embeddings.csv', index=False)
