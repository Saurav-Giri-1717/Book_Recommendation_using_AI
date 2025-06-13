
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (sample CSV with 'title', 'author', 'genre', 'description')
df = pd.read_csv('books.csv')

# Fill missing values and combine relevant features
df['description'] = df['description'].fillna('')
df['content'] = df['title'] + ' ' + df['author'] + ' ' + df['genre'] + ' ' + df['description']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Book Index Mapping
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Recommendation Function
def recommend(title, cosine_sim=cosine_sim):
    title = title.lower()
    if title not in indices:
        return "Book not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices]

# Example Usage
print("Recommended books:")
print(recommend("The Hunger Games"))
