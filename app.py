from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

from nltk.corpus import stopwords
import re

nltk.download('stopwords')

app = Flask(__name__)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

print("Preprocessing documents ...", end=" ", flush=True)
preprocessed_docs = [preprocess_text(doc) for doc in documents]
print("Done")

nfeats = 5000
print("TF-IDF Vectorizing with %d features ... " % nfeats, end=" ", flush=True)
vectorizer = TfidfVectorizer(max_features=nfeats)
X = vectorizer.fit_transform(preprocessed_docs).toarray()
print("Done")



n_docs, n_terms = X.shape
print("Starting SVD on (%d, %d) matrix ..." % (n_docs, n_terms), end=" ", flush=True)
if n_docs > n_terms:
    XtX = np.dot(X.T, X)
    U_, S, Vt = np.linalg.svd(XtX)
    U = np.dot(X, Vt.T) / S
else:
    XXt = np.dot(X, X.T)
    U, S, Vt_ = np.linalg.svd(XXt)
    Vt = np.dot(U.T, X) / S[:, np.newaxis]

print("Done")
k = 100
U_reduced = U[:, :k]
S_reduced = np.diag(S[:k])
Vt_reduced = Vt[:k, :]

print("Reducing dimensionality from %d dimensions to %d dimensions." % (n_terms, k))
X_reduced = np.dot(U_reduced, S_reduced)


def search_engine(query):
    """
    Function to search for the top 5 similar documents given a query.
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query]).toarray()
    query_reduced = np.dot(query_vec, np.dot(Vt_reduced.T, np.linalg.inv(S_reduced)))

    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    top_indices = np.argsort(similarities)[::-1][:5].tolist()
    top_docs = [documents[idx] for idx in top_indices]
    top_similarities = [similarities[idx].tolist() for idx in top_indices]

    return top_docs, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
