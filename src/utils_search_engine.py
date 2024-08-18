from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils_data_cleaning import lemmatize_text


def prepare_tfidf_representation(documents):
    """
    Prepare TF-IDF representation for a list of documents.

    Args:
    - documents (list of dict): A list of dictionaries, where each dictionary represents a document.
                                 Each document dictionary should contain a 'text' key with the text content.

    Returns:
    - tuple: A tuple containing TF-IDF vectorizer and TF-IDF matrix.
             The vectorizer is fitted on the corpus and used to transform documents into TF-IDF representation.
             The TF-IDF matrix represents the TF-IDF values of each term in the corpus.

    This function prepares TF-IDF representation for a list of documents. It first extracts the text content
    from each document in the list and forms a corpus. Then, it initializes a TF-IDF vectorizer and fits it
    on the corpus to learn the vocabulary and IDF weights. Finally, it transforms the corpus into a TF-IDF matrix
    using the fitted vectorizer.

    Example:
    documents = [
        {'text': 'This is the first document'},
        {'text': 'This document is the second document'},
        {'text': 'And this is the third one'},
        {'text': 'Is this the first document?'}
    ]
    vectorizer, tfidf_matrix = prepare_tfidf_representation(documents)
    """
    corpus = [doc['text'] for doc in documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def search_product_by_indication(wskazania_pattern, products, vectorizer, tfidf_matrix):
    """
    Search for medicinal products by indication using TF-IDF similarity.

    Args:
    - wskazania_pattern (str): The indication pattern to search for.
    - products (list of dict): A list of dictionaries, where each dictionary represents a product.
                                Each product dictionary should contain 'title' and 'text' keys representing
                                the product name and description respectively.
    - vectorizer (TfidfVectorizer): The TF-IDF vectorizer fitted on the corpus of products.
    - tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing the TF-IDF values of each term
                                               in the corpus of products.

    Returns:
    - list of dict: A list of dictionaries representing the matching products sorted by similarity score.
                     Each dictionary contains the product name ('product_name'), filename ('filename'),
                     and similarity score ('score').

    This function searches for medicinal products by indication using TF-IDF similarity.
    It first lemmatizes the indication pattern, then transforms it into TF-IDF representation using
    the provided vectorizer. Next, it computes cosine similarity scores between the indication TF-IDF vector
    and the TF-IDF matrix of products. The scores are rounded and sorted in descending order.
    Finally, it constructs a list of dictionaries representing the matching products with their filenames
    and similarity scores.

    Example:
    wskazania_pattern = "hypertension"
    matching_products = search_product_by_indication(wskazania_pattern, products, vectorizer, tfidf_matrix)
    for product in matching_products:
        print(f"{product['filename']}, {product['product_name']}, {product['score']:.3f}")
    """
    
    if not products:
        return []
    lemmatized_wskazania = lemmatize_text(wskazania_pattern)
    query_tfidf = vectorizer.transform([lemmatized_wskazania])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    scores = list(enumerate(cosine_similarities[0]))
    rounded_scores = [(index, round(score, 3)) for index, score in scores]
    sorted_rounded_scores = sorted(rounded_scores, key=lambda x: x[1], reverse=True)
    matching_products = [{'product_name': products[idx]['title'], 'filename': products[idx]['filename'], 'score': score} for idx, score in sorted_rounded_scores]
    return matching_products

