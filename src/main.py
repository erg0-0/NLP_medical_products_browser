import os
import fitz
import spacy
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from utils_dataprep import read_documents, load_to_pd
from utils_info_extract import extract_columns, apply_replacements
from utils_data_cleaning import process_text_columns, convert_to_dict, capitalize_first_letter,convert_to_dict_new_file
from utils_search_engine import prepare_tfidf_representation, search_product_by_indication
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from utils_search_similar import process_new_files_similarity_sklad_only, process_new_files_similarity_only_wskazania, process_new_files_similarity
import argparse

pd.set_option('display.max_colwidth', None)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir,'data')

if __name__ == "__main__":

    text_df = load_to_pd(input_path)
    text_extracted = extract_columns(text_df, 'CHPCL') 
    text_cleaned= process_text_columns(text_extracted)
    
    parser = argparse.ArgumentParser(description = "Search for medicinal products or run similarity model.")
    parser.add_argument("-q", "--query", type=str, help = "Search for medicinal products by indication")
    parser.add_argument("-f", "--file", type=str, help = "Provide a INPUT PATH TO A FOLDER in order to list similar products`")
    args = parser.parse_args()

if args.query:

    list_of_docs = convert_to_dict(text_cleaned)
    tfidf_vectorizer, tfidf_matrix = prepare_tfidf_representation(list_of_docs)
    wskazania_pattern = args.query.lower() 
    
    matching_products = search_product_by_indication(wskazania_pattern, list_of_docs, tfidf_vectorizer, tfidf_matrix)
    threashold = 0.0
    filtered_products = [product for product in matching_products if product['score'] > threashold]
    for product in filtered_products:
        product_name_capitalized = capitalize_first_letter(product["product_name"])
        print(f'{product["filename"]},"{product_name_capitalized}",{product["score"]:.3f}')
        
elif args.file:

    list_of_docs = convert_to_dict_new_file(text_cleaned)
    
    model = SentenceTransformer('sdadas/st-polish-paraphrase-from-distilroberta')
    process_new_files_similarity(args.file, list_of_docs, text_cleaned, model)
    #process_new_files_similarity_sklad_only(args.file, list_of_docs, text_cleaned, model)
    #process_new_files_similarity_only_wskazania(args.file, list_of_docs, text_cleaned, model)
else:
    print("Error")