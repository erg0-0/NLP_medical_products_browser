from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from utils_dataprep import read_documents, load_to_pd
from utils_info_extract import extract_columns, apply_replacements
from utils_data_cleaning import process_text_columns, convert_to_dict, capitalize_first_letter,convert_to_dict_new_file
from sklearn.metrics.pairwise import cosine_similarity




def process_new_files_similarity(input_file, list_of_docs, text_cleaned, model):
    """
    Process new files for similarity search.

    Parameters:
        input_file (str): The path to the new file to be processed.
        list_of_docs (list): List of dictionaries containing information about existing documents.
        text_cleaned (DataFrame): DataFrame containing cleaned text data.
        model (SentenceTransformer): SentenceTransformer model for computing embeddings.

    Returns:
        None: Prints the results of the similarity search.
    """
    new_file_df = load_to_pd(input_file)
    new_file_extracted = extract_columns(new_file_df, 'CHPCL')
    new_file_cleaned = process_text_columns(new_file_extracted)
    new_file_list_of_docs = convert_to_dict_new_file(new_file_cleaned)

    
    existing_doc_embeddings_sklad = model.encode([doc['sklad'] for doc in list_of_docs])
    existing_doc_embeddings_wskazania = model.encode([doc['wskazania'] for doc in list_of_docs])

    new_doc_embeddings_sklad = model.encode([doc['sklad'] for doc in new_file_list_of_docs])
    new_doc_embeddings_wskazania = model.encode([doc['wskazania'] for doc in new_file_list_of_docs])

    
    index_sklad = faiss.IndexFlatL2(existing_doc_embeddings_sklad[0].shape[0])
    index_wskazania = faiss.IndexFlatL2(existing_doc_embeddings_wskazania[0].shape[0])

    index_sklad.add(np.array(existing_doc_embeddings_sklad))
    index_wskazania.add(np.array(existing_doc_embeddings_wskazania))

    for new_doc_sklad, new_embedding_sklad, new_doc_wskazania, new_embedding_wskazania in zip(new_file_list_of_docs, new_doc_embeddings_sklad, new_file_list_of_docs, new_doc_embeddings_wskazania):
        _, indices_sklad = index_sklad.search(np.array([new_embedding_sklad]), 150)
        _, indices_wskazania = index_wskazania.search(np.array([new_embedding_wskazania]), 150)

        top_documents_sklad = [(list_of_docs[idx], cosine_similarity([new_embedding_sklad], [model.encode([list_of_docs[idx]['sklad']])[0]])[0][0]) for idx in indices_sklad[0]]

        top_documents_wskazania = [(list_of_docs[idx], cosine_similarity([new_embedding_wskazania], [model.encode([list_of_docs[idx]['wskazania']])[0]])[0][0]) for idx in indices_wskazania[0]]

        print(f"Results for file: {new_doc_sklad['filename']}")
        
        for i, ((doc_sklad, score_sklad), (doc_wskazania, score_wskazania)) in enumerate(zip(top_documents_sklad, top_documents_wskazania), start=1):
            combined_score = (score_sklad + score_wskazania) / 2
            if combined_score > 0.0:
                print(f"{doc_sklad['filename']},\"{capitalize_first_letter(doc_sklad['nazwa'])}\",{combined_score:.3f}")
        print()

def process_new_files_similarity_sklad_only(input_file, list_of_docs, text_cleaned, model):
    """
    Process new files for similarity search based on the 'sklad' key.

    Parameters:
        input_file (str): The path to the new file to be processed.
        list_of_docs (list): List of dictionaries containing information about existing documents.
        text_cleaned (DataFrame): DataFrame containing cleaned text data.
        model (SentenceTransformer): SentenceTransformer model for computing embeddings.

    Returns:
        None: Prints the results of the similarity search.
    """
    # Load and process the new file
    new_file_df = load_to_pd(input_file)
    new_file_extracted = extract_columns(new_file_df, 'CHPCL')
    new_file_cleaned = process_text_columns(new_file_extracted)
    new_file_list_of_docs = convert_to_dict_new_file(new_file_cleaned)

    # Compute embeddings for the existing documents and the new file
    existing_doc_embeddings_sklad = model.encode([doc['sklad'] for doc in list_of_docs])
    new_doc_embeddings_sklad = model.encode([doc['sklad'] for doc in new_file_list_of_docs])

    # Initialize the FAISS index and add the embeddings
    index_sklad = faiss.IndexFlatL2(existing_doc_embeddings_sklad[0].shape[0])
    index_sklad.add(np.array(existing_doc_embeddings_sklad))

    for new_doc_sklad, new_embedding_sklad in zip(new_file_list_of_docs, new_doc_embeddings_sklad):
        ##_, indices_sklad = index_sklad.search(np.array([new_embedding_sklad]), 5)
        D, indices_sklad = index_sklad.search(np.array([new_embedding_sklad]), len(existing_doc_embeddings_sklad))


        #top_documents_sklad = [(list_of_docs[idx], cosine_similarity([new_embedding_sklad], [model.encode([list_of_docs[idx]['sklad']])[0]])[0][0]) for idx in indices_sklad[0]]
        top_documents_sklad = [
            (list_of_docs[idx], cosine_similarity([new_embedding_sklad], [model.encode([list_of_docs[idx]['sklad']])[0]])[0][0])
            for idx in indices_sklad[0]
        ]

        print(f"Results for file: {new_doc_sklad['filename']}")
            
        for i, (doc_sklad, score_sklad) in enumerate(top_documents_sklad, start=1):
            if score_sklad > 0.0:
                print(f"{doc_sklad['filename']},\"{capitalize_first_letter(doc_sklad['nazwa'])}\",{score_sklad:.3f}")
        print()

def process_new_files_similarity_only_wskazania(input_file, list_of_docs, text_cleaned, model):
    """
    Process new files for similarity search based on the 'wskazania' key.

    Parameters:
        input_file (str): The path to the new file to be processed.
        list_of_docs (list): List of dictionaries containing information about existing documents.
        text_cleaned (DataFrame): DataFrame containing cleaned text data.
        model (SentenceTransformer): SentenceTransformer model for computing embeddings.

    Returns:
        None: Prints the results of the similarity search.
    """
    # Load and process the new file
    new_file_df = load_to_pd(input_file)
    new_file_extracted = extract_columns(new_file_df, 'CHPCL')
    new_file_cleaned = process_text_columns(new_file_extracted)
    new_file_list_of_docs = convert_to_dict_new_file(new_file_cleaned)

    # Compute embeddings for the existing documents and the new file
    existing_doc_embeddings_wskazania = model.encode([doc['wskazania'] for doc in list_of_docs])
    new_doc_embeddings_wskazania = model.encode([doc['wskazania'] for doc in new_file_list_of_docs])

    # Initialize the FAISS index and add the embeddings
    index_wskazania = faiss.IndexFlatL2(existing_doc_embeddings_wskazania[0].shape[0])
    index_wskazania.add(np.array(existing_doc_embeddings_wskazania))
    
    for new_doc_wskazania, new_embedding_wskazania in zip(new_file_list_of_docs, new_doc_embeddings_wskazania):
        # Search the entire index
        D, indices_wskazania = index_wskazania.search(np.array([new_embedding_wskazania]), len(existing_doc_embeddings_wskazania))
        #_, indices_wskazania = index_wskazania.search(np.array([new_embedding_wskazania]), 5)

        # Compute cosine similarity for the retrieved documents
        top_documents_wskazania = [
            (list_of_docs[idx], cosine_similarity([new_embedding_wskazania], [model.encode([list_of_docs[idx]['wskazania']])[0]])[0][0])
            for idx in indices_wskazania[0]
        ]
        #top_documents_wskazania = [(list_of_docs[idx], cosine_similarity([new_embedding_wskazania], [model.encode([list_of_docs[idx]['wskazania']])[0]])[0][0]) for idx in indices_wskazania[0]]

        print(f"Results for file: {new_doc_wskazania['filename']}")
        
        for i, (doc_wskazania, score_wskazania) in enumerate(top_documents_wskazania, start=1):
            if score_wskazania > 0.0:
                print(f"{doc_wskazania['filename']},\"{capitalize_first_letter(doc_wskazania['nazwa'])}\",{score_wskazania:.3f}")
        print()


