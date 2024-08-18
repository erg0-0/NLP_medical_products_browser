import fitz
import re
import spacy
import os
import pandas as pd

import os
import fitz

def read_documents(input_path):
    """
    Read text content from the first four pages of each PDF document in the specified folder.

    Args:
    - input_path (str): The path to the folder containing the PDF documents.

    Returns:
    - documents (list): A list of lists, where each inner list contains the filename
      and "CHCPL" - the combined text content of the first four pages of the corresponding PDF document.
    """
    documents = []
    if os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):  # Sort to ensure consistent order
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.pdf'):
                try:
                    with fitz.open(file_path) as pdf:
                        combined_text = ""
                        for page_number in range(min(4, pdf.page_count)):
                            try:
                                page = pdf.load_page(page_number)
                                footer = 80  
                                page_text = page.get_text("text", clip=(0, 0, page.rect.width, page.rect.height - footer))
                                combined_text += page_text + " "
                            except Exception as e:
                                pass 
                        combined_text = combined_text.replace("\n", "")
                        documents.append([filename, combined_text])
                except Exception as e:
                    pass 
    else:
        print("Invalid input path. Please provide a valid folder path.")
    return documents


def load_to_pd(input_path):
    """
    Load text data from PDF documents in the specified folder into a pandas DataFrame.

    Args:
    - input_folder (str): The path to the folder containing the PDF documents.

    Returns:
    - text_df (DataFrame): A pandas DataFrame containing the text data from the PDF documents.
      The DataFrame has two columns: "filename" and "CHPCL". Each row corresponds to a PDF document,
      where "filename" contains the name of the PDF file, and "CHPCL" contains the combined text content
      of the first four pages of the PDF.

    """
    text = read_documents(input_path)
    text_df = pd.DataFrame(text)
    current_columns = text_df.columns
    new_columns = [f"column{i+1}" for i in range(len(current_columns))]
    text_df.columns = new_columns
    text_df.rename(columns={"column1": "filename", "column2":"CHPCL"}, inplace=True)
    return text_df