import spacy
import re
import pandas as pd
from datetime import datetime
import os

nlp = spacy.load("pl_core_news_lg")


def clean_formatting(text):
    """
    Clean and preprocess the input text by removing punctuation, converting to lowercase, 
    replacing multiple spaces with a single space, and removing digits.
    
    Args:
    - text (str): The input text to be cleaned.
    
    Returns:
    - str: The cleaned and preprocessed text.
    """
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace('\n', ' ').strip().lower()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\d+', '', text)  
    return text

def remove_stop_words(tokens, specified_words_to_remove):
    """
    Remove specified stop words from the input text.
    
    Args:
    - text (str): The input text from which stop words will be removed.
    - specified_words_to_remove (list): A list of stop words to be removed from the text.
    
    Returns:
    - str: The input text with specified stop words removed.
    """
    filtered_tokens = [token for token in tokens if token not in specified_words_to_remove]
    return ' '.join(filtered_tokens)

def remove_duplicated_words(text):
    """
    Remove duplicate words from the input text while maintaining the original order.
    
    Args:
    - text (str): The input text from which duplicate words will be removed.
    
    Returns:
    - str: The input text with duplicate words removed.
    """
    words = text.split()
    unique_words = list(dict.fromkeys(words))  # Maintain order and remove duplicates
    return ' '.join(unique_words)

def tokenize_text(text):
    """
    Tokenize the input text using spaCy.

    Args:
    - text (str): The input text to be tokenized.

    Returns:
    - list: A list of tokenized words.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def lemmatize_text(text):
    """
    Lemmatize the input text using a specified spaCy language model.
    
    Args:
    - text (str): The input text to be lemmatized.
    
    Returns:
    - str: The lemmatized text.
    """
    doc = nlp(text)  
    lemmatized_text = " ".join(token.lemma_ for token in doc if token.pos_ == "NOUN" )  
    return lemmatized_text


def process_text_columns(df):
    """
    Preprocess text data in specified DataFrame columns.

    Args:
    - df (DataFrame): The input DataFrame containing text columns to be processed.

    Returns:
    - DataFrame: The DataFrame with processed text columns.

    The function iterates over specified columns ('nazwa', 'sklad', 'wskazania') in the DataFrame and performs the following preprocessing steps:
    - For the 'nazwa' column:
        - Clean the text formatting.
        - Remove specified stop words.
        - Remove duplicated words.
    - For the 'sklad' column:
        - Clean the text formatting.
        - Remove specified stop words.
        - Lemmatize the text.
        - Remove duplicated words.
    - For the 'wskazania' column:
        - Clean the text formatting.
        - Remove specified stop words.
        - Lemmatize the text.
        - Remove duplicated words.

    The stop words and text cleaning operations are customized for each column to ensure appropriate preprocessing based on the nature of the text data in each column.
    """
    columns_to_process = ['nazwa', 'sklad', 'wskazania']
    stop_words_nazwa = ['nazwa', 'produktu', 'leczniczego', 'mg', 'ithib', 'charakterystyka', 'roztwór', 'aerozol', 'tabletka', 'kapsułka', 'tabletki', 'inhalacyjny', 'lek', 'wstrzykiwań', 'powlekana',\
                'powlekany', 'powlekane', 'kapsułki', 'twarde', 'twarda', 'ampułkostrzykawce','ampułkostrzykawka', 'infuzji', 'żel', 'żucia', "proszek", 'sporządzania','mgg', 'koncentrat', "mgml", "ml", "jm", \
                 "summary", "of" , "product", "characteristics", "rozpuszczalnik", "roztworu", "dojelitowe", "przedłużonym", "uwalnianiu", "krople", "nos", "nosa", "dawka", "dawkę", \
                    "zawiesina", "zawiesinę", "mikrogramy", "donosową", "maść"]
    stop_words_sklad = ['skład', 'jakościowy', 'ilościowy',  "pełny", "wykaz",  "substancji", "twarda", "twarda", "powlekana",\
                        "pomocniczych", "patrz", "punkt", "substancje","substancja", "pomocnicze"," pomocnicza", "mg", "znanym", "każdy", "każda", \
                        "chlorowodorek", "chlorowodorku", "laktoza jednowodna", "substancja", "pomocnicza" , "kapsułka", "kapsułki",\
                        "tabletka", "tabletce" "powlekana", "tabletki", "powlekane", "sodowy", "sodowego", "otoczka", "otoczką", \
                        "laktozy", "laktoza", "jednowodna", "jednowodnej", "laktoza", "jednowodna", "czerwień", "koszelinowa", "czerwieni", "koszenilowej", "fiolka", "fiolki", \
                        "woda", "wody", "postaci", "sacharoza", "patrz", "punkt","stężenia", "elektrolitów",\
                        "znanym" ,"działaniu", "zawiera", "działanie", "działaniu", "oraz", "lub", "gram", "około", "mililitr", "zawiesina","zawiesiny", \
                        "mmoll", "roztwór", "roztworu",  "wyciąg", "zawiera", "pełny", "wykaz substancji pomocniczych", "do wstrzykiwań", "suchego standaryzowanego", "co najmniej", \
                            "dla produktu",\
                        "w ilości odpowiadającej", "szybko uwalnianej", "wolno uwalnianej", "ampułko-strzykawka", "rozgryzania", "żucia" , "dawka", "dawce", "dawkach"\
                        "odmierzona", "mikrogramów", "ustnik", "zawiesiny", "sorbitol","mgml", "aspartam", "zmikronizowanej", "przeliczeniu", \
                        "alkohol cetostearylowy", "kwas sorbowy", "ampułka", "j.m.", "ml", "aktywności", "wody", "otrzymywaną", "wyniku", "tabletka dojelitowa", "elastyczna", \
                        "lecytyna sojowa", "sorbitol", "sorbitolu",\
                        "postać farmaceutyczna", "wymiary", "żelatynowej", "meql", "infuzji", "przypadku", "zasobnik", "zasobnika", "ilość", "inhalator", "inhalatora"]
    stop_words_wskazania = [ "wskazania", "wskazany" ,  "stosowania","stosowanie", "zapobieganie",  "wskazanie","wskazanym", "wskazane", "wskazana", "wskazanego", "wskazanej", "wskazanym", "wskazanych", \
                            "wskazanymi", "wskazanemu", "wskazaną", "wskazani", "leczniczy", "leczenie", "leczeniu","leczenia", " oraz ", " kiedy ", \
                             " który ", " która ", " które ", " których ",  "pacjent", "patrz", "punkty", "osób", "produkt", "substancja", "pomocniczy", "pomocnicza" , \
                            "metody", "celu", "mieszaną", "obejmującego", "stosowanie", "pacjent", "pacjentom", "pacjentowi", "pacjenta", "w celu", "jest", "są", "jako", \
                            "terapia", "dodatkowy", "dodatkowa", "dodatkowe", "standardowe", \
                            "wstrzykiwań", "suchego standaryzowanego", "co najmniej", "dla produktu",\
                        "w ilości odpowiadającej", "szybko uwalnianej", "wolno uwalnianej", "ampułko-strzykawka", "rozgryzania", "żucia" , "dawka", "dawce", "dawkach"\
                        "odmierzona", "mikrogramów", "ustnik", "zawiesiny", "sorbitol","glikol propylenowy","mgml", "benzoesan sodu", "aspartam", "zmikronizowanej", "w przeliczeniu",
                        "alkohol cetostearylowy", "kwas sorbowy", "ampułka", "j.m.", "ml", "aktywności", "wody", "otrzymywaną", "wyniku", "tabletka dojelitowa", "elastyczna", "lecytyna sojowa", \
                        "postać farmaceutyczna", "wymiary", "żelatynowej", "meql", "infuzji", "postaci", "przypadku", "zasobnik", "zasobnika", "ilość", "inhalator", "inhalatora"]
    
    for column in columns_to_process:
        if column == 'nazwa':
            df.loc[:, column] = df[column].apply(clean_formatting)
            df.loc[:, column] = df[column].apply(tokenize_text)
            df.loc[:, column] = df[column].apply(remove_stop_words, specified_words_to_remove=stop_words_nazwa)
           
            df.loc[:, column] = df[column].apply(remove_duplicated_words)


        elif column == 'sklad':
            df.loc[:, column] = df[column].apply(clean_formatting)
            df.loc[:, column] = df[column].apply(tokenize_text)
            df.loc[:, column] = df[column].apply(remove_stop_words, specified_words_to_remove=stop_words_sklad)
            df.loc[:, column] = df[column].apply(lemmatize_text)
            df.loc[:, column] = df[column].apply(remove_duplicated_words)
            
        
        elif column == 'wskazania':
            df.loc[:, column] = df[column].apply(clean_formatting)
            df.loc[:, column] = df[column].apply(tokenize_text)
            df.loc[:, column] = df[column].apply(remove_stop_words, specified_words_to_remove=stop_words_wskazania)
            df.loc[:, column] = df[column].apply(lemmatize_text)
            df.loc[:, column] = df[column].apply(remove_duplicated_words)
    
    return df

def convert_to_dict(df):
    """
    Convert DataFrame rows into a list of dictionaries representing documents.

    Args:
    - df (DataFrame): The input DataFrame containing columns 'nazwa', 'sklad', 'wskazania', and 'filename'.

    Returns:
    - list: A list of dictionaries, where each dictionary represents a document with the following keys:
        - "title": The title of the document (taken from the 'nazwa' column).
        - "text": The text content of the document, obtained by concatenating the 'sklad' and 'wskazania' columns.
        - "filename": The filename associated with the document (taken from the 'filename' column).
        - "timestamp": The timestamp representing when the document was processed (set to the current datetime).

    The function iterates over each row in the DataFrame and creates a dictionary for each row, representing a document.
    Each document dictionary contains the title, text, filename, and timestamp information extracted from the DataFrame columns.
    The list of document dictionaries is then returned as the output.
    """
    list_of_docs = []  
    for index, row in df.iterrows():
        doc = {
            "title": row["nazwa"],  
            "text": row["sklad"] + " " + row["wskazania"],  
            "filename": row["filename"],  
            "timestamp": datetime.now()  
        }
        list_of_docs.append(doc)
    return list_of_docs

def convert_to_dict_new_file(df):
    """
    Convert a pandas DataFrame into a list of dictionaries.

    Each row in the DataFrame is converted into a dictionary with the keys "filename", 
    "nazwa", "sklad", and "wskazania", corresponding to the columns in the DataFrame.

    Args:
    - df (pandas.DataFrame): The input DataFrame containing the data to be converted. 
      The DataFrame must have the columns "filename", "nazwa", "sklad", and "wskazania".

    Returns:
    - list_of_docs (list): A list of dictionaries, where each dictionary represents a row 
      from the DataFrame with keys "filename", "nazwa", "sklad", and "wskazania".
    """
    list_of_docs = []  
    for index, row in df.iterrows():
        doc = {
            "filename": row["filename"], 
            "nazwa": row["nazwa"],  
            "sklad": row["sklad"],
            "wskazania":  row["wskazania"]
        }
        list_of_docs.append(doc)
    return list_of_docs

def capitalize_first_letter(text):
    """
    Capitalize the first letter of the text.
    If the text is empty, return it as is.
    """
    if text:
        return text[0].upper() + text[1:]
    else:
        return text
