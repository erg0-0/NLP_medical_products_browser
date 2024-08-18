import fitz
import re
import spacy
import os
import pandas as pd

def apply_replacements(df, column):
    """
    Apply replacements to the specified column in the DataFrame.

    Args:
    - df (DataFrame): The DataFrame.
    - column (str): The name of the column to apply replacements to.
    - replacements (dict): A dictionary where keys are old values to be replaced
      and values are new values.

    Returns:
    - df (DataFrame): The DataFrame with replacements applied to the specified column.
    """
    replacements = {
        "NAZWA WŁASNA": "NAZWA",
        "LECZNICZNEGO": "LECZNICZEGO ",
        "LECZNICZEGO": "LECZNICZEGO ",
        "PODUKTU": "PRODUKTU",
        "PRODUKU": "PRODUKTU",
        "PODUKTU": "PRODUKTU",
        "1.NAZWA ": "1. NAZWA ",
        "1.NAZWA ": "1. NAZWA ",
        "1 nazwa" :"1. NAZWA ",
        "SKLAD": "SKŁAD",
        "w jamie ustnej skład" : "W JAMIE USTNEJ 2. SKŁAD ",
        "2. skład" : "2. SKŁAD ",
        "2 skład" : "2. SKŁAD ",
        "2.SKŁAD ": "2. SKŁAD ",
        "2 SKŁAD" :"2. SKŁAD ",
        "2 skład": "2. SKŁAD ",
        "4.1 Wskazania": "4.1. Wskazania",
        "4.1Wskazania": "4.1. Wskazania",
        "4.1.Wskazania": "4.1. Wskazania",
        "2.0\\4-": "",
        "\\":"",
        "/":"",
        "jest wskazany": "wskazania"        
    }
    for old_value, new_value in replacements.items():
        df[column] = df[column].str.upper().str.replace(old_value, new_value).str.lower()
    return df


def extract_columns(df, column):
    df = apply_replacements(df, column) 
    for index, row in df.iterrows():
        #nazwa
        start_index_nazwa = row[column].find("1")
        end_index_nazwa = row[column].find("2.") if row[column].find("2.") != -1 else row[column].find("2")
        nazwa = row[column][start_index_nazwa:end_index_nazwa].strip()
        df.at[index, 'nazwa'] = nazwa    
        #sklad
        start_index_sklad = row[column].find("2.") if row[column].find("2.") != -1 else row[column].find("2 skład ilościowy i jakościowy ")
        end_index_sklad = row[column].find("3.") if row[column].find("3.") != -1 else row[column].find("4.1")
        sklad = row[column][start_index_sklad:end_index_sklad].strip()
        df.at[index, 'sklad'] = sklad
        #wskazania
        start_index_wskazania = row[column].find("4.1") if row[column].find("4.1") != -1 else row[column].find("wskazania") 
        end_index_wskazania = row[column].find("4.2")
        wskazania = row[column][start_index_wskazania:end_index_wskazania].strip()
        df.at[index, 'wskazania'] = wskazania
        
    return df[['filename', 'nazwa', 'sklad', 'wskazania']]
