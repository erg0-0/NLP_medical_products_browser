# NLP Project

- Project created in April,29th 2024
- Poznan University of Medical Science, faculty Data Science in Medicine. 
- Project technical mentor: Kornel Lewandowski. 
- Project content provided thanks to dr hab. Krzysztof Kus. 
- Group: Liliana Gmerek, Dominika Szotta, Dariusz Iwanow. 
- The project allowed for the completion of the course.

## General Project Area

Project covers NLP techniques in area of **pharmacoeconomy** and supports analysis of text corpus of **Summary of product characteristics** for healthcare profesionals (SmPC, polish: Charakterystyka produktu leczniczego, ChPCL).

## Specific project goals

Data contains PDF files -  the summaries of medical product characteristics - document required for each medical products in pharmaceutics. The goal is to build a search browser that allows searching a specific products. 2 types of tasks need to be fulfilled:
1. searching the SmPCs that is registered for specific medical indication (polish: "Wskazanie").
2. searching the SmPCs similar to the product input to the browser (similarity bases on the medical indications e.g. tooth pain or blood pressure).

## Files content

1. Text corpus is entirely in Polish languague
2. The PDF files contain actual product characteristics sourced from databases that were presented during the Text Databases course. Correct solutions to programming tasks from the Text and Natural Language Processing (NLP) course include code that will enable the completion of a significant portion of the project.


## Requirements

1. A mechanism for loading PDF files and transforming them into plain text form.
2. An algorithm for extracting relevant information (files typically contain around 20 pages of typescript; only a small fragment of the knowledge contained in them is necessary to complete the task).
3. Text cleaning methods.
4. Building a search structure (index, TF-IDF matrix, or vector matrix) considering all necessary intermediate steps, or feeding an existing search structure (e.g., using Elasticsearch).
5. A mechanism for analyzing text queries (scenario 1, searching by query).
6. A mechanism for analyzing queries in the form of file names (scenario 2, searching by file name).
7. Project organization tracked over gitlab.com


## Course completion

1. The project structure should include a src directory containing the code written in Python, as well as a requirements.txt or environment.yml file that defines the list of all necessary dependencies required to run the code (please specify the version of Python for which the environment was prepared). The code in the src directory should implement a logical division into the components mentioned earlier and be accompanied by necessary annotations and/or comments (https://peps.python.org/pep-0008/#documentation-strings). While it is encouraged to organize the code into packages, it is not mandatory.

2. Code meeting the above conditions will enable the transition to the qualitative verification of the solution, which will consist of executing a series of queries: at least one query for a product registered for indications (q, query) and at least one query to find a file describing a similar product (f, file).
3. Quality verification will be based on the p@k metric (the number of correctly identified documents/products in the top k rows of the response). The passing threshold is achieving a value of 60% or higher for each query prepared by the evaluators.
4. The final grade will be determined based on the evaluation of the team's work (code), quality assessment (p@k for the queries), and the presentation of the solution prepared by the teams during the sessions summarizing the project.



## Environment preparation

- Install the packages as per the requirements.txt

- Additionally for spacy package install the model pl_core_news_lg (python -m spacy download pl_core_news_lg).


## How to run code?

1. Download the catalog with the files from the university repositorium and paste it to the catalogue src/data in the project.
2. Enter the src/ main.py file, select interpreter and run the code.


## How the code works?

1. Run the main.py . The files are read and extracted, it may take some time.
2. Enter the terminal and go to the project / src where main.py is located
3. Enter one of two arguments:
    3.1. Search for medicinal products by indication using argument -q or --query:
    Example: **python main.py -q "astma"**. Search was tested to run for about <2 minutes. Sample output includes the files with the similar SmPC: Charakterystyka-14200-2021-02-13-9574_B-2022-07-20.pdf,"Ribuspir mikrogramówdawkę odmierzoną",0.404 
    3.2. provide an input path to a folder where searched SmPC is placed in order to list similar products. Use argument -f or -file. Example: **python main.py -f "/Users/lili/Projects_studia/_PORTFOLIO/REMOTE/nlp-group2/docs/search_data"** . Code was tested to run for about 2 minutes. Sample output: Results for file: Charakterystyka-173-2023-06-15-13773_N-2023-06-29.pdf Charakterystyka-173-2023-06-15-13773_N-2023-06-29.pdf,"Nicergolin",1.000

## Attention points:

1. Less than 5 files could not be read due to wrong data format (jpg) or PDF corruption.
2. Files are entirely in Polish. At the moment the availability of language models for Polish including the medicinal language corpus is limited to handful of models.
3. Algorithm uses TFIDF matrix for searching. Elastic search has been tested by Liliana with good results, but not chosen due to technical problems on local machines for 2 group members. 
4. The high quality of the -q search was achieved by introducing: 
    a) lematization, 
    b) selecting nouns only for search purpose, 
    c) applying the model DISTILROBERTA with FAISS indexation (Facebook AI Similarity Search) for optimizing. 
    d) Used model was = SentenceTransformer(sdadas/st-polish-paraphrase-from-distilroberta')
5. Major challanges: 
    a) although the footnotes has been removed, some headers could not be read correctly
    b) not precise results have been specified by excluding a excipients (sorbitol, lactose etc.) from the search. Search focused on active substances proved to be of higher quality.

6. Model ignores the "risk factors". For example during the search model does treat medications with indication for "diabetes" and with risk factor "diabetes" in the same way, treating them as equaly similar. In the next iterations, it would be necessary to adress this problem so that the products that cause risk for diabetic are not confused with products that are indicated for diebetics.
