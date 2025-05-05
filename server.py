import logging
import os
import sys
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info("Server Starting")

app = FastAPI(
    title = "Assessment Recommendation Engine",
    version = "1.0",
    description = "SHL Assessment Recommendation Engine using SHL's product catalog"
)

def add_api_keys():
    load_dotenv('.env')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or st.secrets["api_keys"]["GOOGLE_API_KEY"]

    if GOOGLE_API_KEY == 'None' or GOOGLE_API_KEY == '':
        print('You must specify Google API Key in .env file. Use https://aistudio.google.com/app/apikey to generate key')
        sys.exit(0)


def load_data(file_path):
    loader = CSVLoader(file_path = file_path, encoding='utf-8')
    documents = loader.load()
    logger.debug(f"Number of Documents: {len(documents)}")
    return documents


def generate_embeddings():
    if debug_mode:
        embeddings = FakeEmbeddings(size=4096)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings


def create_vector_store(documents, embeddings):
    vector_store = FAISS.from_documents(documents = documents, embedding = embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store


def calculate_recall_3(actual_docs, predicted_docs):
    num_relevant_in_top_3 = len([assessment for assessment in predicted_docs if assessment in actual_docs])
    num_total_relevant = len(actual_docs)
    recall_3 = num_relevant_in_top_3 / num_total_relevant
    return recall_3


def precision_k(actual_docs, predicted_docs, k):
    top_k_preds = predicted_docs[:k]
    num_relevant_in_top_3 = len([assessment for assessment in top_k_preds if assessment in actual_docs])
    precision = num_relevant_in_top_3 / k
    return precision


def calculate_precision_3(actual_docs, predicted_docs):
    min_k_rel = min(3, len(actual_docs))
    sum_p = 0
    for i, doc in enumerate(predicted_docs):
        if doc in actual_docs:
            sum_p += precision_k(actual_docs, predicted_docs, i + 1)

    return sum_p / min_k_rel


test_type_map = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behavior',
    'S': 'Simulations'
}

if __name__ == "__main__":
    add_api_keys()

    debug_mode = False
    file_path = 'data.csv'
    vector_store_path = 'vectordb'

    if os.path.exists('vectordb' + '/index.pkl'):
        logger.info("Loading Vector Store from Local Storage")
        vector_store = FAISS.load_local('vectordb', embeddings = generate_embeddings(),
                                        allow_dangerous_deserialization=True)
    else:
        documents = load_data(file_path)
        embeddings = generate_embeddings()
        vector_store = create_vector_store(documents, embeddings)

    @app.get("/health")
    def check_health():
        try:
            response = requests.get("http://localhost:8000/docs")
            if response.status_code == 200:
                return JSONResponse(content = {"status": "healthy"}, status_code = status.HTTP_200_OK)
            else:
                return JSONResponse(content={"status": "unhealthy"}, status_code = response.status_code)
        except Exception as e:
            logger.error(f"Exception : {e}")
            return JSONResponse(content={"status": "unhealthy"}, status_code = status.HTTP_500_INTERNAL_SERVER_ERROR)

    @app.post("/recommend")
    async def recommend_movies(request: dict):
        query = request.get("query")
        try:
            logger.info("Query: " + query)
            result = urlparse(query)
            if result.scheme and result.netloc:
                response = requests.get(query)
                soup = BeautifulSoup(response.text, 'html.parser')
                query = soup.body.get_text(strip=True)

            res = vector_store.similarity_search(query)

            data_list = []
            df = pd.read_csv(file_path)
            for doc in res:
                row = doc.metadata['row']
                assessment_dict = df.iloc[row].to_dict()

                test_type = assessment_dict['test_type']
                assessment_dict['test_type'] = [test_type_map.get(test, 'Unknown') for test in test_type]
                assessment_dict['url'] = "https://www.shl.com" + assessment_dict['url']
                data_list.append(assessment_dict)

            recommendations = data_list
            return JSONResponse(content = {"recommended_assessments" : recommendations}, status_code = status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error {e}")
            return JSONResponse(content={"message" : "error"}, status_code = status.HTTP_204_NO_CONTENT)

    @app.get("/metric")
    def get_metric():
        test_df = pd.read_csv('test.csv')

        recall_3_list = []
        precision_3_list = []
        for index, row in test_df.iterrows():
            query = row['query']

            results = vector_store.similarity_search_with_score(query, k=3)
            predicted_docs = []
            actual_docs = row['assessments'].split(' | ')

            for i, (doc, score) in enumerate(results):
                predicted_docs_name = doc.page_content.split("url: ")[0].split(":")[1].strip()
                predicted_docs.append(predicted_docs_name)

            recall_3_list.append(calculate_recall_3(actual_docs, predicted_docs))
            precision_3_list.append(calculate_precision_3(actual_docs, predicted_docs))

        mean_recall_3 = sum(recall_3_list) / len(recall_3_list)
        mean_precision_3 = sum(precision_3_list) / len(precision_3_list)

        print(f'Mean Recall@3: {round(mean_recall_3, 3)}')
        print(f'Mean Average Precision @3: {round(mean_precision_3, 3)}')

        return JSONResponse(content={"mean_recall_3" : mean_recall_3, "mean_avg_precision_3" : mean_precision_3})

    uvicorn.run(app, host = "localhost", port = 8003)

