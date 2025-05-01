import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# --------- CONFIGURATION ---------
HEADERS = {'User-Agent': 'Mozilla/5.0'}
URLS = [
    "https://gate.ahram.org.eg/",
    "https://www.egyptair.com/en/pages/HomePage.aspx",
    "https://www.azhar.eg/splash.html"
]
CHUNK_SIZE = 250

# --------- FASTAPI INIT ---------
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- UTILITIES ---------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text_from_soup(soup):
    for tag in soup(['script', 'style', 'footer', 'nav', 'noscript']):
        tag.decompose()
    elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
    content = []
    for el in elements:
        if el.get('class') and any(cls in el['class'] for cls in ['ads', 'footer', 'nav']):
            continue
        text = el.get_text(separator=' ', strip=True)
        if text:
            content.append(clean_text(text))
    return " ".join(content)


# --------- SCRAPER ---------
def scrape_and_preprocess():
    all_data = []
    for url in URLS:
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            title = soup.title.string.strip() if soup.title else "No Title"
            base_url = res.url
            raw_text = extract_text_from_soup(soup)
            chunks = chunk_text(raw_text, CHUNK_SIZE)
            for chunk in chunks:
                all_data.append({"url": base_url, "title": title, "text": chunk})
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    with open("scraped_chunks.json", "w", encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


# --------- BM25 RETRIEVAL ---------
def bm25_search(query, corpus_texts, top_k=3):
    vectorizer = TfidfVectorizer().fit(corpus_texts)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(corpus_texts)
    scores = (doc_vecs * query_vec.T).toarray().flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [corpus_texts[i] for i in top_indices]


# --------- DENSE RETRIEVAL + RAG ---------
retriever_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

with open("scraped_chunks.json", encoding='utf-8') as f:
    dataset = json.load(f)
    text_chunks = [d['text'] for d in dataset]
    embeddings = retriever_model.encode(text_chunks, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())



def dense_search(query, top_k=3):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True)
    D, I = index.search(query_embedding.cpu().numpy(), top_k)
    return [text_chunks[i] for i in I[0]]


def generate_answer(query, contexts):

    messages = [
        {
            "role": "system",
            "content": "You are a question answering agent that uses context information to answer the user briefly"
        },
        {
            "role": "user",
            "content": "question: {} context: {}".format(query, " ".join(contexts))

        }
    ]

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="your api here"
    )

    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=messages,
        temperature=0.1,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    return response


# --------- FASTAPI ROUTE ---------
class QAQuery(BaseModel):
    question: str

@app.post("/answer")
def answer_question(q: QAQuery):
    query = q.question
    retrieved = dense_search(query)
    print(retrieved)
    answer = generate_answer(query, retrieved)
    return {"question": query, "answer": answer}

# --------- MAIN ENTRY ---------
if __name__ == "__main__":
    scrape_and_preprocess()
    uvicorn.run(app, host="0.0.0.0", port=8000)
