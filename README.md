# Arabic-Website-Question-Answering-LM



# Arabic Website Question Answering System

This project is a complete pipeline for **Arabic-language question answering** based on scraped content from Egyptian websites. It integrates:

- Web scraping  
- Text chunking and preprocessing  
- Retrieval using BM25 and dense sentence embeddings  
- Answer generation via LLaMA 3.1 Nemotron hosted by NVIDIA  
- FastAPI backend for real-time Q&A  
- A simple React-based interface for user interaction

---

## Features

- **Web Scraper**  
  Scrapes and parses HTML content from selected Arabic websites (e.g., `gate.ahram.org.eg`, `egyptair.com`, `azhar.eg`)

- **Text Preprocessing**  
  Cleans the DOM, removes unwanted tags (ads, navs, footers), and chunks large text blocks into ~250-token segments

- **BM25 & Dense Retrieval**  
  - Traditional TF-IDF scoring (BM25-like)
  - Dense vector retrieval via `all-mpnet-base-v2` and FAISS

- **Answer Generation with LLM**  
  Sends the question and retrieved chunks to NVIDIA’s LLaMA 3.1 Nemotron via OpenAI-compatible API for answer generation

- **FastAPI Backend**  
  Serves a RESTful `/answer` endpoint that receives a user question and returns an LLM-generated answer

- **React Frontend**  
  A minimalist React app that allows users to:
  - Input questions in Arabic
  - Send them to the FastAPI API
  - View context-aware answers in real time

- **CORS Configuration**  
  CORS-enabled to support connections from `localhost:3000` (React frontend)

---

## Technologies Used

- **Backend**:  
  FastAPI, uvicorn, sentence-transformers, FAISS, transformers, OpenAI SDK, BeautifulSoup, scikit-learn

- **Frontend**:  
  React, Axios (or fetch)

---

## How It Works

1. **Scraping**  
   - Visits a list of Arabic websites  
   - Extracts relevant content using `requests` and `BeautifulSoup`  
   - Cleans and chunks text into manageable pieces  
   - Saves to `scraped_chunks.json`

2. **Embedding and Indexing**  
   - Embeds all text chunks using `all-mpnet-base-v2`  
   - Builds a FAISS index for fast vector search

3. **Dense Retrieval and Answering**  
   - Retrieves top-k relevant chunks for a user query  
   - Sends query and context to LLaMA 3.1 via NVIDIA's OpenAI-compatible API  
   - Returns a concise answer

4. **Frontend Interaction**  
   - User inputs a question on the React web interface  
   - The question is sent to the FastAPI `/answer` endpoint  
   - The LLM-generated answer is displayed immediately

---

## Example API Usage

**POST** `/answer`

```json
{
  "question": "ما هي شروط السفر على مصر للطيران؟"
}
```

**Response**:

```json
{
  "question": "ما هي شروط السفر على مصر للطيران؟",
  "answer": "يرجى مراجعة سياسة الشركة عبر الموقع الرسمي لتحديد شروط الحجز، الوزن المسموح، ومتطلبات التأشيرة."
}
```

---

## File Structure

```
project/
│
├── main.py                 # FastAPI backend with scraping, retrieval, QA logic
├── scraped_chunks.json     # Cleaned and chunked web content
├── frontend/               # React app
│   ├── App.js              # React UI for asking questions
│   └── ...
└── README.md               # This file
```

---

## Setup Instructions

### Backend

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the backend:
   ```bash
   python main.py
   ```

### Frontend (React)

1. Navigate to `frontend/` and install dependencies:
   ```bash
   npm install
   ```

2. Start the React app:
   ```bash
   npm start
   ```

---

## Notes

- Ensure your NVIDIA API key is active and valid in the backend script.
- For deployment, replace hardcoded domains and secure keys appropriately.
- Potential future improvements:
  - Multilingual support
  - Fine-tuned LLMs
  - Enhanced frontend with query history and better styling
