# Fake News Detection using NLP and Cosine Similarity

## Project Overview
This project implements a system to detect fake news by comparing news articles with content fetched from Wikipedia. The system uses BERT-based embeddings and cosine similarity to classify news articles as "true" or "fake." Articles are fetched using News API and NewsData.io, while relevant Wikipedia pages are used as a reference for truthfulness.

## Features
- Fetches news articles from **News API** and **NewsData.io API**.
- Scrapes full article content using **BeautifulSoup**.
- Retrieves Wikipedia page content related to a given query.
- Preprocesses the text (tokenization, lowercasing).
- Uses **BERT-based SentenceTransformer** to generate embeddings.
- Computes **cosine similarity** between news articles and Wikipedia content.
- Classifies news articles based on similarity scores.

## Technologies Used
- **Python 3.x**
- **Libraries**:
  - `requests` (for API calls)
  - `beautifulsoup4` (for web scraping)
  - `wikipedia-api` (for Wikipedia page retrieval)
  - `nltk` (for text preprocessing)
  - `sentence-transformers` (for generating BERT embeddings)
- **APIs**:
  - [News API](https://newsapi.org/register)
  - [NewsData.io API](https://newsdata.io/register)
