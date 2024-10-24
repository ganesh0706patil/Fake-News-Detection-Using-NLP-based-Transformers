import os
import json
import requests
import hashlib
from bs4 import BeautifulSoup
import wikipedia
import nltk
from sentence_transformers import SentenceTransformer, util
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Step 1: Set the custom path for NLTK data
nltk_data_path = '/Users/ganeshpatil/Desktop/nltk_data'  # Update this with your NLTK data folder
nltk.data.path.append(nltk_data_path)  # Add the custom path to NLTK's data search path

# Step 2: Download required NLTK data if necessary
# These will work now that NLTK knows where to look for the data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Load the BERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # A lightweight BERT variant

# Step 4: Setup Environment
def create_topic_folder(topic):
    main_folder = 'scraped_news'
    topic_path = os.path.join(main_folder, topic)
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    if not os.path.exists(topic_path):
        os.makedirs(topic_path)

# Step 5: Fetch News Articles Using News API
def fetch_news_articles_newsapi(topic, search_query, api_key, page_size=15):
    base_url = 'https://newsapi.org/v2/everything'
    params = {
        'q': search_query,
        'apiKey': api_key,
        'pageSize': page_size,
        'language': 'en',
        'sortBy': 'relevancy'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            print("No articles found for the given query.")
            return []

        article_texts = []
        for article in articles:
            title = article.get('title', 'No Title')
            description = article.get('description', 'No Description')
            content = article.get('content', 'No Content')
            url = article.get('url', 'No URL')
            full_content = fetch_full_content(url)
            if full_content == "Content not available":
                continue
            article_text = f"Title: {title}\nDescription: {description}\nContent: {full_content}\nURL: {url}\n"
            save_article(topic, article_text, url)
            article_texts.append(article_text)  # Collect article texts
        return article_texts  # Return list of article texts
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles from News API: {e}")
        return []

def fetch_full_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 403:
            return "Content not available"
        elif response.status_code != 200:
            return "Content not available"

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        return full_text if full_text else "Content not available"
    except Exception as e:
        return "Content not available"

def save_article(topic, content, url):
    file_name = hashlib.md5(url.encode()).hexdigest() + '.txt'
    path = os.path.join('scraped_news', topic)
    with open(os.path.join(path, file_name), 'w', encoding='utf-8') as f:
        f.write(content)

# Step 6: Retrieve Full Wikipedia Page Content
def fetch_wikipedia_page(query):
    try:
        page = wikipedia.page(query)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page available for the given query."
    except Exception as e:
        return f"Error retrieving Wikipedia page: {e}"

def save_wikipedia_page(content, topic):
    file_name = 'wiki.txt'
    path = os.path.join('scraped_news', topic)
    with open(os.path.join(path, file_name), 'w', encoding='utf-8') as f:
        f.write(content)

# Step 7: Preprocess Text
def preprocess_text(text):
    return text.lower()

# Step 8: Calculate Similarity
def calculate_similarity(document, query):
    document_embedding = model.encode(document, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(document_embedding, query_embedding)
    return similarity.item()

# Step 9: Main Function to Coordinate Fetching, Saving, and Retrieving
def main(search_query, topic):
    create_topic_folder(topic)

    news_api_key = 'News API Key'
    article_texts = fetch_news_articles_newsapi(topic, search_query, news_api_key)

    wiki_content = fetch_wikipedia_page(search_query)
    save_wikipedia_page(wiki_content, topic)

    # Calculate similarity scores
    results = []
    preprocessed_wiki = preprocess_text(wiki_content)
    for article in article_texts:
        preprocessed_article = preprocess_text(article)
        similarity_score = calculate_similarity(preprocessed_article, preprocessed_wiki)
        results.append((similarity_score, article))

    # Sort results by similarity score
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# Step 10: Create Tkinter GUI
def run_gui():
    def on_search():
        query = entry.get()
        topic = 'Space'  # You can customize the topic if needed
        results.delete(1.0, tk.END)  # Clear previous results
        if not query:
            messagebox.showwarning("Input Error", "Please enter a search query.")
            return
        ranked_articles = main(query, topic)
        for score, article in ranked_articles:
            results.insert(tk.END, f"Score: {score:.4f}\n{article[:100]}...\n\n")  # Display first 100 chars

    # Set up GUI window
    window = tk.Tk()
    window.title("News Article Similarity Ranking")
    window.geometry("600x400")

    # Search query input
    tk.Label(window, text="Enter Search Query:").pack(pady=10)
    entry = tk.Entry(window, width=50)
    entry.pack(pady=5)

    # Search button
    search_button = tk.Button(window, text="Search", command=on_search)
    search_button.pack(pady=10)

    # ScrolledText for displaying results
    results = scrolledtext.ScrolledText(window, width=70, height=15)
    results.pack(pady=10)

    # Start the GUI loop
    window.mainloop()

if __name__ == "__main__":
    run_gui()
