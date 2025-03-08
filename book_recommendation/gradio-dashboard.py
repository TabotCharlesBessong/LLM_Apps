import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables
load_dotenv()

# Get the directory where your script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", script_dir)

# Construct absolute paths to data files
books_file = os.path.join(script_dir, "../DATA/books_with_emotions.csv")
descriptions_file = os.path.join(script_dir, "../DATA/tagged_description.txt")
cover_not_found = os.path.join(script_dir, "cover-not-found.jpg")

print("Books file path:", books_file)
print("Descriptions file path:", descriptions_file)

# Check if files exist
if not os.path.exists(books_file):
    print(f"Error: Books file not found at {books_file}")
    data_dir = os.path.join(script_dir, "../DATA")
    if os.path.exists(data_dir):
        print(f"Files in DATA directory: {os.listdir(data_dir)}")
    else:
        print(f"DATA directory not found at {data_dir}")
    exit(1)

if not os.path.exists(descriptions_file):
    print(f"Error: Descriptions file not found at {descriptions_file}")
    exit(1)

# Load data
books = pd.read_csv(books_file)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    cover_not_found,
    books["large_thumbnail"],
)

# Try loading the text file with different encodings
try:
    raw_documents = TextLoader(descriptions_file, encoding='utf-8').load()
except Exception as e:
    print(f"Failed to load with UTF-8 encoding: {e}")
    try:
        raw_documents = TextLoader(descriptions_file, encoding='latin-1').load()
    except Exception as e:
        print(f"Failed to load with latin-1 encoding: {e}")
        raw_documents = TextLoader(descriptions_file, encoding='utf-8', errors='replace').load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()