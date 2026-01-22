import streamlit as st
import pandas as pd
import ast
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------- Formatting Helpers ----------

def formatting_description(text):
    if pd.isna(text):
        return []
    try:
        text = BeautifulSoup(str(text), "html.parser").get_text()
        return text.lower().split()
    except Exception:
        return []

def formatting_p_attributes(text):
    extracted = []
    if pd.isna(text):
        return extracted
    if isinstance(text, dict):
        attributes = text
    else:
        try:
            attributes = ast.literal_eval(str(text))
        except Exception:
            return extracted
    for key in [
        "Top Fabric", "Occasion", "Sustainable", "Wash Care",
        "Top Pattern", "Top Shape", "Top Type",
        "Bottom Pattern", "Bottom Type", "Bottom Closure"
    ]:
        if key in attributes and isinstance(attributes[key], str):
            extracted.append(attributes[key].replace(" ", "").lower())
    return extracted

def formatting_price(value):
    if pd.isna(value):
        return []
    try:
        value = float(value)
    except Exception:
        return []
    if value <= 1000:
        return ["priceveryaffordable"]
    elif 1000 < value <= 3000:
        return ["priceaffordable"]
    elif 3000 < value <= 6000:
        return ["pricemoderate"]
    elif 6000 < value <= 9000:
        return ["priceexpensive"]
    else:
        return ["priceveryexpensive"]

# ---------- Build Model ----------

@st.cache_resource
def build_model(dataset_path: str):
    # Check if file exists
    if not os.path.exists(dataset_path):
        st.error(f"Dataset file '{dataset_path}' not found. Please ensure it's in the repo.")
        return None, None, None, None
    
    # Load dataset with error handling
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None, None, None
    
    # Validate required columns
    required_cols = ["name", "img", "brand", "description", "colour", "p_attributes", "price"]
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns in dataset. Required: " + ", ".join(required_cols))
        return None, None, None, None
    
    # Apply preprocessing
    df["description"] = df["description"].apply(formatting_description)
    df["p_attributes"] = df["p_attributes"].apply(formatting_p_attributes)
    df["price"] = df["price"].apply(formatting_price)
    
    # Convert lists to strings
    df["description"] = df["description"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    df["p_attributes"] = df["p_attributes"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    df["price"] = df["price"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    
    # Concatenate features into 'tag'
    df["tag"] = (
        df["brand"].astype(str) + " "
        + df["description"].astype(str) + " "
        + df["colour"].astype(str) + " "
        + df["p_attributes"].astype(str) + " "
        + df["price"].astype(str)
    )
    
    # TF-IDF vectorization (optimized for memory)
    tfidf = TfidfVectorizer(max_features=2000)  # Reduced from 5000 to save resources
    tfidf_matrix = tfidf.fit_transform(df["tag"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, similarity, tfidf, tfidf_matrix

# ---------- Recommendation Function ----------

def search_and_recommend(query, df, tfidf, tfidf_matrix, top_k=5):
    """Search by keywords and return top_k similar products."""
    if df is None or tfidf is None or tfidf_matrix is None:
        return []
    try:
        query_vec = tfidf.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append([df["name"].iloc[idx], df["img"].iloc[idx]])
        return results
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return []

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
st.title("👗 Fashion Recommendation System")

# Load dataset + model with spinner
with st.spinner("Building recommendation model... This may take a moment."):
    df, similarity, tfidf, tfidf_matrix = build_model("Fashion Dataset.csv")

# Check if model loaded successfully
if df is None:
    st.stop()  # Stop the app if model failed to load

# Search bar
query = st.text_input("Enter product name or keywords:")

if query:
    results = search_and_recommend(query, df, tfidf, tfidf_matrix)
    if results:
        st.subheader("Recommended Dresses")
        num_cols = min(3, len(results))  # Dynamic columns based on results
        cols = st.columns(num_cols)
        for i, (title, img_url) in enumerate(results):
            with cols[i % num_cols]:
                try:
                    st.image(img_url, caption=title, width=300)
                    st.markdown(f"[View Product]({img_url})")
                except Exception:
                    st.write(f"Error loading image for {title}")
    else:
        st.write("No recommendations found for your query.")
