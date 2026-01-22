# 👗 Fashion Recommendation System

## 📌 Overview
The **Fashion Recommendation System** is a machine learning–powered web app built with **Streamlit**. It recommends fashion products based on user queries by leveraging **TF‑IDF vectorization** and **cosine similarity**.  

This project demonstrates:
- End‑to‑end ML pipeline design (preprocessing → vectorization → similarity search).
- Integration of **Streamlit** for interactive UI.
- Use of **BeautifulSoup** and `ast` for data cleaning.
- Deployment‑ready structure with reusable `.pkl` models and a requirements file.

---

## 🚀 Features
- **Keyword Search**: Enter product names or attributes (e.g., “red dress”, “summer jeans”).
- **Smart Recommendations**: Returns top similar products with images and links.
- **Preprocessing Pipeline**:
  - Cleans HTML descriptions.
  - Extracts product attributes.
  - Categorizes prices into affordability tiers.
- **TF‑IDF + Cosine Similarity**: Finds the closest matches in the dataset.
- **Streamlit UI**: Simple search bar, grid layout, and product previews.

---

## 🛠️ Tech Stack
- **Language**: Python 3.x  
- **Libraries**:  
  - `streamlit` (UI)  
  - `pandas` (data handling)  
  - `scikit-learn` (TF‑IDF, cosine similarity)  
  - `beautifulsoup4` (HTML parsing)  
  - `ast` (attribute parsing)  

---

## 📂 Project Structure
Fashion-Recommendation-System/
│
├── fashion_app.py          # Streamlit app (main entry point)
├── Fashion Dataset.csv     # Dataset of fashion products
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── (optional) tfidf.pkl, similarity.pkl  # Saved models if persisted


---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fashion-Recommendation-System.git
   cd Fashion-Recommendation-System

2. Create and activate a virtual environment:   
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate # Mac/Linux

3. Install dependencies:

   pip install -r requirements.txt   

---

## ▶️ Usage

Run the Streamlit app:

streamlit run fashion_app.py

Open the provided local URL (usually http://localhost:8501) in your browser.- Enter a query in the search bar (e.g., “blue kurta”).
- View recommended products with images and links.

## 📊 ExampleQuery: "red dress"
Output:- Elegant Red Evening Dress
- Casual Summer Red Dress
- Party Wear Red Gown
Each recommendation includes an image preview and a clickable product link.

## 📦 Requirements
All dependencies are listed in requirements.txt.

Install them with:

pip install -r requirements.txt

## 👩‍💻 Author
Komal Pandey
Early‑career Web Designer & AI/ML Developer
Passionate about creating professional, recruiter‑ready applications that showcase strengths in UI/UX, applied ML, and deployment strategies.

---
