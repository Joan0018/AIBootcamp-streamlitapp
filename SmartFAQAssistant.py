import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # Light and efficient

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("csv/qa_dataset_with_embeddings.csv")
    # Convert the stringified embedding to a real list
    df['Embedding'] = df['Question_Embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df

df = load_data()

# Streamlit UI
st.title("💬 Health Q&A Chat (Heart, Lung, and Blood Topics)")
st.write("Ask any question related to heart, lung, or blood health.")

user_question = st.text_input("Enter your health question:")
submit = st.button("Get Answer")
clear = st.button("Clear")

if clear:
    st.experimental_rerun()

if submit and user_question.strip():
    # Step 1: Embed the user's question
    user_embedding = model.encode([user_question])[0]

    # Step 2: Compute cosine similarity
    similarities = cosine_similarity([user_embedding], df['Embedding'].tolist())[0]

    # Step 3: Find the best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    threshold = 0.75  # Tune this as needed

    if best_score >= threshold:
        st.success("✅ Best Match Found:")
        st.markdown(f"**Question:** {df.iloc[best_idx]['Question']}")
        st.markdown(f"**Answer:** {df.iloc[best_idx]['Answer']}")
        st.caption(f"Similarity Score: {best_score:.2f}")
        
        # Optional: Rating
        with st.expander("Rate this answer"):
            rating = st.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True)
            st.write("Thank you for your feedback!" if rating else "")
    else:
        st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")

# Optional: Display common FAQs
with st.expander("📚 Browse Common Questions"):
    st.write(df[['Question']].head(10).to_markdown(index=False))
