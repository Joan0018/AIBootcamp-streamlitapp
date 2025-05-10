import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Set your OpenAI API key securely
openai.api_key = st.secrets["openai_key"]

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("csv/qa_dataset_with_embeddings.csv")
    df["Question_Embedding"] = df["Question_Embedding"].apply(ast.literal_eval)
    df["Question_Embedding"] = df["Question_Embedding"].apply(np.array)
    return df

df = load_data()

# Function to get embedding using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# Streamlit App UI
st.title("Heart, Lung, and Blood Q&A Bot 💬")
st.write("Ask any question related to heart, lung, or blood health.")

user_question = st.text_input("Enter your question:")

col1, col2 = st.columns(2)
search_clicked = col1.button("Search Answer")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.experimental_rerun()

if search_clicked and user_question.strip() != "":
    with st.spinner("Searching for the best answer..."):
        try:
            user_embedding = get_embedding(user_question)
            similarities = cosine_similarity([user_embedding], list(df["Question_Embedding"]))[0]
            top_index = np.argmax(similarities)
            top_score = similarities[top_index]

            if top_score > 0.80:
                st.success("Answer found:")
                st.markdown(f"**Q:** {df.iloc[top_index]['Question']}")
                st.markdown(f"**A:** {df.iloc[top_index]['Answer']}")
                st.markdown(f"**Similarity Score:** `{top_score:.4f}`")
            else:
                st.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
        except Exception as e:
            st.error(f"Error: {e}")
elif search_clicked:
    st.warning("Please enter a valid question.")

# Optional: Display FAQs
with st.expander("📋 Common FAQs"):
    for i in range(min(5, len(df))):
        st.markdown(f"**Q:** {df.iloc[i]['Question']}")
        st.markdown(f"**A:** {df.iloc[i]['Answer']}")
