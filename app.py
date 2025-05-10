import streamlit as st
from PIL import Image

# Title and Heading
st.title("My First Streamlit App")
st.header("This is a header")
st.write("This is some text.")

# Input and Output
name = st.text_input("Enter your name:", value="Type here")
if st.button("Submit"):
    st.write(f"Hello, {name}! I love TAR UMT! ")

image = Image.open("img/tarumt.png")  # Replace with your image path
st.image(image, caption="TAR UMT, Beyond Education!")
