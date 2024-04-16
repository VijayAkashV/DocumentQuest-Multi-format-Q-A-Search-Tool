import streamlit as st
import random

# Function to get a random interesting fact
def get_random_fact():
    facts = [
        "Did you know? The longest word in the English language, according to the Oxford English Dictionary, is pneumonoultramicroscopicsilicovolcanoconiosis.",
        "Fun Fact: The first computer programmer was a woman named Ada Lovelace.",
        "Interesting Fact: The world's first website, info.cern.ch, was published on August 6, 1991, by Tim Berners-Lee.",
        "Trivia: The term 'Big Data' was first coined in the early 1990s.",
        "Fun Fact: The world's first computer bug was a moth trapped in a relay, discovered by Grace Hopper.",
        "Did you know? The average person has around 70,000 thoughts per day!",
        "Fun Fact: The world's largest data center is located in Langfang, China.",
        "Trivia: The term 'Artificial Intelligence' was coined in 1956 by John McCarthy."
    ]
    return random.choice(facts)

def main():
    st.title("🌟 Welcome to Q&A Hub! 📚🔍")
    st.write("""
    ### Have you ever struggled to find answers within different types of documents? Q&A Hub is here to simplify your search process by allowing you to ask questions and find answers across various data formats such as images, text files, PDFs, and CSV files.
    
    **How it Works:**
    
    1. 📁 **Upload Data**: Upload your documents containing the information you need to search.
    2. 💬 **Ask Questions**: Type your questions into the search bar.
    3. 🎉 **Get Answers**: Q&A Hub will analyze your documents and provide you with accurate answers.
    
    **Why Choose Q&A Hub?**
    
    - 🔄 **Versatile Search**: Whether it's a scanned image, a lengthy PDF, or a structured CSV file, Q&A Hub can handle it all.
    - ⏱️ **Efficient**: Save time and effort by swiftly finding answers within your documents.
    - 🖥️ **Intuitive Interface**: Easy-to-use interface for seamless navigation and interaction.
    - ✔️ **Accurate Results**: Our advanced algorithms ensure precise answers to your questions.
    
    Ready to streamline your search process? Get started now!
    """)
    
    st.subheader("💡 Did You Know?")
    st.write(get_random_fact())

if __name__ == "__main__":
    main()
