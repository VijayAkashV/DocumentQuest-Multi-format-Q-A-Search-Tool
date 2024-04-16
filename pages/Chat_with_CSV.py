import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

conversation_container = None  # Global variable for conversation container
def get_csv_text(csv_file):
    df = pd.read_csv(csv_file)
    text = " ".join(df.values.flatten().astype(str)) 
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Prompt for Analyzing CSV File and Answering Questions

    You have provided a CSV file for analysis. Your task is to thoroughly analyze the file and provide detailed answers to any questions asked about its contents. It is important to be responsible and accurate in your explanations.

    Responsibility and Explanation:

    You are responsible for analyzing the CSV file thoroughly.
    Provide detailed explanations for any questions asked about specific details, such as the highest rating or maximum cost.
    Ensure that your answers are based on the data within the CSV file.
    Summary questions:

    If the users asks about the summary of the csv file provide a detailed summary about the file with all the necessary information.
    Answering Questions:

    When answering questions, ensure accuracy and completeness.
    Address all questions related to the CSV file, including specific details and broader analysis.
    Chart Suggestions:

    If the user asks for chart suggestions, recommend appropriate charts based on the data in the CSV file.
    Suggested charts may include line charts, histograms, box plots, scatter plots, and others that best represent the data.
    Also suggest the correct column names that will be perfect fit for the recommended chart.
    Overall Approach:

    Take a systematic approach to analyzing the CSV file.
    Pay attention to detail and provide comprehensive answers to all inquiries.
    Use visualizations effectively to enhance understanding and convey insights from the data.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def save_conversation(question, answer):
    with open("conversation.txt", "a") as f:
        f.write("Question: " + question + "\n")
        f.write("Answer: " + answer + "\n\n")

def delete_previous_conversation():
    if os.path.exists("conversation.txt"):
        os.remove("conversation.txt")
        st.success("Previous conversation cleared.")
        # Clear conversation displayed in the app
        st.session_state.conversation = ""
        # Force rerender of the conversation text area
        if conversation_container:
            conversation_container.empty()

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    save_conversation(user_question, response["output_text"])

    return response["output_text"]

def main():
    global conversation_container  # Declare conversation_container as global

    st.header("Chat with CSV using Gemini💁")

    if "conversation" not in st.session_state:
        st.session_state.conversation = ""

    user_question = st.text_input("Ask a Question from the CSV File")

    if user_question:
        answer = user_input(user_question)
        st.session_state.conversation += f"Question: {user_question}\nAnswer: {answer}\n\n"
        st.write("Reply: ", answer)
        st.text_area("Conversation", st.session_state.conversation, height=300)
        st.download_button("Download Conversation", st.session_state.conversation, file_name="conversation.txt", mime="text/plain")
        
        # Reset input field after query
        st.session_state.user_question = ""

    with st.sidebar:
        st.title("Menu:")
        csv_file = st.file_uploader("Upload your CSV File and Click on the Submit & Process Button", type=["csv"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                delete_previous_conversation()  # Clear previous conversation before processing CSV
                raw_text = get_csv_text(csv_file)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        if st.button("Clear Previous Conversation"):
            delete_previous_conversation()  # Clear previous conversation displayed in the app
            st.success("Done")

    # Store a reference to the conversation text area container
    conversation_container = st.empty()

if __name__ == "__main__":
    main()
