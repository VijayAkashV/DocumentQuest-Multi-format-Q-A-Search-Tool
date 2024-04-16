import streamlit as st
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
def get_text_from_files(text_files):
    text = ""
    for file in text_files:
        text += file.getvalue().decode("utf-8") + "\n"
    return text.strip()  # Strip trailing newline characters

def get_text_chunks(text):
    # Split text into chunks, you can use your preferred method here
    # For simplicity, let's split by newline characters
    chunks = text.split('\n')
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,if the user asks abou the summary provide a detailed summary of the file ,
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
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

    st.set_page_config("Chat Text Files")
    st.header("Chat with Text Files using Gemini💁")
    if "conversation" not in st.session_state:
        st.session_state.conversation = ""

    user_question = st.text_input("Ask a Question from the Text Files")

    if user_question:
        answer = user_input(user_question)
        st.session_state.conversation += f"Question: {user_question}\nAnswer: {answer}\n\n"
        st.write("Reply: ", answer)
        st.text_area("Conversation", st.session_state.conversation, height=300)
        st.download_button("Download Conversation", st.session_state.conversation, file_name="conversation.txt", mime="text/plain")

    with st.sidebar:
        st.title("Menu:")
        text_files = st.file_uploader("Upload your Text Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                delete_previous_conversation()
                raw_text = get_text_from_files(text_files)
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
