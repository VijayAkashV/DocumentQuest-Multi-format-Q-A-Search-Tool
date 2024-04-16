from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()  # take environment variables from .env.
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load OpenAI model and get responses
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize our Streamlit app

st.set_page_config(page_title="Gemini Image Demo")

# Sidebar for image display, image input, and delete previous conversation button
st.sidebar.header("Image Input")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image.", use_column_width=True)
else:
    st.sidebar.info("Please upload an image.")

# Button to delete previous conversation
if st.sidebar.button("Delete Previous Conversation"):
    # Delete conversation log file
    conversation_file = "conversation_log.txt"
    if os.path.exists(conversation_file):
        os.remove(conversation_file)
        st.sidebar.success("Previous conversation cleared.")
    else:
        st.sidebar.warning("No previous conversation found.")

# Main area for input prompt, response, and conversation log
st.header("Gemini Application")
input_prompt = st.text_input("Input Prompt: ", key="input")
input_prompt_template = """
            You are an expert in understanding visual images.
            You will receive input images &
            you will have to answer questions based on the input image
            """
conversation_file = "conversation_log.txt"  # Define conversation log file

# If user presses Enter in the input prompt, trigger the conversation
if input_prompt:
    conversation_log = st.session_state.get("conversation_log", "")  # Retrieve conversation log from session state
    if st.session_state.get("input_prompt", "") != input_prompt:  # To avoid multiple triggers for the same input
        st.session_state.input_prompt = input_prompt
        if uploaded_file is not None:
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt_template, image_data, input_prompt)
            st.subheader("The Response is")
            st.write(response)
            
            # Append question and answer to conversation log
            conversation_log += f"Question: {input_prompt}\nAnswer: {response}\n\n"
            
            # Write conversation log to file
            with open(conversation_file, "a") as f:
                f.write(f"Question: {input_prompt}\nAnswer: {response}\n\n")
            
            # Display conversation log
            st.subheader("Conversation Log:")
            st.text_area("Conversation Log", value=conversation_log, height=300)
            
            # Store conversation log in session state
            st.session_state.conversation_log = conversation_log
            
        else:
            st.warning("Please upload an image first.")
