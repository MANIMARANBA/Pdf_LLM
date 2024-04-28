import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template  # Import HTML templates

api_key="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"
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

# Define function to get LLM response
def get_llm_response(question, answer):
    template = "Question: {question}\n{answer}"
    prompt = PromptTemplate(template=template, input_variables=["question", "answer"])
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":4000}, huggingfacehub_api_token=api_key))
    response = llm_chain.run(question=question, answer=answer)
    return response

# Load PDF file and extract text
def load_and_extract_text(file_path):
    reader = PdfReader(file_path)
    pdf_text = ""
    page_numbers_to_read = [0]  # Specify which pages you want to read

    for page in page_numbers_to_read:
        page_text = reader.pages[page].extract_text()
        pdf_text += page_text

    return pdf_text

# Main function to run the Streamlit app
def main():
    # Title and description
    st.title("PDF to Text Converter and LLM Chatbot")
    st.write("This app converts PDF file to text and uses a language model to answer questions based on the extracted text.")
    
    # Add CSS styling from htmlTemplates.py
    st.write(css, unsafe_allow_html=True)

    # PDF file uploader
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])

    # Get user question
    question = st.text_input("Enter your question")

    if uploaded_file is not None:
        # Extract text from uploaded PDF file
        pdf_text = load_and_extract_text(uploaded_file)

        # Call the LLM with input data and instruction
        if st.button("Get LLM Response"):
           

            instruction = question
            response = get_llm_response(pdf_text, instruction)
            st.write("LLM Response:")
            
            # Display bot response using bot template
            st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
    
    

# Run the app
if __name__ == "__main__":
    main()
