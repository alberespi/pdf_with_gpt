import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page.extract_page()
            text += page.extract_text()
    return text

def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

    st.header("Chat with your own PDFs :books:")
    st.text_input("Ask a question about the PDFs you have provided:")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload here your PDFs and click on 'Train'", accept_multiple_files=True)
        if st.button("Train"):
            with st.spinner("Training"):
                #Get pdf text:
                raw_text = get_pdf_text(pdf_docs)
                #Get all chunks from text

                #Create vector embedding store (knowledge base)


if __name__ == '__main__':
    main()