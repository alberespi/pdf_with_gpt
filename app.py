import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vectorstore(chunks):
    model = INSTRUCTOR('hkunlp/instructor-xl')
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    embeddings = model.encode(chunks)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore


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
                chunks = get_text_chunks(raw_text)
                #st.write(chunks)

                #Create vector embedding store (knowledge base)
                vectorstore = get_vectorstore(chunks)
                


if __name__ == '__main__':
    main()