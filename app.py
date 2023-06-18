import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

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
    #model = INSTRUCTOR('hkunlp/instructor-xl')
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    #embeddings = model.encode(chunks)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handler_user_input(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with your own PDFs :books:")


    user_input = st.text_input("Ask a question about the PDFs you have provided:")

    if user_input:
        handler_user_input(user_input)
    #if user_input:
    #    handler_user_input(user_input)

    #st.write(user_template.replace("{{MSG}}", "Hello!"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hi dude! :)"), unsafe_allow_html=True)

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

                #Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                


if __name__ == '__main__':
    main()