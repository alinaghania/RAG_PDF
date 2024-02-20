import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from assets.htmlTemplates import bot_template, user_template, css


def get_pdf_text(pdf_docs):
    # text is a string that will store all the text from the pdfs
    text = ""

    for pdf in pdf_docs: # pdf_docs is upload by the user
        pdf_reader = PdfReader(pdf) # read the pdf with PyPDF2 reader object
        for page in pdf_reader.pages: # iterate over the pages of the pdf 
            text += page.extract_text()    # extract the text from the page and add it to the text string
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size=1000, # Chunk size on a 1000 characters
        chunk_overlap =200, # Start the next chunk 200 characters before the end of the previous chunk
        length_function =len # Use the len function to get the length of the text
    ) # create a new instance of the CharacterTextSplitter

    chunks = text_splitter.split_text(text) # split the text into chunks
    return chunks
# Embedding with OpenAI or HuggingFace
def get_vectorstore(text_chunks):

    #Choose your embeddings model 
    embeddings_openai = OpenAIEmbeddings() # create a new instance of the OpenAIEmbeddings
    embeddings_huggingface = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embeddings=embeddings_huggingface) #Create a vectorstore with the text chunks embedded with the embeddings model
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI() 
    memory = ConversationBufferMemory(memory_key=="Chat_history",return_messages=True) 
    #Create a new instance of the ConversationBufferMemory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory

    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with your PDFs")

    # Ajout d'un label au widget text_input
    question = st.text_input(label="Ask a question about your documents:")

    st.write(user_template.format("Hello Robot "), unsafe_allow_html=True)
    st.write(bot_template.format(" Hello Human "), unsafe_allow_html=True)

    

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Just here",type=['pdf'],accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):

                #1- get the data from the pdf
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #2- get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)


                #3-create vector store
                vectorstore = get_vectorstore(text_chunks)
                print(vectorstore)

                #4- Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore) #ADD SESSION STATE TO NOT RELOAD ALL THE CODE 

                   
if __name__ == "__main__":
    main()


