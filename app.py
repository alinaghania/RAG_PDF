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
from langchain_core.messages import AIMessage,HumanMessage


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
    embeddings= OpenAIEmbeddings() # create a new instance of the OpenAIEmbeddings
    #embeddings= HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #Create a vectorstore with the text chunks embedded with the embeddings model
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="Chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if 'conversation' in st.session_state and st.session_state.conversation:
        # Prepare the input with both 'question' and 'chat_history'
        conversation_input = {
            'question': user_question,
            'chat_history': st.session_state.Chat_history
        }

        # Get the response from the conversation model
        response = st.session_state.conversation(conversation_input)

        # Update the chat history in the session state
        st.session_state.Chat_history = response.get('chat_history', [])

        # Display messages
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation model is not initialized. Please upload and process your PDFs.")



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with your PDFs :books:")

    user_question = st.text_input(label="**Ask a question about your documents:**")

    # Initialize chat_history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        handle_userinput(user_question)

    # Rest of your code...

    st.write(user_template.replace("{{MSG}}","Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Just here", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize conversation chain and history
                # Initialize conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.conversation_history = []

if __name__ == "__main__":
    main()