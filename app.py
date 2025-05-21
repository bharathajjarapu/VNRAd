import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatPerplexity
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
import streamlit as st

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        with open(txt, 'r', encoding='utf-8') as file:
            text += file.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("Indexes")

def ingest_data():

    pdf_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".pdf")]
    txt_files = [os.path.join("dataset", file) for file in os.listdir("dataset") if file.endswith(".txt")]
    
    pdf_text = get_pdf_text(pdf_files) if pdf_files else ""
    txt_text = get_txt_text(txt_files) if txt_files else ""
    
    combined_text = pdf_text + "\n" + txt_text
    text_chunks = get_text_chunks(combined_text)
    create_vector_store(text_chunks)

def get_conversational_chain():
    prompt_template = """
    You are VNRVJIET Admissions Chatbot, a friendly and helpful assistant for prospective students. 
    
    Initial Greeting:
    If this is the first message, provide a warm welcome and briefly explain the admission quotas:

    Mention these clearly:
    - CAT-A: Through State-level engineering entrance exam
    - CAT-B: Through Management Quota evaluated by JEE Mains Rank 
    - NRI: Through Management Quota for NRI students
    
    Guidelines:
    - Provide conversational, concise, clear answers based on the context
    - Do not answer any other questions other than those related to admissions of VNRVJIET.
    - If unsure, respond with "I apologize, I don't have that specific information. Would you like me to connect you with the admissions office and provide contact details?"
    
    Rank Check:
    - If user asks about seat for his rank check for estimate from previous years and inform user.
    - If user asks about CSE seat also include all the allied branches of CSE to check their rank.
    - If user asks about Seat also include all the branches of VNRVJIET to check the seat for their rank.
    - Link to https://vnrvjiet.ac.in/assets/pdfs/EAPCET_First-and-Last-Ranks-2024.pdf for cutoff ranks.

    Link to Admissions is https://vnrvjiet.ac.in/admission/

    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """

    model = ChatPerplexity(model="sonar", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = FastEmbedEmbeddings()
    new_db = FAISS.load_local("Indexes", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("VNR Admission Bot", page_icon=":school:")
    st.header("AI Admissions Assistant :school:")
    
    if not os.path.exists("Indexes"):
        st.write("Ingesting data, please wait...")
        ingest_data()
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I assist you with your admission journey today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your admission-related question here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = user_input(prompt, chat_history)
                    st.write(response)

            if response is not None:
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()