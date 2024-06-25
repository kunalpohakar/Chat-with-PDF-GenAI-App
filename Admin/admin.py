import boto3
import streamlit as st
import os
import uuid
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

# Set environment variables (replace these with your actual values)
os.environ['AWS_ACCESS_KEY_ID'] = 'XXXXXXXXXXXXXXXXX'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'XXXXXXXXXXXXXXXXXXXXXXX'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['BUCKET_NAME'] = 'XXXXXXXXXXXXXXXXXXX'

# Log environment variables for debugging
st.write(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
st.write(f"AWS_SECRET_ACCESS_KEY: {os.getenv('AWS_SECRET_ACCESS_KEY')}")
st.write(f"BUCKET_NAME: {os.getenv('BUCKET_NAME')}")

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# Update the model ID if necessary
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

#Define other functions as needed (e.g., split_text, create_vector_store)
def main():
    st.header("Admin site for Chat with PDF Project")
    # Implement your main logic here

if __name__ == "__main__":
    main()
def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    vectrostore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectrostore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    
    # Upload to S3 
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    return True

def main():
    st.header("This is Admin site for Chat with PDF Project")
    uploaded_file = st.file_uploader("Choose a File", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request ID : {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        
        st.write(f"Total Pages: {len(pages)}")
        
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted docs length: {len(splitted_docs)}")
        st.write("==========================")
        st.write(splitted_docs[0])
        st.write("===========================")
        st.write(splitted_docs[1])
        
        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)
        
        if result:
            st.write("Hurray!!!, PDF Processed Successfully.")
        else:
            st.write("Error!!!, please check the logs.")

if __name__ == "__main__":
    main()