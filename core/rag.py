from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from dotenv import load_dotenv
import os
import time
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()




client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

search_clinet = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

def extract_text_from_pdf(file_path):
    '''
    Docstring for extract_text_from_pdf
    
    :param file_path: Description
    '''
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_text_into_chunks(data, chunk_size=400, chunk_overlap=20):
  '''
  Docstring for split_text_into_chunks
  
  :param data: data that needs to be split and processed
  :param chunk_size: size of each chunk (400 characters by default)
  :param chunk_overlap: Description
  '''
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  return text_splitter.split_documents(data)

def process_pdf(file_path):
    '''
    Docstring for process_pdf
    
    :param file_path: Description
    '''
    data = extract_text_from_pdf(file_path)
    chunks = split_text_into_chunks(data, chunk_size=200, chunk_overlap=10)
    batch = []
    for i, chunk in enumerate(chunks):

        chunk_id = str(uuid.uuid4())
        title = chunk.metadata["source"]
        embedding = get_embeddings_for_chunk(chunk.page_content)

        doc = {            
            "chunk_id": chunk_id,
            "parent_id": "",
            "title": title,
            "chunk": chunk.page_content,
            "text_vector": embedding
        }

        batch.append(doc)

        if len(batch) >= 10:
            print(f"Uploading batch to Azure Search... {len(batch)}")
            results = search_clinet.upload_documents(documents=batch)
            print(f"Uploaded {len(results)} documents.")
            batch = []
    
    if batch:
        print(f"Uploading remeaning batch to Azure Search... {len(batch)}")
        results = search_clinet.upload_documents(documents=batch)
        print(f"Uploaded {len(results)} documents.")

def get_embeddings_for_chunk(chunk, max_retries=5):
    '''
    Docstring for get_embeddings_for_chunk
    
    :param chunk: Description
    '''
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
                input=chunk
            )
            return response.data[0].embedding
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt
                print(f"Rate limited. Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Failed after retries")
        
if __name__ == '__main__':
    sample_pdf_path = "C:\\Users\\User\\Downloads\\Sagar_Garate_CV.pdf"
    process_pdf(sample_pdf_path)
    