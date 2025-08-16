import os
import re
import requests
import io
import pymupdf
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class ReportProcessor:
    def __init__(self, chunk_size=350, chunk_overlap=50, embedding_model_name="Qwen/Qwen3-Embedding-0.6B"):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': 'cuda'} 
            )
    
    # 1. Parse the PDF
    def parse_pdf(self, path=None, url=None):
        assert (path is not None) != (url is not None), "Provide either a local path or a URL."
        if path:
            pdf = pymupdf.open(path)
        else:
            response = requests.get(url)
            pdf = pymupdf.open(stream=io.BytesIO(response.content), filetype='pdf')

        pages = [page.get_text() for page in pdf]
        return pages

    # 2. Chunk the text
    def chunk_text(self, pages):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "],
        )
        chunks, metadata = [], []
        for idx, page in enumerate(pages):
            page_chunks = splitter.split_text(page)
            chunks.extend(page_chunks)
            metadata.extend([{"page": str(idx + 1)}] * len(page_chunks))
        return chunks, metadata
    
    # 3. Generate and store vector representations
    def get_vectorstore(self, chunks, metadata, db_path):
        if os.path.exists(db_path):
            vectorstore = FAISS.load_local(db_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_texts(chunks, self.embedding_model, metadatas=metadata)
            vectorstore.save_local(db_path)
        return vectorstore   
    