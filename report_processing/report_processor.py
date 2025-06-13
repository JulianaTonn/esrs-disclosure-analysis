import pymupdf
import requests
import os
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

TOP_K = 20
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
COMPRESSION = False
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class ReportProcessor:
    def __init__(self, path=None, url=None, top_k=TOP_K, queries=None):
        self.path = path
        self.url = url
        assert (path is not None) != (url is not None), "Either path or url must be provided"
        self.text_list = []
        self.all_text = ''
        # self.load_pdf() # if I want to automatically load the pdf when creating the object
        # self.extract_text() # if I want to automatically extract text when creating the object
        self.text_list = []
        self.all_text = ''
        self.top_k = top_k
        self.queries = queries or {}
        self.page_idx = []
        self.chunks = []


    def load_pdf(self):
        if self.path:
            self.pdf = pymupdf.open(self.path)
        else:
            response = requests.get(self.url)
            pdf_bytes = io.BytesIO(response.content)
            self.pdf = pymupdf.open(stream=pdf_bytes, filetype='pdf')


    def extract_text(self):
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ''.join(self.text_list)


    def _get_retriever(self, db_path):
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " "],
        )

        # text_list = [page.get_text() for page in self.pdf]
        # full_text = '\n\n'.join(text_list)
        
        for i, page in enumerate(self.pdf):
            page_chunks = text_splitter.split_text(page.get_text())
            self.page_idx.extend([i + 1] * len(page_chunks))
            self.chunks.extend(page_chunks)

        if os.path.exists(db_path):
            doc_search = FAISS.load_local(db_path, embeddings=embeddings)
        else:
            doc_search = FAISS.from_texts(
                self.chunks, 
                embeddings,
                metadatas=[{"source": str(i), "page": str(page_idx)} for i, page_idx in enumerate(self.page_idx)]
            )

            doc_search.save_local(db_path)

        self.retriever = doc_search.as_retriever(search_kwargs={"k": self.top_k})

        return self.retriever, doc_search


    def _retrieve_chunks(self):
        section_text_dict = {}
        for key in self.queries.keys():
            if key == 'general':
                docs_1 = self.retriever.get_relevant_documents(self.queries[key][0])[:5]
                docs_2 = self.retriever.get_relevant_documents(self.queries[key][1])[:5]
                docs_3 = self.retriever.get_relevant_documents(self.queries[key][2])[:5]
                section_text_dict[key] = docs_1 + docs_2 + docs_3
            else:
                section_text_dict[key] = self.retriever.get_relevant_documents(self.queries[key])
        return section_text_dict

    
# Notes: could include title and section recognition, but did not work well so far