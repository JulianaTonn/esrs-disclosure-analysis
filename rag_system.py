import os
import io
import re
import gc
import json
import requests
import torch
import pymupdf
import transformers
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Helper Functions ---

def _docs_to_string(docs):
    """Formats a list of LangChain Documents into a string for the prompt."""
    output = ""
    for i, doc in enumerate(docs):
        output += f"Source {i+1}:\nContent: {doc.page_content}\nPage: {doc.metadata.get('page', 'N/A')}\n---\n"
    return output

def _find_verdict(answer_text):
    """Extracts a [[YES]] or [[NO]] verdict from the analysis text."""
    if not answer_text: return "N/A"
    match = re.search(r'\[\[\s*(YES|NO)\s*\]\]', answer_text, re.IGNORECASE)
    return match.group(1).upper() if match else "N/A"

def _find_answer(full_text):
    try:
        for line in full_text.splitlines():
            if "ANSWER" in line:
                idx = line.find(":") + 1
                return line[idx:].strip().strip('",')
        return full_text.strip()  # fallback if no ANSWER found
    except Exception:
        return full_text.strip()

def _find_sources(full_text):
    # Attempt to match a SOURCES line first
    sources_match = re.search(r"SOURCES\s*:\s*\[([^\]]+)\]", full_text)
    if sources_match:
        number_list = re.findall(r'\d+', sources_match.group(1))
        return [int(n) for n in number_list]

    # Fallback: extract all numbers from full text
    return [int(n) for n in re.findall(r'\b\d{3,5}\b', full_text)]  # assumes sources have 3-5 digits



class RAGSystem:
    """An all-in-one class to process, retrieve, and analyze sustainability reports."""

    def __init__(self, esrs_metadata_path, chunk_size=350, chunk_overlap=50, top_k=8, max_new_tokens=500):
        """
        Initializes the entire system by loading models, prompts, and settings.
        """
        print("Initializing RAG System...")
        # Settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.answer_length = 200

        # Load prompts and guidelines
        self.queries, self.guidelines = self._load_esrs_metadata(esrs_metadata_path)
        self.system_prompt, self.disclosure_prompt = self._setup_prompts()
        
        # Load models
        self.embedding_model = self._load_embedding_model()
        self.tokenizer, self.model = self._load_generation_model()
        
        # Create inference pipeline
        self.generate_text_pipeline = self._create_pipeline(max_new_tokens)
        
        print("RAG System initialized successfully.")

    ### --- SETUP METHODS (called by __init__) --- ###
    
    def _load_esrs_metadata(self, path):
        esrs_metadata = pd.read_excel(path)
        queries = dict(zip(esrs_metadata["query_id"], esrs_metadata["query"]))
        guidelines = dict(zip(esrs_metadata["query_id"], esrs_metadata["guidelines"]))
        return queries, guidelines

    def _setup_prompts(self):
        system_prompt = "You are an AI assistant in the role of a Senior Equity Analyst with expertise in sustainability reporting that analyzes companys' annual reports."
        prompt_template_str = """
You are a senior sustainabiliy analyst with expertise in the european reporting standards evaluating a company's disclosure on social sustainability.


You are presented with the following sources from the company's annual report:
--------------------- [BEGIN OF SOURCES]\n
{sources}\n
--------------------- [END OF SOURCES]\n

Given the sources information and no prior knowledge, your main task is to respond to the posed question encapsulated in "||".
Question: ||{query}||

Please consider the following additional explanation to the question encapsulated in "+++++" as crucial for answering the question:
+++++ [BEGIN OF EXPLANATION]
{guideline}
+++++ [END OF EXPLANATION]

### Response Instructions ###
Please enforce to the following guidelines in your ANSWER:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
4. Cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
5. Always acknowledge that the information provided is representing the company's view based on its report.
6. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
7. Start your ANSWER with a "[[YES]]"" or ""[[NO]]"" depending on whether you would answer the question with a yes or no. Always complement your judgement on yes or no with a short explanation that summarizes the sources in an informative way, i.e. provide details.
8. Keep your ANSWER within {answer_length} words.

### Formatting Instructions ###
- Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the SOURCE numbers that were referenced in your answer).
- Your response **must** be returned as a **valid JSON object**.
- Only output the JSON object — no preamble, no markdown, no extra commentary.
- Use this exact format for your final output:
{{
  "ANSWER": "[[YES]] or [[NO]] Here follows your explanation",
  "SOURCES": ["1", "216", "181-182", "174"]
}}

Your FINAL_ANSWER in JSON (ensure there's no format error):
"""
        disclosure_prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["query", "sources", "guideline", "answer_length"]
        )
        return system_prompt, disclosure_prompt

    def _load_embedding_model(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        print(f"Loading embedding model: {model_name}...")
        return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'})

    def _load_generation_model(self, llm_name="meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading generation model: {llm_name}...")
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.bfloat16,
            device_map="auto", quantization_config=quantization_config
        )
        model.eval()
        return tokenizer, model

    def _create_pipeline(self, max_new_tokens, temperature=0.01):
        return transformers.pipeline(
            model=self.model, 
            tokenizer=self.tokenizer, 
            task='text-generation',
            return_full_text=True, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens,
            batch_size=65
        )

    ### --- PROCESSING METHODS --- ###
    def _parse_pdf(self, path=None, url=None):
        assert (path is not None) != (url is not None), "Provide either a local path or a URL."
        if path:
            pdf = pymupdf.open(path)
        else:
            response = requests.get(url)
            pdf = pymupdf.open(stream=io.BytesIO(response.content), filetype='pdf')
        pages = [page.get_text() for page in pdf]
        return pages


    def _chunk_text(self, pages):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
            length_function=len, separators=["\n\n", "\n", " "],
        )
        chunks, metadata = [], []
        for idx, page in enumerate(pages):
            page_chunks = splitter.split_text(page)
            chunks.extend(page_chunks)
            metadata.extend([{"page": str(idx + 1)}] * len(page_chunks))
        return chunks, metadata

    def _get_vectorstore(self, chunks, metadata, db_path):
        if os.path.exists(db_path):
            print(f"Loading existing vector store from {db_path}...")
            return FAISS.load_local(db_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        else:
            print(f"Creating new vector store at {db_path}...")
            vectorstore = FAISS.from_texts(chunks, self.embedding_model, metadatas=metadata)
            vectorstore.save_local(db_path)
            return vectorstore


    def _retrieve_chunks(self, vectorstore, report_id):
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        return {report_id: {key: retriever.invoke(query) for key, query in self.queries.items()}}
        
    def _prepare_prompts(self, report_id, section_text_dict):
        prompts, metadata, final_results = [], [], {report_id: {}}
        for key, query_text in self.queries.items():
            context_str = _docs_to_string(section_text_dict[report_id].get(key, []))
            if not context_str.strip():
                final_results[report_id][key] = {
                    "verdict": "NO", 
                    "analysis": "No context was retrieved to answer the question.", 
                    "sources": []}
                continue
            prompt_text = self.disclosure_prompt.format(
                query=query_text, sources=context_str,
                guideline=self.guidelines.get(key, ""), answer_length=self.answer_length
            )
            prompts.append([
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text}
            ])
            metadata.append({"report": report_id, "key": key})
        return prompts, metadata, final_results


    def _parse_results(self, responses, metadata, existing_results):
        for meta, response in zip(metadata, responses):
            report, key = meta["report"], meta["key"]
            full_text = response[0]['generated_text'][-1]['content']
            json_match = re.search(r'\{.*?\}', full_text, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    answer = parsed_json.get("ANSWER", "")
                    sources = parsed_json.get("SOURCES", [])
                except json.JSONDecodeError:
                    answer = _find_answer(full_text)
                    sources = _find_sources(full_text)
            else:
                answer = _find_answer(full_text)
                sources = _find_sources(full_text)
            
            verdict = _find_verdict(answer)
            if verdict in [None, "N/A"]:
                answer = "N/A"
                sources = "N/A"
                
            existing_results[report][key] = {
                "verdict": verdict,
                "analysis": answer,
                "sources": sources
            }
            
        return existing_results
        
    def _clear_memory(self):
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()

    
    ### --- MAIN PUBLIC METHOD --- ###

    def process_and_analyze_report(self, report_id, db_path, pdf_path=None, pdf_url=None):
        """
        Runs the full end-to-end pipeline for a single report from path or URL.
        """
        # 1. Process Document only if the DB doesn't exist
        if not os.path.exists(db_path):
            pages = self._parse_pdf(path=pdf_path, url=pdf_url)
            chunks, metadata = self._chunk_text(pages)
            vectorstore = self._get_vectorstore(chunks, metadata, db_path)
        else:
            vectorstore = self._get_vectorstore(None, None, db_path)
        
        # 2. Retrieve, Prepare, Infer, and Parse
        retrieved_chunks = self._retrieve_chunks(vectorstore, report_id)
        prompts, metadata, final_results = self._prepare_prompts(report_id, retrieved_chunks)
        
        if prompts:
            print(f"Starting augmented generation for {len(prompts} Prompts...")
            responses = self.generate_text_pipeline(prompts)
        else:
            responses = []
        #print("DEBUG responses:", responses)
            
        final_analysis = self._parse_results(responses, metadata, final_results)

        # 3. Clear Memory
        self._clear_memory()

        print(f"--- Finished Pipeline for: {report_id} ---")
        return final_analysis
