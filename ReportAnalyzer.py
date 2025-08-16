# report_analyzer.py
import os
import re
import json
import torch
import gc
import transformers
import pandas as pd
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ReportAnalyzer:
    def __init__(self, queries, guidelines, model_name="meta-llama/Llama-3.1-8B-Instruct", 
                 answer_length=200, max_token=500, temperature=0.01, batch_size=64):
        
        self.QUERIES = queries
        self.GUIDELINES = guidelines
        self.answer_length = answer_length
        self.batch_size = batch_size

        self.prompt_template = PromptTemplate(
            template=self._get_prompt_template(),
            input_variables=["query", "sources", "guideline", "answer_length"]
        )
        self.system_prompt = "You are an AI assistant in the role of a Senior Equity Analyst..."

        # Load model
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
        )
        model.eval()

        self.generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            return_full_text=True,
            temperature=temperature,
            max_new_tokens=max_token,
            batch_size=batch_size
        )

    def _get_prompt_template(self):
        return """You are a senior sustainabiliy analyst... 
        (full prompt text from your original code)"""

    def _docs_to_string(self, docs):
        output = ""
        for doc in docs:
            output += f"Content: {doc.page_content}\nSource: {doc.metadata['page']}\n\n---\n"
        return output

    def _find_verdict(self, answer_text):
        match = re.search(r'\[\[\s*(YES|NO)\s*\]\]', answer_text, re.IGNORECASE)
        return match.group(1).upper() if match else "N/A"

    # 4. Retrieve relevant chunks
    def retrieve_chunks(self, vectorstore, queries, report_id, top_k=8):
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        section_text_dict = {key: retriever.invoke(query) for key, query in queries.items()}
        return {report_id: section_text_dict}
    
    # 5. Prepare Prompts and Metadata
    def prepare_prompts(self, report_list, section_text_dict):
        prompts, metadata, results = [], [], {r: {} for r in report_list}
        for report in report_list:
            for key, query_text in self.QUERIES.items():
                context = self._docs_to_string(section_text_dict[report].get(key, []))
                if not context.strip():
                    results[report][key] = {"verdict": "NO", "analysis": "No relevant context", "sources": []}
                    continue
                prompt_text = self.prompt_template.format(
                    query=query_text, sources=context, 
                    guideline=self.GUIDELINES.get(key, ""), 
                    answer_length=self.answer_length
                )
                prompts.append([{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt_text}])
                metadata.append({"report": report, "key": key})
        return prompts, metadata, results

    # 6. Run inference in batches
    def run_batched_inference(self, prompts):
        return self.generate_text(prompts, batch_size=self.batch_size)

    # 7. Parse results
    def parse_results(self, responses, metadata, existing_results):
        for meta, response in zip(metadata, responses):
            report, key = meta["report"], meta["key"]
            full_text = response[0]['generated_text'][-1]['content']
            json_match = re.search(r'\{.*?\}', full_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    answer, sources = parsed.get("ANSWER", ""), parsed.get("SOURCES", [])
                except json.JSONDecodeError:
                    answer, sources = full_text, []
            else:
                answer, sources = full_text, []
            verdict = self._find_verdict(answer)
            existing_results[report][key] = {
                "verdict": verdict, "analysis": answer, "sources": sources
            }

        # empty GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        return existing_results
