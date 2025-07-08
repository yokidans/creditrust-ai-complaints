import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from transformers import pipeline
import logging
from tqdm import tqdm
import torch
from transformers import BitsAndBytesConfig

class EliteRAGSystem:
    def __init__(self, vector_store, llm_model="HuggingFaceH4/zephyr-7b-beta"):
        self.vector_store = vector_store
        self.llm = self._init_llm(llm_model)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.logger = logging.getLogger(__name__)
        
  
    def _init_llm(self, model_name):
        """Initialize LLM with proper quantization"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        return pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code= True   
        )

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid retrieval with semantic + keyword search"""
        # Semantic search
        semantic_results = self.vector_store.similarity_search(query, k=k*3)
        
        # Rerank with cross-encoder
        pairs = [(query, doc.text) for doc in semantic_results]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Combine and sort
        ranked_results = []
        for doc, score in zip(semantic_results, rerank_scores):
            doc.metadata["similarity_score"] = float(score)
            ranked_results.append(doc)
        
        return sorted(ranked_results, key=lambda x: x.metadata["similarity_score"], reverse=True)[:k]

    def generate_response(self, query: str, context: List[str]) -> str:
        """Elite prompt engineering with meta-reasoning"""
        context_str = '\n'.join([f'--- Excerpt {i+1} ---\n{text}\n' for i, text in enumerate(context)])
        prompt = f"""You are CrediTrust's Chief AI Analyst. Adhere strictly to:
1. Analyze these {len(context)} complaint excerpts
2. Identify root causes and patterns
3. Propose 3 actionable solutions
4. Rate severity (1-10)

Context:
{context_str}

Question: {query}

Response Template:
Analysis: <pattern detection>
Root Causes: <bulleted list>
Solutions: <numbered actionable items>
Severity: <rating with justification>"""
        
        response = self.llm(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.9
        )
        return response[0]["generated_text"]