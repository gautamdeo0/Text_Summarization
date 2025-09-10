from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
import time
from huggingface_hub import HfFolder
import requests
from tqdm import tqdm

# Set Hugging Face cache directory to D drive and enable offline mode after first download
os.environ['HF_HOME'] = 'D:/huggingface_models'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Enable offline mode

# Create cache directory if it doesn't exist
os.makedirs('D:/huggingface_models', exist_ok=True)

def download_with_retry(url, filename, max_retries=5):
    """Download file with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            return True
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            time.sleep(5)  # Wait 5 seconds before retrying
            continue
    return False

class KannadaSummarizer:
    def __init__(self):
        # Initialize models and tokenizers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Use smaller models and optimize loading
        self.mt5_model_name = "csebuetnlp/mT5_multilingual_XLSum"
        self.bert_model_name = "bert-base-multilingual-cased"
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        try:
            print("Loading MT5 model...")
            self.mt5_tokenizer = AutoTokenizer.from_pretrained(
                self.mt5_model_name,
                cache_dir='D:/huggingface_models',
                local_files_only=True,  # Use local files only after first download
            )
            self.mt5_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.mt5_model_name,
                cache_dir='D:/huggingface_models',
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map='auto'  # Optimize device placement
            )
            print("MT5 model loaded!")
        except Exception as e:
            print(f"Error loading MT5 model: {str(e)}")
            self.mt5_model = None
            self.mt5_tokenizer = None
        
        try:
            print("Loading BERT model...")
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                self.bert_model_name,
                cache_dir='D:/huggingface_models',
                local_files_only=True
            )
            self.bert_model = BertModel.from_pretrained(
                self.bert_model_name,
                cache_dir='D:/huggingface_models',
                local_files_only=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            print("BERT model loaded!")
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            self.bert_model = None
            self.bert_tokenizer = None
        
        print("Model loading completed!")

    def summarize_mt5(self, text, max_length=150):
        """Summarize Kannada text using mT5 model"""
        if self.mt5_model is None or self.mt5_tokenizer is None:
            return "MT5 model not available"
            
        try:
            input_text = "summarize: " + text
            inputs = self.mt5_tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                summary_ids = self.mt5_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=2,  # Reduced beam size for faster generation
                    early_stopping=True
                )
            return self.mt5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in MT5 summarization: {str(e)}")
            return "Error generating summary"

    def summarize_bert_extractive(self, text, num_sentences=3):
        """Extractive summarization using BERT embeddings"""
        if self.bert_model is None or self.bert_tokenizer is None:
            return "BERT model not available"
            
        try:
            # Split text into sentences
            sentences = text.split('।')  # Kannada full stop
            sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
            
            if not sentences:
                return text
            
            # Get BERT embeddings for each sentence
            sentence_embeddings = []
            for sent in sentences:
                inputs = self.bert_tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                sentence_embeddings.append(embedding)
            
            # Compute similarity matrix
            sim_matrix = cosine_similarity(sentence_embeddings)
            
            # Rank sentences
            sentence_scores = np.sum(sim_matrix, axis=1)
            ranked_sentences = [sent for _, sent in sorted(zip(sentence_scores, sentences), reverse=True)]
            
            # Return top sentences
            return '। '.join(ranked_sentences[:num_sentences]) + '।'
        except Exception as e:
            print(f"Error in BERT summarization: {str(e)}")
            return "Error generating summary"

    def get_all_summaries(self, text):
        """Get summaries using all available methods"""
        summaries = {}
        
        if self.mt5_model and self.mt5_tokenizer:
            summaries['mt5'] = self.summarize_mt5(text)
        
        if self.bert_model and self.bert_tokenizer:
            summaries['bert'] = self.summarize_bert_extractive(text)
        
        if not summaries:
            summaries['error'] = "No models available for summarization"
        
        return summaries 