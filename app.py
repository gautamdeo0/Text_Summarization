from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import time
import re
from rouge_score import rouge_scorer

app = Flask(__name__)

# Set cache directory to D drive
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_models'

# Download required NLTK data
nltk.download('punkt', quiet=True)

print("Loading models...")

# Initialize model for text processing
model_name = "facebook/bart-large-cnn"  # Using BART model which is better for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='D:/huggingface_models')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='D:/huggingface_models')

# Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Models loaded successfully! Using device: {device}")

def detect_language(text):
    """Detect language with error handling"""
    # Use character analysis for language detection
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    kannada_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total_chars = len(text.strip())
    if total_chars == 0:
        return 'en'
    
    # Calculate percentages
    hindi_percent = hindi_chars / total_chars
    kannada_percent = kannada_chars / total_chars
    english_percent = english_chars / total_chars
    
    if hindi_percent > 0.3:
        return 'hi'
    elif kannada_percent > 0.3:
        return 'kn'
    elif english_percent > 0.3:
        return 'en'
    return 'en'  # default to English if uncertain

def translate_text(text, src='auto', dest='en'):
    """Translate text using Google Translator with chunking and retries"""
    if not text.strip():
        return text
        
    if src == dest:
        return text
    
    try:
        # Split text into smaller chunks (Google Translate has a limit)
        max_chunk_size = 4500  # Google's limit is 5000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        translated_chunks = []
        max_retries = 3
        
        for chunk in chunks:
            # Try multiple times for each chunk
            for attempt in range(max_retries):
                try:
                    translator = GoogleTranslator(source=src, target=dest)
                    translated = translator.translate(text=chunk)
                    if translated and translated.strip():
                        translated_chunks.append(translated)
                        break  # Success, move to next chunk
                    else:
                        raise Exception("Empty translation result")
                except Exception as chunk_error:
                    print(f"Chunk translation error (attempt {attempt + 1}): {str(chunk_error)}")
                    if attempt == max_retries - 1:  # Last attempt
                        raise  # Re-raise the error if all attempts failed
                    time.sleep(2)  # Wait before retrying
        
        # Combine all translated chunks
        if translated_chunks:
            return ' '.join(translated_chunks)
        else:
            return "Translation error: No successful translations"
            
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"Translation error: Service temporarily unavailable"

def generate_summary(text, max_length=None, min_length=None):
    """Generate summary using BART model with optimized parameters for better ROUGE scores"""
    try:
        # Preprocess text
        text = text.strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Count input words
        word_count = len(text.split())
        
        # Calculate target summary length (30-40% of input for better coverage)
        target_min_length = max(30, int(word_count * 0.3))  # minimum 30 words
        target_max_length = max(50, int(word_count * 0.4))  # minimum 50 words
        
        # Convert word counts to token counts (approximate)
        token_min_length = int(target_min_length * 1.3)
        token_max_length = int(target_max_length * 1.3)
        
        print(f"Input words: {word_count}, Target summary length: {target_min_length}-{target_max_length} words")
        
        # Tokenize input text with better parameters
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        inputs = inputs.to(device)
        
        # Generate summary with optimized parameters
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=token_max_length,
            min_length=token_min_length,
            length_penalty=1.5,  # Reduced from 2.0 to allow more natural length
            num_beams=5,  # Increased from 4 for better quality
            early_stopping=True,
            no_repeat_ngram_size=2,  # Reduced from 3 to allow more natural repetition
            do_sample=True,  # Enable sampling for more natural text
            temperature=0.7,  # Add temperature for controlled randomness
            top_k=50,  # Add top-k sampling
            top_p=0.95  # Add nucleus sampling
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process summary
        summary = ' '.join(summary.split())  # Remove extra whitespace
        
        # Print summary statistics
        summary_words = len(summary.split())
        print(f"Summary length: {summary_words} words")
        
        return summary
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        # Fallback to extractive summarization if model fails
        sentences = text.split('.')
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        return text

def summarize_text(text, lang='en'):
    """Generate summary based on language"""
    if not text.strip():
        return {"error": "Empty text provided"}
    
    try:
        if lang == 'hi':
            # For Hindi text, translate to English first
            english_text = translate_text(text, src='hi', dest='en')
            if "error" in english_text:
                return {"error": english_text}
            
            # Generate summary in English
            english_summary = generate_summary(english_text)
            
            # Translate summary back to Hindi
            hindi_summary = translate_text(english_summary, src='en', dest='hi')
            return {"summary": hindi_summary}
            
        elif lang == 'kn':
            # For Kannada text, translate to English first
            english_text = translate_text(text, src='kn', dest='en')
            if "error" in english_text:
                return {"error": english_text}
            
            # Generate summary in English
            english_summary = generate_summary(english_text)
            
            # Translate summary back to Kannada
            kannada_summary = translate_text(english_summary, src='en', dest='kn')
            return {"summary": kannada_summary}
            
        else:
            # For English text, summarize directly
            summary = generate_summary(text)
            return {"summary": summary}
            
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return {"error": f"Summarization failed: {str(e)}"}

def calculate_rouge_scores(reference, candidate):
    """Calculate ROUGE scores between reference and candidate texts"""
    scores = rouge_scorer.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        direction = data.get('direction', 'English to Hindi')
        
        if not text:
            return jsonify({"error": "No text provided"})
            
        # Detect source language
        detected_lang = detect_language(text)
        
        # Handle translation based on direction
        if direction == 'English to Hindi':
            if detected_lang != 'en':
                return jsonify({"error": "Please enter English text for English to Hindi translation"})
                
            # Get English summary first
            eng_summary = summarize_text(text, lang='en')
            if "error" in eng_summary:
                return jsonify({"error": eng_summary["error"]})
                
            # Translate original text and summary to Hindi
            hindi_translation = translate_text(text, src='en', dest='hi')
            hindi_summary = translate_text(eng_summary["summary"], src='en', dest='hi')
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(text, eng_summary["summary"])
            
            response = {
                "translation": hindi_translation,
                "original_summary": eng_summary["summary"],
                "translated_summary": hindi_summary,
                "rouge_scores": rouge_scores
            }
            
        elif direction == 'English to Kannada':
            if detected_lang != 'en':
                return jsonify({"error": "Please enter English text for English to Kannada translation"})
                
            # Get English summary first
            eng_summary = summarize_text(text, lang='en')
            if "error" in eng_summary:
                return jsonify({"error": eng_summary["error"]})
                
            # Translate original text and summary to Kannada
            kannada_translation = translate_text(text, src='en', dest='kn')
            kannada_summary = translate_text(eng_summary["summary"], src='en', dest='kn')
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(text, eng_summary["summary"])
            
            response = {
                "translation": kannada_translation,
                "original_summary": eng_summary["summary"],
                "translated_summary": kannada_summary,
                "rouge_scores": rouge_scores
            }
            
        elif direction == 'Hindi to English':
            if detected_lang != 'hi':
                return jsonify({"error": "Please enter Hindi text for Hindi to English translation"})
                
            # Get Hindi summary first
            hindi_summary = summarize_text(text, lang='hi')
            if "error" in hindi_summary:
                return jsonify({"error": hindi_summary["error"]})
                
            # Translate text to English
            eng_translation = translate_text(text, src='hi', dest='en')
            
            # Get English summary
            eng_summary = summarize_text(eng_translation, lang='en')
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(eng_translation, eng_summary["summary"])
            
            response = {
                "translation": eng_translation,
                "original_summary": hindi_summary["summary"],
                "translated_summary": eng_summary["summary"],
                "rouge_scores": rouge_scores
            }
            
        else:  # Kannada to English
            if detected_lang != 'kn':
                return jsonify({"error": "Please enter Kannada text for Kannada to English translation"})
                
            # Get Kannada summary first
            kannada_summary = summarize_text(text, lang='kn')
            if "error" in kannada_summary:
                return jsonify({"error": kannada_summary["error"]})
                
            # Translate text to English
            eng_translation = translate_text(text, src='kn', dest='en')
            
            # Get English summary
            eng_summary = summarize_text(eng_translation, lang='en')
            
            # Calculate ROUGE scores
            rouge_scores = calculate_rouge_scores(eng_translation, eng_summary["summary"])
            
            response = {
                "translation": eng_translation,
                "original_summary": kannada_summary["summary"],
                "translated_summary": eng_summary["summary"],
                "rouge_scores": rouge_scores
            }
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Process error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"})

def test_summarization():
    """Test function to demonstrate proportional summarization"""
    test_cases = [
        # Short text (about 50 words)
        """Climate change is a global environmental challenge. Rising temperatures, melting ice caps, and extreme weather events are affecting ecosystems worldwide. Governments and individuals must take immediate action to reduce carbon emissions and protect our planet for future generations.""",
        
        # Medium text (about 100 words)
        """Artificial Intelligence has transformed the way we live and work. From virtual assistants to autonomous vehicles, AI technologies are becoming increasingly integrated into our daily lives. Machine learning algorithms can now process vast amounts of data to make predictions and decisions. However, this rapid advancement also raises important ethical questions about privacy, bias, and the future of human employment. Researchers and policymakers are working to ensure AI development remains beneficial and responsible while addressing potential risks.""",
        
        # Long text (about 200 words)
        """The human brain is one of the most complex and fascinating organs in existence. Containing approximately 86 billion neurons, it controls everything from basic bodily functions to complex cognitive processes. Scientists have made significant progress in understanding how the brain works, but many mysteries remain unsolved. Memory formation, consciousness, and emotional processing are areas of ongoing research. The brain's remarkable plasticity allows it to adapt and rewire itself throughout our lives, enabling learning and recovery from injuries. Recent studies have revealed the importance of sleep in brain health, showing how it helps clear toxic proteins and consolidate memories. Neuroscientists are also investigating the relationship between brain structure and various neurological conditions. Advanced imaging techniques have revolutionized our ability to observe brain activity in real-time, providing insights into decision-making processes and mental health disorders. Understanding the brain better could lead to breakthrough treatments for conditions like Alzheimer's, depression, and anxiety."""
    ]
    
    print("\n=== Testing Proportional Summarization ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i} ({len(text.split())} words):")
        print("Original:", text)
        print("\nSummary:", generate_summary(text))
        print("\n" + "="*50)

if __name__ == '__main__':
    # Run tests if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_summarization()
    else:
        app.run(debug=True) 