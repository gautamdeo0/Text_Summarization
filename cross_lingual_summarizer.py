from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator
import re
import torch
import os

class CrossLingualSummarizer:
    def __init__(self):
        # Set cache directory for models
        os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_models'
        
        print("Loading models...")
        
        # Initialize the summarization model
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"Models loaded successfully! Using device: {self.device}")

    def detect_language(self, text):
        """Detect if the text is in English or Kannada"""
        kannada_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(text.strip())
        if total_chars == 0:
            return 'en'
        
        # Calculate percentages
        kannada_percent = kannada_chars / total_chars
        english_percent = english_chars / total_chars
        
        if kannada_percent > 0.3:
            return 'kn'
        elif english_percent > 0.3:
            return 'en'
        return 'en'  # default to English if uncertain

    def translate_text(self, text, src='auto', dest='en'):
        """Translate text between English and Kannada"""
        if not text.strip():
            return text
            
        if src == dest:
            return text
        
        try:
            translator = GoogleTranslator(source=src, target=dest)
            translated = translator.translate(text=text)
            return translated if translated else "Translation error"
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"

    def generate_summary(self, text, max_length=150, min_length=30):
        """Generate abstractive summary using BART model"""
        try:
            # Clean the text
            text = text.strip()
            if not text:
                return "Empty text provided"
            
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.7
            )
            
            # Decode the summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Clean up the summary
            summary = summary.strip()
            if not summary:
                return "Unable to generate summary"
                
            return summary
            
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            return f"Summarization error: {str(e)}"

    def process_text(self, text):
        """Process text for cross-lingual summarization"""
        if not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            # Detect input language
            detected_lang = self.detect_language(text)
            
            if detected_lang == 'en':
                # English to Kannada flow
                # 1. Generate summary in English first
                english_summary = self.generate_summary(text)
                
                # 2. Translate original text to Kannada
                kannada_text = self.translate_text(text, src='en', dest='kn')
                
                # 3. Translate English summary to Kannada
                kannada_summary = self.translate_text(english_summary, src='en', dest='kn')
                
                return {
                    "original_text": text,
                    "translated_text": kannada_text,
                    "kannada_summary": kannada_summary,
                    "english_summary": english_summary
                }
                
            else:  # Kannada to English flow
                # 1. Translate Kannada to English
                english_text = self.translate_text(text, src='kn', dest='en')
                
                # 2. Generate summary in English
                english_summary = self.generate_summary(english_text)
                
                # 3. Translate English summary back to Kannada
                kannada_summary = self.translate_text(english_summary, src='en', dest='kn')
                
                return {
                    "original_text": text,
                    "translated_text": english_text,
                    "english_summary": english_summary,
                    "kannada_summary": kannada_summary
                }
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}

def get_user_input():
    """Get input text from user"""
    print("\n" + "="*50)
    print("Cross-Lingual Text Summarization System")
    print("="*50)
    print("\nEnter your text (English or Kannada):")
    print("(Press Enter twice to finish input)")
    
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    
    return "\n".join(lines)

def display_results(result):
    """Display the results in a formatted way"""
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    
    print("\nOriginal Text:")
    print("-"*20)
    print(result["original_text"])
    
    print("\nTranslated Text:")
    print("-"*20)
    print(result["translated_text"])
    
    if "kannada_summary" in result:
        print("\nKannada Summary:")
        print("-"*20)
        print(result["kannada_summary"])
    
    if "english_summary" in result:
        print("\nEnglish Summary:")
        print("-"*20)
        print(result["english_summary"])

def main():
    # Initialize the summarizer
    summarizer = CrossLingualSummarizer()
    
    while True:
        # Get input from user
        text = get_user_input()
        
        if not text.strip():
            print("\nNo text provided. Exiting...")
            break
        
        # Process the text
        result = summarizer.process_text(text)
        
        # Display results
        display_results(result)
        
        # Ask if user wants to continue
        choice = input("\nDo you want to process another text? (y/n): ").lower()
        if choice != 'y':
            print("\nThank you for using the Cross-Lingual Text Summarization System!")
            break

if __name__ == "__main__":
    main() 