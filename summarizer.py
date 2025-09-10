from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the pretrained MBart model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set source language to Hindi
tokenizer.src_lang = "hi_IN"

# Summary generation function
def generate_summary(text, max_length=50):
    # Tokenize the Hindi input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Generate summary in English
    generated_tokens = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]  # Force generation in English
    )

    # Decode the generated tokens into readable text
    summary = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return summary

# Example Hindi input
hindi_text = "भारत ने आज चंद्रमा पर अपना दूसरा मिशन सफलतापूर्वक भेजा। इसरो के वैज्ञानिकों ने चंद्रयान-2 को सफलतापूर्वक लॉन्च किया।"

# Generate and print the summary
english_summary = generate_summary(hindi_text)
print(f"Hindi Input: {hindi_text}")
print(f"English Summary: {english_summary}")
