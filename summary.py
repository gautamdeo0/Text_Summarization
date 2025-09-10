from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set source language (Hindi or English)
def generate_translation(text, src_lang="en_XX", tgt_lang="hi_IN", max_length=100):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=max_length,
        num_beams=4
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# English → Hindi
en_text = "*Nature* is the beautiful and diverse world around us, filled with lush forests, majestic mountains, sparkling rivers, and vibrant wildlife. It provides fresh air, clean water, and nourishment, sustaining all life on Earth. The changing seasons, from blooming springs to snowy winters, showcase nature's endless wonders. Spending time in nature brings peace, reduces stress, and reminds us of the planet's delicate balance that we must protect."
hi_summary = generate_translation(en_text, src_lang="en_XX", tgt_lang="hi_IN")
print("\n✅ English to Hindi:")
print("English:", en_text)
print("Hindi Summary:", hi_summary)

# Hindi → English
hi_text = "प्रकृति हमारे चारों ओर सुंदर और विविधतापूर्ण विश्व है, जो घने जंगलों, भव्य पर्वतों, चमकती नदियों और जीवंत वन्य जीवों से भरा है। यह ताजा हवा, स्वच्छ पानी और पोषण प्रदान करता है, जो पृथ्वी पर सभी जीवों को बनाए रखता है। फूलते हुए स्रोतों से लेकर बर्फबारी शीत ऋतुओं तक, परिवर्तनशील मौसम प्रकृति के अनन्त चमत्कारों को प्रदर्शित करता है। प्रकृति में समय"
en_summary = generate_translation(hi_text, src_lang="hi_IN", tgt_lang="en_XX")
print("\n✅ Hindi to English:")
print("Hindi:", hi_text)
print("English Summary:", en_summary)
