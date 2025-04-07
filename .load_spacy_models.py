import spacy
import os

# Print some debug info
print("SPACY_DATA_DIR:", os.environ.get('SPACY_DATA_DIR', 'Not set'))
print("Current data path:", spacy.util.get_module_path())

# Try to load models
try:
    nlp_en = spacy.load('en_core_web_sm')
    print("Successfully loaded en_core_web_sm")
except Exception as e:
    print(f"Failed to load en_core_web_sm: {e}")

try:
    nlp_ru = spacy.load('ru_core_news_sm')
    print("Successfully loaded ru_core_news_sm")
except Exception as e:
    print(f"Failed to load ru_core_news_sm: {e}")
