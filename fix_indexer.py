#!/usr/bin/env python3
"""
Patch script to fix issues in the Lecture Video Content Indexer.
"""

import os
import sys
import logging
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_search_engine_fix():
    """Fix the execute_update issue in SearchEngine.index_content."""
    # Path to the search_engine.py file
    search_engine_path = "search_engine.py"

    if not os.path.exists(search_engine_path):
        logger.error(f"File not found: {search_engine_path}")
        return False

    # Read the current content
    with open(search_engine_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the file already has the fix
    if "self.data_access.execute_update" in content:
        logger.info("SearchEngine.execute_update already fixed")
    else:
        # Apply the fix - replace self.execute_update with self.data_access.execute_update
        content = re.sub(
            r'self\.execute_update\(',
            'self.data_access.execute_update(',
            content
        )
        logger.info("Fixed SearchEngine.execute_update calls")

    # Add the missing _maybe_optimize_database method if not present
    if "_maybe_optimize_database" not in content:
        # Find a good spot to add the method - usually before the last class method
        optimize_method = """
    def _maybe_optimize_database(self):
        \"\"\"
        Optimize the database if needed based on operations count and time since last optimization.
        \"\"\"
        # This is a helper method to decide whether to run optimization
        # In a production system, you would track operations and time since last optimization

        # For now, we'll just skip optimization to avoid performance impact
        logger.debug("Database optimization skipped for now")

        # Uncomment to perform optimization periodically
        # import random
        # if random.random() < 0.05:  # 5% chance of optimization
        #     logger.info("Running scheduled database optimization")
        #     self.optimize_database()

        return
"""
        # Try to find the optimize_database method to add our method after it
        optimize_pattern = r'def optimize_database\(.*?\):'
        optimize_match = re.search(optimize_pattern, content, re.DOTALL)

        if optimize_match:
            # Find the end of the method
            method_start = optimize_match.start()
            method_text = content[method_start:]

            # Find the end of the method (indentation changes)
            lines = method_text.split('\n')
            method_end = 0
            indent = 0

            # Get the indentation of the method
            for i, line in enumerate(lines):
                if i == 0:  # First line contains the method definition
                    indent = len(line) - len(line.lstrip())
                    continue

                if line.strip() and len(line) - len(line.lstrip()) <= indent:
                    method_end = method_start + sum(len(l) + 1 for l in lines[:i])
                    break

            if method_end > 0:
                # Insert the new method after the optimize_database method
                content = content[:method_end] + optimize_method + content[method_end:]
            else:
                # If we couldn't find the method end, add to the end of the file
                content += optimize_method
        else:
            # If we couldn't find the optimize_database method, add to the end of the file
            content += optimize_method

        logger.info("Added missing _maybe_optimize_database method")
    else:
        logger.info("_maybe_optimize_database method already exists")

    # Option 2: Simply remove the call to _maybe_optimize_database
    # This approach is simpler but doesn't provide the method for future use
    # content = content.replace("self._maybe_optimize_database()", "# Optimization skipped")

    # Write the fixed content
    with open(search_engine_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Fixed SearchEngine")
    return True

def apply_concept_dedup_fix():
    """Fix the concept deduplication issues."""
    # Path to the concept_dedup.py file
    concept_dedup_path = "concept_dedup.py"

    if not os.path.exists(concept_dedup_path):
        logger.error(f"File not found: {concept_dedup_path}")
        return False

    # Read the current content
    with open(concept_dedup_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix deduplicate_concepts
    # Look for the function definition
    deduplicate_pattern = r'def deduplicate_concepts\(.*?Skip invalid concepts after normalization(.*?)continue\s+# STEP 1'

    # Replace with our fixed version
    fixed_deduplicate = r'def deduplicate_concepts\(.*?Skip invalid concepts after normalization\1continue\n\n        # STEP 1'

    # Find and replace the line that needs fixing
    content = re.sub(
        r'if normalized_text == "":',
        r'if normalized_text == "":',
        content
    )

    # 2. Fix apply_concept_deduplication
    # Look for the function definition and the problematic part
    apply_pattern = r'def apply_concept_deduplication\(.*?# First pass: filter out invalid concepts(.*?)filtered_key_concepts = \[c for c in key_concepts if self\.is_valid_concept\(c\.get\("text", ""\), language\)\]'

    # Replace with our fixed version that keeps original concepts if all would be filtered
    fixed_apply = r'def apply_concept_deduplication\(.*?# First pass: filter out invalid concepts\1filtered_key_concepts = [c for c in key_concepts if self.is_valid_concept(c.get("text", ""), language)]\n\n    # Important: If all were filtered, keep original concepts\n    if not filtered_key_concepts and key_concepts:\n        logger.warning(f"All {len(key_concepts)} concepts would be filtered out. Keeping original concepts.")\n        filtered_key_concepts = key_concepts'

    # Apply the replacement
    content = content.replace(
        "        # Skip invalid concepts after normalization\n        if not normalized_text:",
        "        # Skip invalid concepts after normalization - BUT only if normalization returned empty string\n        if normalized_text == \"\":"
    )

    if "# Important: If all were filtered, keep original concepts" not in content:
        # Add the code after filtered_key_concepts assignment
        content = content.replace(
            "    filtered_key_concepts = [c for c in key_concepts if self.is_valid_concept(c.get(\"text\", \"\"), language)]",
            "    filtered_key_concepts = [c for c in key_concepts if self.is_valid_concept(c.get(\"text\", \"\"), language)]\n\n    # Important: If all were filtered, keep original concepts\n    if not filtered_key_concepts and key_concepts:\n        logger.warning(f\"All {len(key_concepts)} concepts would be filtered out. Keeping original concepts.\")\n        filtered_key_concepts = key_concepts"
        )

    # Also add check after deduplication
    if "# If deduplication resulted in 0 concepts, keep the original filtered concepts" not in content:
        # Add the code after canonical_concepts assignment
        content = content.replace(
            "    canonical_concepts = self.deduplicate_concepts(filtered_key_concepts, language)",
            "    canonical_concepts = self.deduplicate_concepts(filtered_key_concepts, language)\n    \n    # If deduplication resulted in 0 concepts, keep the original filtered concepts\n    if not canonical_concepts and filtered_key_concepts:\n        logger.warning(\"Deduplication resulted in 0 concepts. Keeping original filtered concepts.\")\n        canonical_concepts = filtered_key_concepts"
        )

    # Write the fixed content
    with open(concept_dedup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Fixed concept_dedup.py")
    return True

def apply_data_pipeline_fix():
    """Fix the concept extraction in data_pipeline.py."""
    # Path to the data_pipeline.py file
    data_pipeline_path = "data_pipeline.py"

    if not os.path.exists(data_pipeline_path):
        logger.error(f"File not found: {data_pipeline_path}")
        return False

    # Read the current content
    with open(data_pipeline_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for the _extract_key_concepts_enhanced method
    # Add concept_id generation if not already present
    if "concept_id = hashlib.md5" not in content:
        # Add import if needed
        if "import hashlib" not in content:
            content = content.replace(
                "import math",
                "import math\nimport hashlib"
            )

        # Add concept_id generation before creating the ranked_concepts list
        content = content.replace(
            "    for concept_text, concept_data in filtered_candidates.items():",
            "    for concept_text, concept_data in filtered_candidates.items():\n        # Generate a deterministic concept_id based on text, domain and language\n        import hashlib\n        concept_id = hashlib.md5(f\"{concept_text.lower().strip()}:{domain}:{language}\".encode()).hexdigest()"
        )

        # Add concept_id to the ranked_concepts item
        content = content.replace(
            "        ranked_concepts.append({",
            "        ranked_concepts.append({\n            \"concept_id\": concept_id,"
        )

    # Add fallback extraction methods
    if "_extract_simple_terms" not in content:
        # Add the fallback methods at the end of the file
        content += """

def _extract_simple_terms(self, text: str, language: str) -> List[Tuple[str, int]]:
    \"\"\"
    Extract simple frequent terms as a last resort when other methods fail.

    Args:
        text: Text to analyze
        language: Language code

    Returns:
        List of (term, frequency) tuples
    \"\"\"
    # Get language-specific stopwords
    lang_code = language if language in self.stopwords else 'en'
    stop_words = self.stopwords.get(lang_code, set())

    # Try to tokenize
    try:
        words = word_tokenize(text.lower())
    except:
        words = text.lower().split()

    # Filter words
    filtered_words = [
        word for word in words
        if word not in stop_words
        and len(word) > 3
        and not word.isdigit()
        and not all(c in string.punctuation for c in word)
    ]

    # Count frequencies
    from collections import Counter
    word_counts = Counter(filtered_words)

    # Get most common words
    return word_counts.most_common(20)

def _extract_frequent_phrases(self, segments: List[Dict], language: str) -> Dict[str, int]:
    \"\"\"
    Extract frequent phrases as a fallback method when main extraction fails.

    Args:
        segments: Transcript segments
        language: Language code

    Returns:
        Dictionary mapping phrases to frequencies
    \"\"\"
    # Combine segment texts
    text = " ".join([segment.get("text", "") for segment in segments])

    # Get language-specific stopwords
    lang_code = language if language in self.stopwords else 'en'
    stop_words = self.stopwords.get(lang_code, set())

    # Try to tokenize and get n-grams
    try:
        words = word_tokenize(text.lower())
    except:
        words = text.lower().split()

    # Filter words
    filtered_words = [
        word for word in words
        if word not in stop_words
        and len(word) > 3
        and not word.isdigit()
        and not all(c in string.punctuation for c in word)
    ]

    # Extract bigrams and trigrams
    bigrams = []
    for i in range(len(filtered_words) - 1):
        bigrams.append(f"{filtered_words[i]} {filtered_words[i+1]}")

    trigrams = []
    for i in range(len(filtered_words) - 2):
        trigrams.append(f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}")

    # Count frequencies
    from collections import Counter
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    # Combine results - giving higher weight to trigrams
    phrases = {}
    for bigram, count in bigram_counts.most_common(10):
        phrases[bigram] = count

    for trigram, count in trigram_counts.most_common(10):
        phrases[trigram] = count * 1.5  # Weight trigrams higher

    return phrases
"""

    # Modify the min_score_threshold for Russian
    content = content.replace(
        "    min_score_threshold = 2.0 if language == 'ru' else 1.0",
        "    min_score_threshold = 0.5 if language == 'ru' else 1.0  # Lowered threshold for Russian"
    )

    # Add fallback extraction to _extract_key_concepts_enhanced
    if "No concepts extracted using primary methods" not in content:
        # Add fallback before applying filters
        content = content.replace(
            "    # 2. Apply filters to remove unlikely concepts",
            "    # If we have no candidates at all, extract simple frequent phrases as a fallback\n    if not candidates:\n        logger.warning(f\"No concepts extracted using primary methods. Using fallback extraction for {language}.\")\n        top_phrases = self._extract_frequent_phrases(segments, language)\n        for phrase, count in top_phrases.items():\n            candidates[phrase] = {\n                \"text\": phrase,\n                \"frequency\": count,\n                \"ngram_type\": \"frequent_phrase\",\n                \"score\": count * 0.8,  # Lower weight for simple frequent phrases\n                \"source\": \"fallback_extraction\"\n            }\n\n    # 2. Apply filters to remove unlikely concepts"
        )

    # Add fallback for empty ranked_concepts
    if "Make sure we have at least some concepts" not in content:
        # Add check after sorting ranked_concepts
        content = content.replace(
            "    # Take top concepts with a adaptive limit based on content length",
            "    # Make sure we have at least some concepts\n    if not ranked_concepts:\n        # If no concepts at all, try to extract at least some basic terms\n        simple_terms = self._extract_simple_terms(combined_text, language)\n        for term, freq in simple_terms:\n            import hashlib\n            concept_id = hashlib.md5(f\"{term.lower().strip()}:{domain}:{language}\".encode()).hexdigest()\n            \n            ranked_concepts.append({\n                \"text\": term,\n                \"concept_id\": concept_id,\n                \"frequency\": freq,\n                \"domain\": domain,\n                \"theoretical\": True,  # Default to theoretical\n                \"concept_class\": \"theoretical\",\n                \"ngram_type\": \"simple_term\",\n                \"pattern_match\": False,\n                \"definitional\": False,\n                \"collocation\": False,\n                \"score\": freq * 0.5,\n                \"source\": \"simple_term_extraction\",\n                \"language\": language\n            })\n        \n        logger.warning(f\"Using simple term extraction as last resort. Found {len(ranked_concepts)} terms.\")\n        max_concepts = min(10, len(ranked_concepts))\n    \n    # Take top concepts with a adaptive limit based on content length"
        )

    # Write the fixed content
    with open(data_pipeline_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info("Fixed data_pipeline.py")
    return True

def main():
    """Apply all fixes."""
    try:
        logger.info("Starting fixes for Lecture Video Content Indexer")

        # Apply fixes
        search_engine_fixed = apply_search_engine_fix()
        concept_dedup_fixed = apply_concept_dedup_fix()
        data_pipeline_fixed = apply_data_pipeline_fix()

        # Report results
        if search_engine_fixed and concept_dedup_fixed and data_pipeline_fixed:
            logger.info("All fixes applied successfully!")
            logger.info("\nNow you can run:\n  python demo.py --playlist <your-playlist-url> --max-videos 1")
            return 0
        else:
            logger.error("Some fixes failed to apply")
            return 1

    except Exception as e:
        logger.error(f"Error applying fixes: {e}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
