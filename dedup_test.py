"""
Test script for the unified concept deduplication approach.
Demonstrates how the system handles duplicates across concept categories.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the deduplication module
from concept_dedup import ConceptDedupExtension, apply_concept_deduplication

def create_test_data():
    """Create test data with cross-category duplicates."""
    # Russian physics concepts
    concepts = {
        # Key concepts with some duplicates
        "key_concepts": [
            {
                "text": "уравнение шредингера",
                "concept_id": hashlib.md5("уравнение шредингера:physics:ru".encode()).hexdigest(),
                "frequency": 7,
                "score": 5.2,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "волновая функция",
                "concept_id": hashlib.md5("волновая функция:physics:ru".encode()).hexdigest(),
                "frequency": 8,
                "score": 6.1,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "собственное значение",
                "concept_id": hashlib.md5("собственное значение:physics:ru".encode()).hexdigest(),
                "frequency": 6,
                "score": 5.5,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "квантовая механика",
                "concept_id": hashlib.md5("квантовая механика:physics:ru".encode()).hexdigest(),
                "frequency": 5,
                "score": 5.3,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            }
        ],

        # Theoretical concepts with duplicates from key_concepts
        "theoretical_concepts": [
            {
                "text": "уравнения шредингера",  # Variant of "уравнение шредингера"
                "concept_id": hashlib.md5("уравнения шредингера:physics:ru".encode()).hexdigest(),
                "frequency": 4,
                "score": 4.8,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "волновую функцию",  # Variant of "волновая функция"
                "concept_id": hashlib.md5("волновую функцию:physics:ru".encode()).hexdigest(),
                "frequency": 3,
                "score": 4.2,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "гамильтониан",  # Unique to theoretical concepts
                "concept_id": hashlib.md5("гамильтониан:physics:ru".encode()).hexdigest(),
                "frequency": 5,
                "score": 5.0,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            },
            {
                "text": "собственные значения",  # Variant of "собственное значение"
                "concept_id": hashlib.md5("собственные значения:physics:ru".encode()).hexdigest(),
                "frequency": 4,
                "score": 4.9,
                "domain": "physics",
                "language": "ru",
                "concept_class": "theoretical"
            }
        ],

        # Practical concepts with some duplicates
        "practical_concepts": [
            {
                "text": "уравнение Шредингер",  # Another variant of "уравнение шредингера"
                "concept_id": hashlib.md5("уравнение Шредингер:physics:ru".encode()).hexdigest(),
                "frequency": 2,
                "score": 3.5,
                "domain": "physics",
                "language": "ru",
                "concept_class": "practical"  # Note: Different class for testing
            },
            {
                "text": "эксперимент",  # Unique to practical concepts
                "concept_id": hashlib.md5("эксперимент:physics:ru".encode()).hexdigest(),
                "frequency": 6,
                "score": 5.2,
                "domain": "physics",
                "language": "ru",
                "concept_class": "practical"
            },
            {
                "text": "квантовый механика",  # Variant of "квантовая механика"
                "concept_id": hashlib.md5("квантовый механика:physics:ru".encode()).hexdigest(),
                "frequency": 3,
                "score": 4.1,
                "domain": "physics",
                "language": "ru",
                "concept_class": "practical"
            }
        ]
    }

    return concepts

def test_unified_deduplication():
    """Test the unified deduplication approach."""
    # Create test data
    concept_categories = create_test_data()

    # Count concepts in each category
    key_count = len(concept_categories["key_concepts"])
    theoretical_count = len(concept_categories["theoretical_concepts"])
    practical_count = len(concept_categories["practical_concepts"])
    total_count = key_count + theoretical_count + practical_count

    logger.info(f"Created test data with {total_count} concepts across categories:")
    logger.info(f"  Key concepts: {key_count}")
    logger.info(f"  Theoretical concepts: {theoretical_count}")
    logger.info(f"  Practical concepts: {practical_count}")

    # Create mock processing result with test data
    mock_result = {
        "job_id": "test_job",
        "status": "completed",
        "video_id": "test_video",
        "transcript": {
            "language": "ru",
            "segments": []
        },
        "domain_features": {
            "domain": "physics",
            "key_concepts": concept_categories["key_concepts"],
            "theoretical_concepts": concept_categories["theoretical_concepts"],
            "practical_concepts": concept_categories["practical_concepts"]
        }
    }

    # Apply unified deduplication
    logger.info("Applying unified concept deduplication...")
    start_time = time.time()

    deduplicated_result = apply_concept_deduplication(mock_result, "ru")

    dedup_time = time.time() - start_time

    # Extract deduplicated concept lists
    deduplicated_key = deduplicated_result["domain_features"]["key_concepts"]
    deduplicated_theoretical = deduplicated_result["domain_features"]["theoretical_concepts"]
    deduplicated_practical = deduplicated_result["domain_features"]["practical_concepts"]

    # Check deduplication results
    logger.info(f"Deduplication completed in {dedup_time:.4f} seconds")
    logger.info(f"Deduplication stats:")

    if "deduplication_stats" in deduplicated_result:
        stats = deduplicated_result["deduplication_stats"]
        logger.info(f"  Original total: {stats['original_total']} concepts")
        logger.info(f"  Deduplicated total: {stats['deduplicated_total']} concepts")
        logger.info(f"  Reduction: {stats['reduction_percentage']}%")

        logger.info(f"\nCategory-specific results:")
        logger.info(f"  Key concepts: {stats['key_concepts_original']} → {stats['key_concepts_deduplicated']}")
        logger.info(f"  Theoretical concepts: {stats['theoretical_concepts_original']} → {stats['theoretical_concepts_deduplicated']}")
        logger.info(f"  Practical concepts: {stats['practical_concepts_original']} → {stats['practical_concepts_deduplicated']}")
    else:
        logger.info(f"  Key concepts: {key_count} → {len(deduplicated_key)}")
        logger.info(f"  Theoretical concepts: {theoretical_count} → {len(deduplicated_theoretical)}")
        logger.info(f"  Practical concepts: {practical_count} → {len(deduplicated_practical)}")

    # Analyze deduplicated concepts
    logger.info("\nDeduplicated key concepts:")
    for i, concept in enumerate(deduplicated_key):
        variants = concept.get("variant_texts", [])
        variant_str = f" (variants: {', '.join(variants)})" if variants else ""
        logger.info(f"  {i+1}. {concept['text']}{variant_str}")

    logger.info("\nDeduplicated theoretical concepts:")
    for i, concept in enumerate(deduplicated_theoretical):
        variants = concept.get("variant_texts", [])
        variant_str = f" (variants: {', '.join(variants)})" if variants else ""
        logger.info(f"  {i+1}. {concept['text']}{variant_str}")

    logger.info("\nDeduplicated practical concepts:")
    for i, concept in enumerate(deduplicated_practical):
        variants = concept.get("variant_texts", [])
        variant_str = f" (variants: {', '.join(variants)})" if variants else ""
        logger.info(f"  {i+1}. {concept['text']}{variant_str}")

    # Verify no duplicates across categories
    all_canonical_texts = set()
    duplicates_found = []

    for concept in deduplicated_key + deduplicated_theoretical + deduplicated_practical:
        text = concept.get("text", "").lower()
        if text in all_canonical_texts:
            duplicates_found.append(text)
        else:
            all_canonical_texts.add(text)

    if duplicates_found:
        logger.warning(f"Found {len(duplicates_found)} duplicates across categories: {duplicates_found}")
    else:
        logger.info("\nSuccess: No duplicates found across categories!")

    return deduplicated_result

if __name__ == "__main__":
    logger.info("Running unified deduplication test")
    test_unified_deduplication()
    logger.info("Test completed")
