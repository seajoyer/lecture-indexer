"""
Test script for enhanced concept extraction with multilingual support.
Tests the improved concept extraction on both English and Russian quantum physics content.
"""

import logging
import json
from typing import List, Dict, Any

# Import our enhanced modules
from unified_concept_extractor import UnifiedConceptExtractor
from transcript_processor import TranscriptProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_concept_extraction(text: str, language: str, domain: str = "physics") -> Dict[str, Any]:
    """
    Test concept extraction from text with detailed evaluation.

    Args:
        text: Source text
        language: Language code ('en' or 'ru')
        domain: Content domain

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing concept extraction for language: {language}")

    # Split text into simulated segments
    segments = []

    # Create simple segments from paragraphs
    paragraphs = text.split('\n\n')
    for i, p in enumerate(paragraphs):
        if p.strip():
            segments.append({
                "id": f"segment_{i}",
                "start_time": i * 10.0,
                "end_time": (i + 1) * 10.0,
                "text": p,
                "language": language
            })

    # Create transcript processor and unified concept extractor
    transcript_processor = TranscriptProcessor()
    concept_extractor = UnifiedConceptExtractor(language=language)

    # Create metadata
    metadata = {
        "video_id": "test_video",
        "domain": domain,
        "title": "Test Video",
        "language": language
    }

    # Process transcript
    processed_transcript = transcript_processor.process_transcript(segments, metadata)

    # Extract concepts
    domain_features = concept_extractor.extract_concepts_from_transcript(processed_transcript)

    # Evaluate results
    return {
        "language": language,
        "domain": domain,
        "segment_count": len(segments),
        "processed_segment_count": len(processed_transcript.get("segments", [])),
        "theoretical_concepts": domain_features.get("theoretical_concepts", []),
        "practical_concepts": domain_features.get("practical_concepts", []),
        "theoretical_count": len(domain_features.get("theoretical_concepts", [])),
        "practical_count": len(domain_features.get("practical_concepts", [])),
        "total_concepts": len(domain_features.get("theoretical_concepts", [])) + len(domain_features.get("practical_concepts", []))
    }

def print_concept_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the extracted concepts.

    Args:
        results: Results dictionary from test_concept_extraction
    """
    print(f"\n===== Concept Extraction Results ({results['language']}) =====")
    print(f"Domain: {results['domain']}")
    print(f"Total Concepts: {results['total_concepts']}")
    print(f"  - Theoretical: {results['theoretical_count']}")
    print(f"  - Practical: {results['practical_count']}")

    # Display top concepts of each type
    max_display = 10

    print("\nTheoretical Concepts:")
    for i, concept in enumerate(results['theoretical_concepts'][:max_display]):
        confidence = concept.get("classification_confidence", 0.0)
        is_educational = concept.get("is_educational", False)
        educational_marker = "✓" if is_educational else "✗"
        print(f"  {i+1}. {concept['text']} (ID: {concept['concept_id'][:8]}) [Educational: {educational_marker}]")

    print("\nPractical Concepts:")
    for i, concept in enumerate(results['practical_concepts'][:max_display]):
        confidence = concept.get("classification_confidence", 0.0)
        is_educational = concept.get("is_educational", False)
        educational_marker = "✓" if is_educational else "✗"
        print(f"  {i+1}. {concept['text']} (ID: {concept['concept_id'][:8]}) [Educational: {educational_marker}]")

def main():
    """Run concept extraction tests."""
    # Test with English quantum physics text
    english_test_text = """
Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.

Classical physics, the collection of theories that existed before the advent of quantum mechanics, describes many aspects of nature at an ordinary (macroscopic) scale, but it is not sufficient for describing them at small (atomic and subatomic) scales. Most theories in classical physics can be derived from quantum mechanics as an approximation valid at large (macroscopic) scale.

Quantum mechanics differs from classical physics in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to discrete values (quantization); objects have characteristics of both particles and waves (wave–particle duality); and there are limits to how accurately the value of a physical quantity can be predicted prior to its measurement, given a complete set of initial conditions (the uncertainty principle).

Let's explore the concept of wave functions in quantum mechanics. The wave function is a mathematical function used in quantum mechanics to describe the quantum state of a system. It is a complex-valued probability amplitude, and the probabilities for the possible results of measurements made on the system can be derived from it.

For example, if we consider a single non-relativistic particle in one dimension, the wave function is a function of position and time: Ψ(x, t). The square of the absolute value of the wave function, |Ψ(x, t)|², gives the probability density of finding the particle at position x at time t.
    """

    # Test with Russian quantum physics text
    russian_test_text = """
Квантовая механика — это фундаментальная теория в физике, которая описывает физические свойства природы на масштабе атомов и субатомных частиц. Она является основой всей квантовой физики, включая квантовую химию, квантовую теорию поля, квантовые технологии и квантовую информатику.

Классическая физика, совокупность теорий, существовавших до появления квантовой механики, описывает многие аспекты природы в обычном (макроскопическом) масштабе, но она недостаточна для их описания в малых (атомных и субатомных) масштабах. Большинство теорий в классической физике могут быть выведены из квантовой механики как приближение, справедливое в большом (макроскопическом) масштабе.

Квантовая механика отличается от классической физики тем, что энергия, импульс, момент импульса и другие величины связанной системы ограничены дискретными значениями (квантование); объекты имеют характеристики как частиц, так и волн (волново-корпускулярный дуализм); и существуют пределы того, насколько точно можно предсказать значение физической величины до ее измерения, при заданном полном наборе начальных условий (принцип неопределенности).

Давайте рассмотрим понятие волновой функции в квантовой механике. Волновая функция — это математическая функция, используемая в квантовой механике для описания квантового состояния системы. Это комплекснозначная амплитуда вероятности, и вероятности для возможных результатов измерений, проводимых на системе, могут быть получены из нее.

Например, если мы рассматриваем одну нерелятивистскую частицу в одном измерении, волновая функция является функцией положения и времени: Ψ(x, t). Квадрат модуля волновой функции, |Ψ(x, t)|², дает плотность вероятности найти частицу в положении x в момент времени t.
    """

    # Run tests
    english_results = test_concept_extraction(english_test_text, "en", "physics")
    print_concept_summary(english_results)

    russian_results = test_concept_extraction(russian_test_text, "ru", "physics")
    print_concept_summary(russian_results)

    # Compare results
    print("\n===== Comparison =====")
    print(f"English concepts: {english_results['total_concepts']}")
    print(f"Russian concepts: {russian_results['total_concepts']}")

if __name__ == "__main__":
    main()
