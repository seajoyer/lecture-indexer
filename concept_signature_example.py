#!/usr/bin/env python3
"""
Example demonstrating the Concept Signature functionality of the Lecture Video Content Indexer.
Shows how to use the concept signatures to analyze educational content.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from concept_signature_generator import (
        ConceptSignature,
        RelationshipGraph,
        MLCSProcessor,
        get_concept_signature_generator
    )
    from data_pipeline import DataPipeline
    from search_engine import SearchEngine
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you have all required modules installed.")
    sys.exit(1)

def main():
    """Main function running the concept signature example."""
    parser = argparse.ArgumentParser(description='Concept Signature Example')
    parser.add_argument('video_urls', nargs='*', help='YouTube video URLs to process')
    parser.add_argument('--demo', action='store_true', help='Run a demo with sample data')
    parser.add_argument('--output', default='concept_graph.json',
                      help='Output file for concept relationship graph')
    parser.add_argument('--api-key', help='YouTube API key')
    parser.add_argument('--load-graph', help='Load existing concept graph file')

    args = parser.parse_args()

    # Get YouTube API key
    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key and not args.demo and not args.load_graph:
        print("No YouTube API key provided. Please set YOUTUBE_API_KEY environment variable or use --api-key.")
        sys.exit(1)

    # Create config
    config = {
        "youtube_api_key": api_key,
        "output_dir": "data/processed",
        "index_dir": "data/index"
    }

    # Create necessary directories
    for directory in ["data/processed", "data/index"]:
        os.makedirs(directory, exist_ok=True)

    # Load existing graph if specified
    if args.load_graph:
        print(f"Loading concept graph from {args.load_graph}")
        graph = RelationshipGraph.load_from_json(args.load_graph)
        print(f"Loaded graph with {len(graph.concepts)} concepts and "
              f"{sum(len(targets) for targets in graph.adjacency_list.values())} relationships")

        # Display graph statistics
        display_graph_statistics(graph)

        # Export to specified output
        graph.save_to_json(args.output)
        print(f"Saved graph to {args.output}")

        return

    # Run demo if requested
    if args.demo:
        run_demo(config, args.output)
        return

    # Process video URLs
    if not args.video_urls:
        print("No video URLs provided. Use --demo for a demonstration or provide YouTube URLs.")
        sys.exit(1)

    # Create components
    data_pipeline = DataPipeline(config)
    search_engine = SearchEngine(config)

    # Get concept signature generator
    concept_generator = get_concept_signature_generator(config)

    # Process videos
    relationship_graph = RelationshipGraph()

    for i, url in enumerate(args.video_urls, 1):
        print(f"Processing video {i}/{len(args.video_urls)}: {url}")
        try:
            # Process video
            result = data_pipeline.process_video(url)

            if result.get("status") == "completed":
                print("Video processed successfully!")

                # Index content
                search_engine.index_content(result)

                # Extract video ID and concepts
                video_id = result.get("video_id", "")
                domain_features = result.get("domain_features", {})
                key_concepts = domain_features.get("key_concepts", [])
                concept_signatures = domain_features.get("concept_signatures", [])

                print(f"Extracted {len(key_concepts)} concepts and {len(concept_signatures)} signatures")

                # Add to relationship graph
                for signature_data in concept_signatures:
                    # Create signature object
                    signature = ConceptSignature.from_dict(signature_data)
                    relationship_graph.add_concept(signature)
            else:
                print(f"Error processing video: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Calculate hierarchy scores
    relationship_graph.calculate_all_hierarchy_scores()

    # Save the graph
    relationship_graph.save_to_json(args.output)
    print(f"Saved concept relationship graph to {args.output}")

    # Display graph statistics
    display_graph_statistics(relationship_graph)

def run_demo(config: Dict[str, Any], output_file: str) -> None:
    """
    Run a demonstration of concept signatures with sample data.

    Args:
        config: Configuration dictionary
        output_file: Output file path
    """
    print("Running concept signature demonstration...")

    # Sample transcript segments
    segments = [
        {
            "id": "segment1",
            "video_id": "demo_video",
            "start_time": 0.0,
            "end_time": 30.0,
            "text": "Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables.",
            "content_type": "theoretical"
        },
        {
            "id": "segment2",
            "video_id": "demo_video",
            "start_time": 30.0,
            "end_time": 60.0,
            "text": "In simple linear regression, we have one independent variable and the relationship is modeled using a straight line.",
            "content_type": "theoretical"
        },
        {
            "id": "segment3",
            "video_id": "demo_video",
            "start_time": 60.0,
            "end_time": 90.0,
            "text": "Let's see how to implement linear regression using Python. First, import the necessary libraries like numpy and scikit-learn.",
            "content_type": "practical"
        },
        {
            "id": "segment4",
            "video_id": "demo_video",
            "start_time": 90.0,
            "end_time": 120.0,
            "text": "Machine learning models can be used to predict outcomes based on input features. Linear regression is one of the simplest models.",
            "content_type": "theoretical"
        },
        {
            "id": "segment5",
            "video_id": "demo_video",
            "start_time": 120.0,
            "end_time": 150.0,
            "text": "Now let's create a dataset for our regression example. We'll generate some random data with a linear relationship plus some noise.",
            "content_type": "practical"
        }
    ]

    # Sample concepts
    concepts = [
        {
            "concept_id": "linear_regression",
            "text": "linear regression",
            "domain": "mathematics",
            "concept_class": "theoretical",
            "language": "en"
        },
        {
            "concept_id": "machine_learning",
            "text": "machine learning",
            "domain": "programming",
            "concept_class": "theoretical",
            "language": "en"
        },
        {
            "concept_id": "python_implementation",
            "text": "Python implementation",
            "domain": "programming",
            "concept_class": "practical",
            "language": "en"
        },
        {
            "concept_id": "dataset_creation",
            "text": "dataset creation",
            "domain": "programming",
            "concept_class": "practical",
            "language": "en"
        }
    ]

    # Create MLCS processor
    processor = MLCSProcessor()

    # Generate concept signatures
    signatures = processor.generate_concept_signatures(concepts, segments)
    print(f"Generated {len(signatures)} concept signatures")

    # Create relationship graph
    graph = RelationshipGraph()

    # Add concepts to graph
    for signature in signatures:
        graph.add_concept(signature)

    # Identify relationships
    print("Identifying concept relationships...")

    # Add some relationships
    graph.add_relationship("linear_regression", "machine_learning", "prerequisite", 0.8)
    graph.add_relationship("machine_learning", "python_implementation", "prerequisite", 0.7)
    graph.add_relationship("python_implementation", "dataset_creation", "related", 0.9)

    # Calculate hierarchy scores
    graph.calculate_all_hierarchy_scores()

    # Save the graph
    graph.save_to_json(output_file)
    print(f"Saved concept relationship graph to {output_file}")

    # Generate a learning path
    print("\nGenerating learning path...")
    path = graph.generate_learning_path(
        ["dataset_creation", "linear_regression"],
        theory_practice_ratio=0.6
    )

    print("Learning path:")
    for i, concept_id in enumerate(path, 1):
        concept = graph.concepts[concept_id]
        print(f"{i}. {concept.text} ({concept.concept_class})")

    # Display graph statistics
    display_graph_statistics(graph)

def display_graph_statistics(graph: RelationshipGraph) -> None:
    """
    Display statistics about a concept relationship graph.

    Args:
        graph: RelationshipGraph instance
    """
    concept_count = len(graph.concepts)
    if concept_count == 0:
        print("Graph is empty.")
        return

    # Count relationships
    rel_count = sum(len(targets) for targets in graph.adjacency_list.values())

    print("\n=== Concept Graph Statistics ===")
    print(f"Total concepts: {concept_count}")
    print(f"Total relationships: {rel_count}")
    print(f"Average relationships per concept: {rel_count / concept_count:.2f}")

    # Count by domain
    domain_counts = {}
    for concept in graph.concepts.values():
        domain = concept.domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print("\nConcepts by domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {count} ({count / concept_count * 100:.1f}%)")

    # Count by concept class
    theoretical = sum(1 for c in graph.concepts.values() if c.concept_class == "theoretical")
    practical = sum(1 for c in graph.concepts.values() if c.concept_class == "practical")

    print("\nConcepts by class:")
    print(f"  Theoretical: {theoretical} ({theoretical / concept_count * 100:.1f}%)")
    print(f"  Practical: {practical} ({practical / concept_count * 100:.1f}%)")

    # Count relationship types
    rel_types = {}
    for attrs in graph.edge_attributes.values():
        rel_type = attrs.get("type", "unknown")
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

    print("\nRelationships by type:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel_type}: {count} ({count / rel_count * 100:.1f}%)")

if __name__ == "__main__":
    main()
