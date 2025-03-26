#!/usr/bin/env python3

"""
Theory-Practice Visualization Script - Creates visualizations from the batch processor output.

This script reads the JSON output files from the YouTube Lecture Batch Processor
and generates visual representations of the theory vs practice content distribution.

Usage:
    python visualize_results.py output_dir [--format {png,pdf,svg}]

Requirements:
    - matplotlib
    - numpy
"""

import os
import sys
import json
import glob
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("visualizer")

def load_results(output_dir: str) -> List[Dict[str, Any]]:
    """
    Load all batch processing result files from a directory.

    Args:
        output_dir: Directory containing JSON result files

    Returns:
        List of parsed result objects
    """
    results = []

    # Load individual video results
    video_files = glob.glob(os.path.join(output_dir, "*.json"))

    # Sort files by modification time (newest first)
    video_files.sort(key=os.path.getmtime, reverse=True)

    for file_path in video_files:
        try:
            # Skip batch summary files
            if "batch_" in os.path.basename(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get("status") == "completed":
                results.append(data)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(results)} video results from {output_dir}")
    return results

def create_theory_practice_pie_charts(results: List[Dict[str, Any]], output_dir: str, format: str = 'png'):
    """
    Create pie charts showing theory vs practice distribution for each domain.

    Args:
        results: List of video analysis results
        output_dir: Directory to save charts
        format: Output file format (png, pdf, svg)
    """
    if not results:
        logger.warning("No results to visualize")
        return

    # Group results by domain
    domains = {}
    for result in results:
        domain = result.get('metadata', {}).get('domain', 'unknown')
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(result)

    # Create charts directory
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Create aggregate pie chart for all domains
    labels = ['Theoretical', 'Practical', 'Mixed']
    colors = ['#4285F4', '#34A853', '#FBBC05']  # Blue, Green, Yellow

    domain_theory_ratios = {}

    # Create individual pie charts for each domain
    for domain, domain_results in domains.items():
        if domain == 'unknown':
            continue

        # Calculate total segments by type
        theoretical = 0
        practical = 0
        mixed = 0

        for result in domain_results:
            tp_results = result.get('theory_practice_results', {})
            theoretical += tp_results.get('theoretical_segments', 0)
            practical += tp_results.get('practical_segments', 0)
            mixed += tp_results.get('mixed_segments', 0)

        # Skip if no segments
        total = theoretical + practical + mixed
        if total == 0:
            continue

        # Calculate percentages
        theory_percent = (theoretical / total) * 100
        practice_percent = (practical / total) * 100
        mixed_percent = (mixed / total) * 100

        # Store for aggregate chart
        domain_theory_ratios[domain] = theory_percent / 100

        # Create pie chart
        plt.figure(figsize=(10, 7))
        plt.pie(
            [theoretical, practical, mixed],
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0, 0)  # Explode the theoretical slice
        )
        plt.axis('equal')
        plt.title(f'Theory vs Practice Content Distribution - {domain.capitalize()}', fontsize=16)

        # Add subtitle with video count
        plt.figtext(
            0.5, 0.01,
            f'Based on {len(domain_results)} videos with {total} segments',
            ha='center',
            fontsize=12
        )

        # Save the figure
        output_path = os.path.join(charts_dir, f'theory_practice_pie_{domain}.{format}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Created pie chart for {domain} domain: {output_path}")

    # Create domain comparison bar chart
    if domain_theory_ratios:
        domains_to_show = [d for d in domain_theory_ratios.keys() if d != 'unknown']

        if domains_to_show:
            plt.figure(figsize=(12, 8))

            domain_names = [d.capitalize() for d in domains_to_show]
            theory_values = [domain_theory_ratios[d] * 100 for d in domains_to_show]
            practice_values = [100 - v for v in theory_values]

            x = np.arange(len(domain_names))
            width = 0.35

            plt.bar(x, theory_values, width, label='Theoretical', color='#4285F4')
            plt.bar(x, practice_values, width, bottom=theory_values, label='Practical', color='#34A853')

            plt.xlabel('Domain', fontsize=14)
            plt.ylabel('Percentage', fontsize=14)
            plt.title('Theory vs Practice Ratio by Domain', fontsize=16)
            plt.xticks(x, domain_names)
            plt.yticks(np.arange(0, 101, 10))
            plt.legend()

            # Add data labels
            for i, v in enumerate(theory_values):
                plt.text(i, v/2, f"{v:.1f}%", ha='center', fontsize=12, color='white')
                plt.text(i, v + (practice_values[i]/2), f"{practice_values[i]:.1f}%", ha='center', fontsize=12, color='white')

            # Save the figure
            output_path = os.path.join(charts_dir, f'domain_comparison.{format}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            logger.info(f"Created domain comparison chart: {output_path}")

def create_transitions_chart(results: List[Dict[str, Any]], output_dir: str, format: str = 'png'):
    """
    Create charts showing theory-to-practice and practice-to-theory transitions.

    Args:
        results: List of video analysis results
        output_dir: Directory to save charts
        format: Output file format (png, pdf, svg)
    """
    if not results:
        return

    # Group results by domain
    domain_transitions = {}

    for result in results:
        domain = result.get('metadata', {}).get('domain', 'unknown')
        if domain not in domain_transitions:
            domain_transitions[domain] = {
                'theory_to_practice': 0,
                'practice_to_theory': 0,
                'video_count': 0
            }

        domain_transitions[domain]['video_count'] += 1

        # Count transitions
        patterns = result.get('theory_practice_patterns', {})
        domain_transitions[domain]['theory_to_practice'] += len(patterns.get('theory_to_practice_sequences', []))
        domain_transitions[domain]['practice_to_theory'] += len(patterns.get('practice_to_theory_sequences', []))

    # Create charts directory
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Skip unknown domain
    if 'unknown' in domain_transitions:
        del domain_transitions['unknown']

    if not domain_transitions:
        return

    # Create transitions comparison chart
    domains = list(domain_transitions.keys())
    t2p_counts = [domain_transitions[d]['theory_to_practice'] for d in domains]
    p2t_counts = [domain_transitions[d]['practice_to_theory'] for d in domains]
    video_counts = [domain_transitions[d]['video_count'] for d in domains]

    # Normalize by video count
    t2p_per_video = [t / max(1, v) for t, v in zip(t2p_counts, video_counts)]
    p2t_per_video = [p / max(1, v) for p, v in zip(p2t_counts, video_counts)]

    # Create bar chart
    plt.figure(figsize=(12, 8))

    x = np.arange(len(domains))
    width = 0.35

    plt.bar(x - width/2, t2p_per_video, width, label='Theory → Practice', color='#4285F4')
    plt.bar(x + width/2, p2t_per_video, width, label='Practice → Theory', color='#34A853')

    plt.xlabel('Domain', fontsize=14)
    plt.ylabel('Transitions per Video', fontsize=14)
    plt.title('Theory-Practice Transitions by Domain', fontsize=16)
    plt.xticks(x, [d.capitalize() for d in domains])
    plt.legend()

    # Add data labels
    for i, v in enumerate(t2p_per_video):
        plt.text(i - width/2, v + 0.1, f"{v:.1f}", ha='center')
    for i, v in enumerate(p2t_per_video):
        plt.text(i + width/2, v + 0.1, f"{v:.1f}", ha='center')

    # Save the figure
    output_path = os.path.join(charts_dir, f'transitions_comparison.{format}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    logger.info(f"Created transitions comparison chart: {output_path}")

def create_concept_distribution(results: List[Dict[str, Any]], output_dir: str, format: str = 'png'):
    """
    Create charts showing distribution of theoretical vs practical concepts.

    Args:
        results: List of video analysis results
        output_dir: Directory to save charts
        format: Output file format (png, pdf, svg)
    """
    if not results:
        return

    # Group concepts by domain
    domain_concepts = {}

    for result in results:
        domain = result.get('metadata', {}).get('domain', 'unknown')
        if domain not in domain_concepts:
            domain_concepts[domain] = {
                'theoretical': [],
                'practical': []
            }

        # Add concepts
        domain_features = result.get('domain_features', {})
        key_concepts = domain_features.get('key_concepts', [])

        for concept in key_concepts:
            concept_type = 'theoretical' if concept.get('theoretical', False) else 'practical'
            concept_text = concept.get('text', '').lower()

            # Check if concept already exists
            found = False
            for existing in domain_concepts[domain][concept_type]:
                if existing['text'] == concept_text:
                    existing['frequency'] += concept.get('frequency', 1)
                    existing['count'] += 1
                    found = True
                    break

            if not found:
                domain_concepts[domain][concept_type].append({
                    'text': concept_text,
                    'frequency': concept.get('frequency', 1),
                    'count': 1
                })

    # Create charts directory
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Skip unknown domain
    if 'unknown' in domain_concepts:
        del domain_concepts['unknown']

    if not domain_concepts:
        return

    # Create concept distribution charts for each domain
    for domain, concepts in domain_concepts.items():
        # Sort concepts by frequency
        theoretical = sorted(concepts['theoretical'], key=lambda x: x['frequency'], reverse=True)[:15]
        practical = sorted(concepts['practical'], key=lambda x: x['frequency'], reverse=True)[:15]

        if not theoretical and not practical:
            continue

        # Create horizontal bar chart
        plt.figure(figsize=(14, 10))

        # Calculate maximum frequency for scaling
        max_freq = max(
            max([c['frequency'] for c in theoretical], default=0),
            max([c['frequency'] for c in practical], default=0)
        )

        # Create theoretical concepts subplot
        plt.subplot(1, 2, 1)
        theo_texts = [c['text'] for c in theoretical]
        theo_freqs = [c['frequency'] for c in theoretical]

        y_pos = np.arange(len(theo_texts))
        plt.barh(y_pos, theo_freqs, color='#4285F4')
        plt.yticks(y_pos, theo_texts)
        plt.xlim(0, max_freq * 1.1)  # Add some padding
        plt.title('Theoretical Concepts')
        plt.tight_layout()

        # Create practical concepts subplot
        plt.subplot(1, 2, 2)
        prac_texts = [c['text'] for c in practical]
        prac_freqs = [c['frequency'] for c in practical]

        y_pos = np.arange(len(prac_texts))
        plt.barh(y_pos, prac_freqs, color='#34A853')
        plt.yticks(y_pos, prac_texts)
        plt.xlim(0, max_freq * 1.1)  # Add some padding
        plt.title('Practical Concepts')
        plt.tight_layout()

        # Add overall title
        plt.suptitle(f'Top Concepts in {domain.capitalize()} Videos', fontsize=16)
        plt.subplots_adjust(top=0.9)

        # Save the figure
        output_path = os.path.join(charts_dir, f'concept_distribution_{domain}.{format}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        logger.info(f"Created concept distribution chart for {domain}: {output_path}")

def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Theory-Practice Visualization Script')
    parser.add_argument('output_dir', help='Directory containing batch processor output files')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png', help='Output file format')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        # Check if output directory exists
        if not os.path.isdir(args.output_dir):
            print(f"Error: Directory not found: {args.output_dir}")
            return 1

        # Load results
        results = load_results(args.output_dir)

        if not results:
            print(f"No valid result files found in {args.output_dir}")
            return 1

        # Create visualizations
        create_theory_practice_pie_charts(results, args.output_dir, args.format)
        create_transitions_chart(results, args.output_dir, args.format)
        create_concept_distribution(results, args.output_dir, args.format)

        print(f"Created visualizations in {args.output_dir}/charts")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
