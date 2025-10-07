#!/usr/bin/env python3
"""Test script to extract and display coreference examples from CorefUD files."""

import sys
from corefud_qa import extract_mentions_from_conllu, build_pronoun_qa, load_english_files

def show_examples(corefud_root: str, max_files: int = 1, max_examples: int = 5):
    """Extract and display coreference examples."""
    files = load_english_files(corefud_root)

    if not files:
        print(f"No English CorefUD files found in {corefud_root}")
        return

    print(f"Found {len(files)} English CorefUD files:")
    for f in files:
        print(f"  - {f}")
    print()

    total_items = 0
    for i, path in enumerate(files[:max_files]):
        print(f"\n{'='*80}")
        print(f"Processing: {path}")
        print('='*80)

        # Extract mentions and clusters
        sent_texts, sentences, clusters = extract_mentions_from_conllu(path)

        print(f"\nExtracted {len(sent_texts)} sentences and {len(clusters)} coreference clusters")

        # Show some cluster info
        print(f"\nSample clusters:")
        for cid in list(clusters.keys())[:3]:
            mentions = clusters[cid]
            print(f"\n  Cluster {cid} ({len(mentions)} mentions):")
            for m in mentions[:3]:
                pron_marker = " [PRONOUN]" if m.is_pron else ""
                print(f"    - Sent {m.sent_idx}: '{m.span_text}'{pron_marker}")

        # Build QA items
        dsname = path.split('/')[-1].replace('.conllu', '')
        items = build_pronoun_qa(sent_texts, sentences, clusters, dataset_name=dsname)

        print(f"\n\nGenerated {len(items)} pronoun coreference QA items")
        print(f"\n{'='*80}")
        print("SAMPLE QA ITEMS:")
        print('='*80)

        for j, item in enumerate(items[:max_examples]):
            print(f"\n--- Example {j+1} ---")
            print(f"ID: {item.qid}")
            print(f"Pronoun: {item.pron_form}")
            print(f"\nContext:")
            print(f"  {item.context}")
            print(f"\nQuestion:")
            print(f"  {item.question}")
            print(f"\nGold answer(s):")
            for ans in item.gold_spans:
                print(f"  - '{ans}'")

        total_items += len(items)

    print(f"\n{'='*80}")
    print(f"SUMMARY: Extracted {total_items} total QA items from {min(max_files, len(files))} file(s)")
    print('='*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_coref_extraction.py <path_to_CorefUD>")
        print("\nExample:")
        print("  python test_coref_extraction.py CorefUD-1.3-public")
        sys.exit(1)

    corefud_path = sys.argv[1]
    show_examples(corefud_path, max_files=1, max_examples=5)
