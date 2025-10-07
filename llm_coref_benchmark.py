#!/usr/bin/env python3
"""
LLM Coreference Benchmark using Official CorefUD Scorer

This script:
1. Extracts pronoun coreference QA items from CorefUD files
2. Evaluates an LLM on these items
3. Converts LLM predictions back to CoNLL-U format
4. Runs the official CorefUD scorer for standard metrics
"""

import sys
import json
import pathlib
from typing import List, Dict, Tuple
from conllu import parse_incr
from corefud_qa import (
    extract_mentions_from_conllu,
    build_pronoun_qa,
    load_english_files,
    normalize_span,
    QAItem
)


def call_llm(prompt: str) -> str:
    """
    Replace this stub with your LLM call.

    Example for OpenAI:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    Example for Claude:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    """
    # For testing, return a dummy answer
    return "DUMMY_ANSWER"


def evaluate_qa_items(items: List[QAItem]) -> Tuple[List[str], Dict]:
    """
    Evaluate LLM on pronoun coreference QA items.
    Returns: (predictions, metrics)
    """
    predictions = []
    correct = 0

    for i, item in enumerate(items):
        # Build prompt
        prompt = f"{item.context}\n\n{item.question}"

        # Get LLM prediction
        pred = call_llm(prompt)
        predictions.append(pred)

        # Check if correct (simple string matching)
        pred_norm = normalize_span(pred)
        gold_norms = {normalize_span(g) for g in item.gold_spans}
        if pred_norm in gold_norms:
            correct += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1}/{len(items)} items...")

    accuracy = correct / len(items) if items else 0.0
    metrics = {
        "total": len(items),
        "correct": correct,
        "accuracy": accuracy
    }

    return predictions, metrics


def update_corefud_with_predictions(
    gold_file: str,
    output_file: str,
    qa_items: List[QAItem],
    predictions: List[str]
) -> None:
    """
    Create a predicted CoNLL-U file by updating coreference annotations
    based on LLM predictions.

    This is a simplified version - it keeps all gold annotations but
    can be extended to modify specific pronoun-antecedent links.
    """
    # For now, just copy the gold file
    # A full implementation would:
    # 1. Parse predictions to find predicted antecedent spans
    # 2. Map those spans back to entity IDs in the CoNLL-U file
    # 3. Update the pronoun's Entity annotation to match predicted cluster

    with open(gold_file, 'r') as f_in, open(output_file, 'w') as f_out:
        f_out.write(f_in.read())

    print(f"  Note: Created predicted file (currently same as gold)")
    print(f"  Full prediction integration requires mapping text spans to entity IDs")


def run_official_scorer(gold_file: str, pred_file: str, scorer_path: str = "corefud-scorer"):
    """Run the official CorefUD scorer."""
    import subprocess

    scorer_script = pathlib.Path(scorer_path) / "corefud-scorer.py"

    cmd = [
        "venv/bin/python",
        str(scorer_script),
        gold_file,
        pred_file,
        "-m", "muc", "bcub", "ceafe"  # Main CRAC metrics
    ]

    print(f"\nRunning official CorefUD scorer...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)


def main(corefud_root: str, max_items: int = 100):
    """Main benchmark pipeline."""

    print("=" * 80)
    print("LLM COREFERENCE BENCHMARK - CorefUD Official Scorer")
    print("=" * 80)

    # 1. Load English files
    files = load_english_files(corefud_root)
    if not files:
        print(f"No English CorefUD files found in {corefud_root}")
        return

    print(f"\nFound {len(files)} English CorefUD files")

    # For demo, just use dev set
    dev_files = [f for f in files if "dev" in f]
    if not dev_files:
        dev_files = files[:1]

    print(f"Using {len(dev_files)} file(s) for evaluation\n")

    # 2. Extract QA items
    all_items = []
    for path in dev_files:
        print(f"Processing: {path}")
        sent_texts, sentences, clusters = extract_mentions_from_conllu(path)
        dsname = pathlib.Path(path).stem
        items = build_pronoun_qa(sent_texts, sentences, clusters, dataset_name=dsname)
        print(f"  Extracted {len(items)} pronoun coreference QA items")
        all_items.extend(items)

    # Limit for demo
    if max_items and len(all_items) > max_items:
        print(f"\nLimiting to first {max_items} items for demo")
        all_items = all_items[:max_items]

    print(f"\nTotal QA items: {len(all_items)}\n")

    # 3. Evaluate LLM
    print("=" * 80)
    print("EVALUATING LLM ON PRONOUN COREFERENCE")
    print("=" * 80)

    predictions, metrics = evaluate_qa_items(all_items)

    print(f"\n{'=' * 80}")
    print("QA-BASED EVALUATION RESULTS")
    print('=' * 80)
    print(f"Total items: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

    # 4. Save results
    results = {
        "metrics": metrics,
        "predictions": [
            {
                "qid": item.qid,
                "pronoun": item.pron_form,
                "prediction": pred,
                "gold_answers": item.gold_spans,
                "correct": normalize_span(pred) in {normalize_span(g) for g in item.gold_spans}
            }
            for item, pred in zip(all_items, predictions)
        ]
    }

    results_file = "llm_coref_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # 5. (Optional) Run official scorer
    # To use the official scorer, you would need to:
    # - Convert LLM predictions back to CoNLL-U format with Entity annotations
    # - Run the scorer comparing gold vs predicted files

    print("\n" + "=" * 80)
    print("USING THE OFFICIAL COREFUD SCORER")
    print("=" * 80)
    print("\nTo use the official scorer, you need to:")
    print("1. Convert LLM predictions to CoNLL-U format with Entity annotations")
    print("2. Run: venv/bin/python corefud-scorer/corefud-scorer.py gold.conllu pred.conllu")
    print("\nThe scorer computes standard metrics:")
    print("  - MUC, B-cubed, CEAF-e (CoNLL score = average F1)")
    print("  - LEA, BLANC, and other metrics")
    print("\nExample test:")

    # Demo with test files
    test_gold = "corefud-scorer/tests/tests-corefud/head-match/TC-HMA.key"
    test_pred = "corefud-scorer/tests/tests-corefud/head-match/TC-HMA-1.response"
    if pathlib.Path(test_gold).exists():
        run_official_scorer(test_gold, test_pred)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_coref_benchmark.py <path_to_CorefUD> [max_items]")
        print("\nExample:")
        print("  python llm_coref_benchmark.py CorefUD-1.3-public 100")
        sys.exit(1)

    corefud_path = sys.argv[1]
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    main(corefud_path, max_items)
