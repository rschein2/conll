# CorefUD Pronoun Coreference Benchmark for LLMs

This repository contains tools to test LLMs on pronoun coreference resolution using CorefUD data and the official CRAC scorer.

## Files

- **`corefud_qa.py`** - Extracts pronoun coreference QA items from CorefUD files
- **`test_coref_extraction.py`** - Shows example coreference problems extracted from CONLL files
- **`llm_coref_benchmark.py`** - Full benchmark pipeline with official scorer integration
- **`corefud-scorer/`** - Official CorefUD scorer (from https://github.com/ufal/corefud-scorer)
- **`CorefUD-1.3-public/`** - CorefUD 1.3 dataset

## Quick Start

### 1. View Examples

See 10 example pronoun coreference problems:

```bash
venv/bin/python test_coref_extraction.py CorefUD-1.3-public
```

Example output:
```
EXAMPLE 1
Pronoun: 'she'

Context:
And her mom wanted to take her home early, and I'm like, no let's stay longer.
But her mom wouldn't let her. And so she went home...

Question:
In the passage below, what does "she" refer to?

Correct Answer(s):
  • Kim
```

### 2. Count Available QA Items

```bash
venv/bin/python3 -c "
from corefud_qa import load_english_files, extract_mentions_from_conllu, build_pronoun_qa
import pathlib

files = load_english_files('CorefUD-1.3-public')
total = 0
for path in files:
    sent_texts, sentences, clusters = extract_mentions_from_conllu(path)
    items = build_pronoun_qa(sent_texts, sentences, clusters, dataset_name=pathlib.Path(path).stem)
    total += len(items)
    print(f'{pathlib.Path(path).stem:40s}: {len(items):5d} items')
print(f'\nTOTAL: {total} pronoun coreference QA items')
"
```

**Result: 21,591 QA items** across 4 English files:
- en_gum-corefud-dev: 1,410 items
- en_gum-corefud-train: 7,440 items
- en_litbank-corefud-dev: 1,391 items
- en_litbank-corefud-train: 11,350 items

### 3. Run LLM Benchmark

Edit `llm_coref_benchmark.py` to add your LLM API call in the `call_llm()` function:

```python
def call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
```

Then run:

```bash
venv/bin/python llm_coref_benchmark.py CorefUD-1.3-public 100
```

This will:
1. Extract 100 pronoun coreference QA items
2. Evaluate your LLM using 4 different matching strategies
3. Report accuracy for each matching mode
4. Save detailed results to `llm_coref_results.json`

**Example output:**
```
Matching Mode Results:
  Exact Match:       45 / 100 = 45.00%
  Partial Match:     68 / 100 = 68.00%
  Token F1 ≥ 0.6:    63 / 100 = 63.00%
  Head Word Match:   71 / 100 = 71.00%  ← Primary (CRAC-inspired)

Primary Metric (Head Word Match): 71.00%
```

### 4. Evaluation Matching Modes

The benchmark uses **4 matching strategies** inspired by CRAC shared task evaluation:

#### **Head Word Match** (Primary Metric - CRAC-inspired)
- Matches on syntactic head noun only
- Most linguistically motivated
- Accepts: `"native-like levels"` for gold `"native-like levels of use and neurocognitive processing"` ✓
- Both have head word `"levels"`

#### **Partial Match** (Lenient)
- Accepts if prediction is substring of gold OR vice versa
- Good for recall-focused evaluation
- Accepts: `"native-like levels"` ⊆ `"native-like levels of use..."` ✓

#### **Token F1 Match** (Balanced)
- Computes token-level F1 score
- Accepts if F1 ≥ 0.6 threshold
- Handles partial overlaps fairly
- Example: {native-like, levels} ∩ {such, native-like, levels} → F1 = 0.80 ✓

#### **Exact Match** (Strict)
- Requires normalized string match
- Most conservative metric
- Baseline for comparison

**Why multiple metrics?** LLMs may identify the correct referent but use a different span length (e.g., `"levels"` vs `"native-like levels of use"`). Head word matching gives credit when the LLM correctly identifies the entity, similar to how CRAC 2024 evaluates systems.

### 5. Use Official CorefUD Scorer

The official scorer compares two CoNLL-U files (gold vs predicted) and computes standard metrics:

```bash
venv/bin/python corefud-scorer/corefud-scorer.py \
    gold_file.conllu \
    predicted_file.conllu \
    -m muc bcub ceafe
```

**Metrics:**
- **MUC** - Link-based metric
- **B-cubed** - Mention-based metric
- **CEAF-e** - Entity-based metric
- **CoNLL score** - Average F1 of MUC, B-cubed, CEAF-e (official CRAC metric)

**Example test:**

```bash
venv/bin/python corefud-scorer/corefud-scorer.py \
    corefud-scorer/tests/tests-corefud/head-match/TC-HMA.key \
    corefud-scorer/tests/tests-corefud/head-match/TC-HMA-1.response
```

## Understanding the Data

### CorefUD Format

CorefUD files are in CoNLL-U format with Entity annotations in the `MISC` column:

**Single-token mention (e.g., pronoun "he"):**
```
5  he  ...  Entity=(e26060-person-1-salience:sssss-giv:act-cf1*-1-ana-Lord_Byron)
```
- `(eID` = mention opens
- `)` at end = mention closes on same token

**Multi-token mention (e.g., "Lord Byron"):**
```
Token 1: Entity=(e26060-person-2-...    # Opens
Token 2: Entity=e26060)                  # Closes
```

All mentions with the same entity ID (e.g., `e26060`) are in the same coreference cluster.

### Pronoun Types Included

```python
PRON_FORMS = {
    "he", "him", "his",           # masculine
    "she", "her", "hers",         # feminine
    "it", "its",                  # neuter
    "they", "them", "their",      # plural/singular they
    "we", "us", "our", "ours",    # 1st person plural
    "i", "me", "my", "mine",      # 1st person singular
    "you", "your", "yours",       # 2nd person
}
```

Edit `PRON_FORMS` in `corefud_qa.py` to focus on specific pronouns (e.g., only `{"he", "him", "his", "she", "her"}`).

## Next Steps

### For QA-Based Evaluation

1. Implement your `call_llm()` function in `llm_coref_benchmark.py`
2. Run the benchmark: `venv/bin/python llm_coref_benchmark.py CorefUD-1.3-public 1000`
3. Review results across all 4 matching modes
4. Use **Head Word Match** as primary metric for comparing models (CRAC-inspired)
5. Analyze per-example results in `llm_coref_results.json`

**Comparing models:**
```python
# Baseline model results
baseline_head_match = 65.3%

# Your experimental model results
experimental_head_match = 72.1%

# Improvement: +6.8 percentage points
```

### For Official Scorer Evaluation

The official scorer requires predicted Entity annotations in CoNLL-U format. To use it:

1. **Generate predictions:** Run your LLM on the test data
2. **Convert to CoNLL-U:** Map LLM predictions back to Entity annotations
3. **Run scorer:** Compare gold vs predicted files

This is more complex but gives you standard CRAC shared task metrics.

## Key Improvements

### Bug Fix: Single-Token Mention Parsing
The original code had a critical bug in parsing single-token mentions. CorefUD uses:
- `(eID-attributes)` for single tokens (both open & close brackets)
- The regex only detected closes in format `eID)` without preceding `(`
- **Result:** Missed most pronouns (only found 3 instead of 1,917!)

**Fix:** Added logic to detect when Entity annotation contains both `(` and `)`, indicating a complete single-token mention.

### Fair Evaluation with Multiple Matching Modes
Added CRAC-inspired evaluation that recognizes correct answers even with different span lengths:
- **Head Word Match**: Primary metric matching on syntactic heads (like CRAC 2024)
- **Partial Match**: Lenient substring matching
- **Token F1**: Balanced token overlap scoring
- **Exact Match**: Strict baseline

This gives fair credit when LLMs identify the correct referent but use different boundaries (e.g., `"levels"` vs `"native-like levels of use"`), similar to how the official CRAC shared task evaluates systems.

## References

- **CRAC 2025:** https://ufal.mff.cuni.cz/corefud/crac25
- **CorefUD Scorer:** https://github.com/ufal/corefud-scorer
- **CorefUD Format:** https://ufal.mff.cuni.cz/corefud/
- **CoNLL-U Format:** https://universaldependencies.org/format.html
