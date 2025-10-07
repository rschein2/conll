# corefud_pronoun_qa.py
# Benchmark English pronoun coreference via LLM prompting on CorefUD data.

import re, glob, json, random, collections, pathlib
from typing import List, Dict, Tuple, Optional
from conllu import parse_incr

# -----------------------------
# 0) Small utilities
# -----------------------------

PRON_FORMS = {
    "he", "him", "his", "she", "her", "hers",  # gendered
    "it", "its",  # neuter
    "they", "them", "their", "theirs",  # plural/singular they
    "we", "us", "our", "ours",  # first person plural
    "i", "me", "my", "mine",  # first person singular
    "you", "your", "yours",  # second person
}  # expand or tighten as needed
DET_ARTICLES = {"a", "an", "the"}

def normalize_span(s: str) -> str:
    s = s.strip()
    # remove leading determiners for len>1 spans: "the White House" -> "White House" optional
    toks = s.split()
    if toks and toks[0].lower() in DET_ARTICLES and len(toks) > 1:
        toks = toks[1:]
    s = " ".join(toks)
    # simple punctuation strip at ends
    return s.strip(" ,.;:!?\"'()[]{}").lower()

def join_tokens(tokens):
    # reconstruct sentence text from tokens honoring SpaceAfter=No
    out = []
    for tok in tokens:
        form = tok["form"]
        misc = tok.get("misc") or {}
        out.append(form)
        if not (misc.get("SpaceAfter") == "No"):
            out.append(" ")
    return "".join(out).strip()

# -----------------------------
# 1) Parse CorefUD Entity brackets
# -----------------------------
# CorefUD Entity value marks mention starts "(eID...)" and ends "eID)".
# We only need cluster IDs and span boundaries (start/end token indices).
# Spec: https://ufal.mff.cuni.cz/~zeman/2022/docs/corefud-1.0-format.pdf

EID = r"e\d+(?:\[\d+/\d+\])?"             # allow discontinuous notation
OPEN_RE  = re.compile(r"\((" + EID + r")")  # captures eID after '('
CLOSE_RE = re.compile(r"(?<!\()(" + EID + r")\)")

def entity_ops(entity_val: str) -> Tuple[List[str], List[str]]:
    """Return (opens, closes) eID lists found at this token."""
    if not entity_val:
        return [], []
    # Remove any attribute tail after the eID in opens, e.g. "(e12-person-...)".
    # OPEN_RE gets "e12" (possibly with [i/n]); attributes follow, but we ignore.
    opens  = OPEN_RE.findall(entity_val)
    closes = CLOSE_RE.findall(entity_val)

    # Handle CorefUD format where single-token mentions use: (eID-attributes)
    # If we see "(eID...)" pattern, it means both open and close on same token
    for opened_id in opens:
        # Check if this opened ID also has a closing ) somewhere after it
        # Pattern: (eID-attributes...) or (eID[i/n]-attributes...)
        base_id = opened_id.split("[")[0]
        # Look for pattern like (eID*) where * is any attributes
        if f"({opened_id}" in entity_val and ")" in entity_val:
            # This is a complete mention on one token
            if base_id not in closes:
                closes.append(base_id)

    # Strip [i/n] part for cluster-level identity, we evaluate spans per part anyway.
    opens  = [o.split("[")[0] for o in opens]
    closes = [c.split("[")[0] for c in closes]
    return opens, closes

# -----------------------------
# 2) Read CoNLL-U and recover mentions/clusters
# -----------------------------

Mention = collections.namedtuple("Mention", "cluster_id sent_idx start end tokens is_pron span_text")

def extract_mentions_from_conllu(path: str) -> Tuple[List[str], List[List[Dict]], Dict[str, List[Mention]]]:
    """
    Returns:
      sent_texts: list of sentence texts
      sentences: list of token dict lists (per sentence)
      clusters: dict cluster_id -> list of Mention
    """
    sent_texts, sentences = [], []
    clusters: Dict[str, List[Mention]] = collections.defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        # parse_incr yields TokenList objects per sentence
        open_spans = collections.defaultdict(list)  # cluster_id -> list of (sent_idx, start_token_idx)
        for sent_idx, sent in enumerate(parse_incr(f)):
            # CoNLL-U gives .metadata (may include "text") and list of tokens; ignore multi-word/empty here
            toks = [t for t in sent if isinstance(t["id"], int)]
            sentences.append(toks)
            text = sent.metadata.get("text")
            if not text:
                text = join_tokens(toks)
            sent_texts.append(text)

            for i, tok in enumerate(toks):
                misc = tok.get("misc") or {}
                ent = misc.get("Entity")
                if not ent:
                    continue
                opens, closes = entity_ops(ent)
                # Open mentions start here
                for cid in opens:
                    open_spans[cid].append((sent_idx, i))
                # Close mentions end here (close the *most recent* open for this cid)
                for cid in closes:
                    if not open_spans[cid]:
                        continue  # tolerate malformed cases
                    start_sent, start_i = open_spans[cid].pop()  # LIFO for nesting
                    # Collect tokens from start to here (same sentence only for simplicity).
                    # Mentions can cross sentences but are rare in English sets; we skip cross-sent spans.
                    if start_sent == sent_idx:
                        mtoks = toks[start_i:i+1]
                        span = join_tokens(mtoks)
                        # Pronoun if single token and UPOS=PRON and form in our target set
                        is_pron = (
                            len(mtoks) == 1
                            and mtoks[0].get("upostag") == "PRON"
                            and mtoks[0]["form"].lower() in PRON_FORMS
                        )
                        mention = Mention(
                            cluster_id=cid,
                            sent_idx=sent_idx,
                            start=start_i,
                            end=i,
                            tokens=mtoks,
                            is_pron=is_pron,
                            span_text=span,
                        )
                        clusters[cid].append(mention)
                    else:
                        # Cross-sentence mention: ignore for this pronoun QA benchmark
                        pass

    # Sort mentions within each cluster by (sentence, token)
    for cid in clusters:
        clusters[cid].sort(key=lambda m: (m.sent_idx, m.start))
    return sent_texts, sentences, clusters

# -----------------------------
# 3) Build pronoun QA items
# -----------------------------

QAItem = collections.namedtuple("QAItem", "qid context question gold_spans pron_form dataset")

def build_pronoun_qa(sent_texts, sentences, clusters, dataset_name=""):
    items = []
    qid_counter = 0

    # Map (sent_idx, tok_idx) -> cluster ids that open/close there to find pronouns quickly
    # Instead, scan all clusters for pronouns (cheaper to implement)
    for cid, mentions in clusters.items():
        # gather all antecedent (non-pronoun) mentions for this cluster, by position
        for j, m in enumerate(mentions):
            if not m.is_pron:
                continue
            # find previous non-PRON mention in this cluster
            candidates = [x for x in mentions[:j] if not x.is_pron]
            if not candidates:
                continue  # skip cataphora or no nominal antecedent before
            # Compose context: previous 2 sentences + current
            start_sent = max(0, m.sent_idx - 2)
            ctx = " ".join(sent_texts[start_sent : m.sent_idx + 1])
            # Acceptable gold spans: every earlier non-PRON mention that appears in the context text
            gold_raw = sorted({c.span_text for c in candidates}, key=len, reverse=True)
            gold_in_ctx = [g for g in gold_raw if normalize_span(g) in normalize_span(ctx)]
            if not gold_in_ctx:
                # fall back to nearest candidate even if context window is too short
                gold_in_ctx = [candidates[-1].span_text]
            pron_tok = m.tokens[0]["form"]
            q = f'In the passage below, what does "{pron_tok}" refer to? Answer with an exact span copied from the passage.'
            qid_counter += 1
            items.append(QAItem(
                qid=f"{dataset_name}-{qid_counter}",
                context=ctx,
                question=q,
                gold_spans=gold_in_ctx,
                pron_form=pron_tok.lower(),
                dataset=dataset_name,
            ))
    return items

# -----------------------------
# 4) LLM hook + evaluation
# -----------------------------

def call_llm(prompt: str) -> str:
    """
    Replace this stub with your model call (OpenAI, local, etc.) and
    ensure you return a single string containing ONLY the answer span.

    Example (OpenAI Responses API pseudocode):
        from openai import OpenAI
        client = OpenAI()
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return out.choices[0].message.content.strip()
    """
    raise NotImplementedError("Plug in your LLM here")

def evaluate_predictions(items: List[QAItem], preds: List[str]) -> Dict:
    assert len(items) == len(preds)
    total = len(items)
    correct = 0
    by_pron = collections.Counter()
    by_pron_correct = collections.Counter()

    for it, pred in zip(items, preds):
        pred_norm = normalize_span(pred)
        gold_norms = {normalize_span(g) for g in it.gold_spans}
        ok = pred_norm in gold_norms
        by_pron[it.pron_form] += 1
        if ok:
            by_pron_correct[it.pron_form] += 1
            correct += 1

    overall = correct / total if total else 0.0
    per_pron = {p: (by_pron_correct[p] / n if n else 0.0) for p, n in by_pron.items()}
    return {
        "n_items": total,
        "accuracy": overall,
        "per_pron_accuracy": per_pron,
    }

# -----------------------------
# 5) Putting it together
# -----------------------------

def load_english_files(corefud_root: str) -> List[str]:
    """
    Finds typical English CorefUD file names under a root directory.
    Adjust the glob if your layout differs.
    """
    patterns = [
        "en_*corefud-train.conllu",
        "en_*corefud-dev.conllu",
        "en_*corefud-test.conllu",
        # Some distributions use 'english_*'
        "english_*corefud-*.conllu",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(pathlib.Path(corefud_root) / "**" / pat), recursive=True))
    # Keep only English-GUM and English-LitBank by default; include ParCorFull if you want it.
    keep = []
    for f in files:
        lf = f.lower()
        if "en_gum" in lf or "english-gum" in lf or "litbank" in lf:
            keep.append(f)
        # Uncomment to include ParCorFull (CC BY-NC 4.0 if TED omitted).
        # elif "parcorfull" in lf:
        #     keep.append(f)
    return sorted(set(keep))

def make_benchmark(corefud_root: str, max_items_per_file: int = 500, seed: int = 13) -> List[QAItem]:
    random.seed(seed)
    qa = []
    for path in load_english_files(corefud_root):
        sent_texts, sentences, clusters = extract_mentions_from_conllu(path)
        dsname = pathlib.Path(path).stem
        items = build_pronoun_qa(sent_texts, sentences, clusters, dataset_name=dsname)
        # downsample for quicker runs if desired
        if max_items_per_file and len(items) > max_items_per_file:
            items = random.sample(items, max_items_per_file)
        qa.extend(items)
        print(f"{dsname}: {len(items)} items")
    return qa

def run_and_score(corefud_root: str):
    items = make_benchmark(corefud_root)
    prompts = []
    for it in items:
        prompt = f"{it.context}\n\n{it.question}\n\n(Answer with one span only.)"
        prompts.append(prompt)
    # Call your LLM here
    preds = []
    for p in prompts:
        preds.append(call_llm(p))  # <- supply your implementation
    report = evaluate_predictions(items, preds)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    # Example usage:
    # python corefud_pronoun_qa.py /path/to/CorefUD-1.3
    import sys
    if len(sys.argv) != 2:
        print("Usage: python corefud_pronoun_qa.py /path/to/CorefUD")
        sys.exit(1)
    run_and_score(sys.argv[1])
