# ivc_cfg.py
"""
Sequitur-style grammar induction (lightweight CFG inference) for symbol sequences.
Author: Generated helper for IVC research
Date: 2025-11-06
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter, deque
import json

# --------------------------
# Sequitur-like implementation
# --------------------------
class Rule:
    def __init__(self, name: str, symbols: List[str]):
        self.name = name        # e.g., "R1"
        self.symbols = list(symbols)  # list of tokens (terminals or nonterminal names)
        self.count = 0          # occurrences (to be filled after algorithm)
    def __repr__(self):
        return f"{self.name} -> {' '.join(self.symbols)} (count={self.count})"

def _make_nt_name(idx: int) -> str:
    return f"R{idx}"

def sequitur_infer(sequences: List[List[str]], min_rule_occurrence: int = 2) -> Dict[str, Any]:
    """
    Infer hierarchical rules from a set of symbol sequences using a
    Sequitur-like repeated-digram replacement approach.

    Args:
        sequences: list of symbol sequences (each sequence is a list of token strings)
        min_rule_occurrence: smallest occurrence count to keep a rule

    Returns:
        {
          "start_rules": [list of top-level sequences as lists using rule names],
          "rules": {rule_name: {"expansion": [...], "count": N}},
          "rule_order": [rule_name1, rule_name2, ...] (creation order),
          "stats": { ... }
        }

    Notes:
        - This is a practical variant of Sequitur; it focuses on repeated digrams across the corpus,
          creates rules, then re-applies replacement until no repeated digrams remain.
        - Works on token sequences (strings), not characters.
    """
    # flatten by joining sequences with a special separator token to avoid cross-sequence digrams
    SEP = "<SEP>"  # will not be considered as repeated across sequences
    corpus = []
    for seq in sequences:
        if not seq:
            continue
        corpus.extend(seq)
        corpus.append(SEP)
    if corpus and corpus[-1] == SEP:
        corpus = corpus[:-1]

    # We'll maintain the corpus as a list we mutate
    tokens = list(corpus)

    # digram -> list of indices (positions of first token)
    digram_index = defaultdict(list)

    def index_all_digrams(tokens_list):
        digram_index.clear()
        for i in range(len(tokens_list)-1):
            if tokens_list[i] == SEP or tokens_list[i+1] == SEP:
                continue
            dg = (tokens_list[i], tokens_list[i+1])
            digram_index[dg].append(i)

    index_all_digrams(tokens)

    rules: Dict[str, Rule] = {}
    created = 0
    replaced = True

    # Replacement loop: find the most frequent digram (occurring >=2), create a rule, replace all non-overlapping occurrences, repeat
    while True:
        # find digrams with count >= 2
        freq_digrams = [(dg, len(pos_list)) for dg, pos_list in digram_index.items() if len(pos_list) >= 2]
        if not freq_digrams:
            break
        # pick the most frequent digram (tie-breaker: leftmost)
        freq_digrams.sort(key=lambda x: (-x[1], digram_index[x[0]][0]))
        digram_to_replace, occ = freq_digrams[0]
        if occ < 2:
            break
        created += 1
        rule_name = _make_nt_name(created)
        # create rule expansion
        expansion = [digram_to_replace[0], digram_to_replace[1]]
        rules[rule_name] = Rule(rule_name, expansion)

        # Replace non-overlapping occurrences in the token list
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] != SEP and tokens[i+1] != SEP and (tokens[i], tokens[i+1]) == digram_to_replace:
                new_tokens.append(rule_name)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        # reindex digrams for next round
        index_all_digrams(tokens)

    # At this point tokens contain top-level mixture of terminals and rule names.
    # Now we may have rules that contain rule names; we need to expand them to final expansions as lists.
    # Also compute counts by scanning token occurrences (count rule_name occurrences)
    text = tokens
    # count occurrences of rule names in text
    counts = Counter([t for t in text if t in rules])
    # Also expand rule counts recursively: if rule contains other rule names, counts propagate later when we compute occurrences across replacements

    # Replace repeated rule bodies that themselves contain repeated digrams (do refinement pass)
    # We'll iterate a few times to collapse repeated adjacent rule sequences (optional)
    for _ in range(2):
        # build temporary digram index across rule bodies
        found_new = False
        bodies = {r.name: r.symbols for r in rules.values()}
        digram_pos = defaultdict(list)
        for rname, symbols in bodies.items():
            for i in range(len(symbols)-1):
                a,b = symbols[i], symbols[i+1]
                if a in rules and b in rules:
                    digram_pos[(a,b)].append((rname, i))
        # if any digram of rule names repeats, replace inside rule bodies with new rule
        repeated = [(dg, len(pos)) for dg, pos in digram_pos.items() if len(pos) >= 2]
        if not repeated:
            break
        repeated.sort(key=lambda x: -x[1])
        dg, occ = repeated[0]
        created += 1
        new_rule_name = _make_nt_name(created)
        # create expansion as the two rule names
        rules[new_rule_name] = Rule(new_rule_name, [dg[0], dg[1]])
        # perform replacements inside bodies
        for rname, poslist in digram_pos[dg]:
            # replace occurrences from right to left to preserve indices
            indices = [p for rn,p in poslist if rn == rname]
            indices.sort(reverse=True)
            for idx in indices:
                syms = rules[rname].symbols
                syms = syms[:idx] + [new_rule_name] + syms[idx+2:]
                rules[rname].symbols = syms
        # loop again to find further compressions

    # compute final counts: count top-level tokens and expand rules to count internal usage
    # First, count occurrences in top-level text tokens
    top_counts = Counter([t for t in text if t in rules])
    # propagate counts into rule internals to get absolute counts
    # We'll perform a simple iterative expansion: each occurrence of a rule contributes its count to its constituents
    rule_counts = {rname: 0 for rname in rules}
    for rname, cnt in top_counts.items():
        rule_counts[rname] += cnt

    # iterate to propagate counts down the rule graph; repeat until stable
    changed = True
    iter_guard = 0
    while changed and iter_guard < 10:
        changed = False
        iter_guard += 1
        for rname, rule in rules.items():
            # each occurrence of rname implies expansion contributes to counts of any child rule names
            rc = rule_counts.get(rname, 0)
            for sym in rule.symbols:
                if sym in rules:
                    prev = rule_counts.get(sym, 0)
                    newval = prev + rc
                    if newval != prev:
                        rule_counts[sym] = newval
                        changed = True

    # set counts on Rule objects
    for rname in rules:
        rules[rname].count = int(rule_counts.get(rname, 0))

    # prune rules below min_rule_occurrence (we only keep rules with count >= threshold)
    pruned_rules = {rname: rule for rname, rule in rules.items() if rule.count >= min_rule_occurrence}

    # rebuild top-level representation replacing pruned rules by their expansions (do a single pass)
    final_text = []
    for t in text:
        if t in pruned_rules:
            final_text.append(t)
        elif t in rules and t not in pruned_rules:
            final_text.extend(rules[t].symbols)
        else:
            final_text.append(t)

    # compute final counts for pruned rules by counting in final_text (simple)
    final_counts = Counter([t for t in final_text if t in pruned_rules])
    for rname in pruned_rules:
        pruned_rules[rname].count = int(final_counts.get(rname, pruned_rules[rname].count))

    # produce ordered list (creation order)
    rule_order = [r for r in sorted(pruned_rules.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)]

    # Prepare exportable dict
    rules_export = {}
    for rname, r in pruned_rules.items():
        rules_export[rname] = {"expansion": list(r.symbols), "count": int(r.count)}

    # produce start rules as list of tokens (terminals + pruned rule names)
    start_rules = final_text

    result = {
        "start_rules": start_rules,
        "rules": rules_export,
        "rule_order": rule_order,
        "stats": {
            "total_rules_created": created,
            "rules_kept": len(pruned_rules),
            "tokens_after_compression": len(start_rules)
        }
    }
    return result

# --------------------------
# Utils: pretty printing & graph building
# --------------------------
def rules_to_text(rules_export: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for rname, info in rules_export.items():
        lines.append(f"{rname} -> {' '.join(info['expansion'])}    # occurrences: {info.get('count',0)}")
    return "\n".join(lines)

def rules_to_graph_edges(rules_export: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Return edges for a directed graph: (rule -> child) for visualization.
    """
    edges = []
    for rname, info in rules_export.items():
        for sym in info.get("expansion", []):
            if sym in rules_export:  # only connect rule->rule edges
                edges.append((rname, sym))
    return edges

def export_rules_json(path: str, seq_result: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seq_result, f, indent=2)

# --------------------------
# Example quick usage
# --------------------------
if __name__ == "__main__":
    # quick demo
    sample_seqs = [
        ["A","B","C","D","E","A","B","C","D","E"],
        ["A","B","C","F","G","A","B","C","F","G"],
        ["X","Y","Z","A","B","C","D"]
    ]
    res = sequitur_infer(sample_seqs, min_rule_occurrence=2)
    print("Rules:")
    print(rules_to_text(res["rules"]))
    print("Start text (compressed):", res["start_rules"])
