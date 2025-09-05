import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


FSHARP_KEYWORDS = {
    # core bindings and control flow
    "let", "in", "do", "done", "rec", "mutable", "if", "then", "elif", "else",
    "match", "with", "when", "function", "fun", "return", "yield",
    "for", "to", "downto", "while", "try", "finally", "raise", "exception",
    # modules/types
    "module", "open", "namespace", "type", "member", "inherit", "interface",
    # logical
    "and", "or", "not",
    # computation expressions/common
    "bind", "use", "new", "class", "struct", "end",
}


# Multi-character operators should be matched before single-character ones
FSHARP_MULTI_CHAR_OPERATORS = [
    ">>=", "<=", ">=", "<>", "<-", ":=", "|>", "<|", ">>", "<<", "::",
]

FSHARP_SINGLE_CHAR_OPERATORS = list(
    {
        "+", "-", "*", "/", "%", "=", "<", ">", ".", ";", ",",
        ":", "|", "&", "^", "~", "?", "!", "[", "]", "{", "}", "(", ")",
    }
)


_RE_BLOCK_COMMENT = re.compile(r"\(\*[\s\S]*?\*\)")
_RE_LINE_COMMENT = re.compile(r"//.*?(?=\n|$)")

_RE_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_RE_CHAR = re.compile(r"'(?:[^'\\]|\\.)'")
_RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_RE_IDENTIFIER = re.compile(r"\b[_A-Za-z][A-Za-z0-9_']*\b")


@dataclass
class HalsteadResult:
    operator_frequencies: List[Tuple[str, int]]
    operand_frequencies: List[Tuple[str, int]]
    eta1_unique_operators: int
    eta2_unique_operands: int
    N1_total_operators: int
    N2_total_operands: int
    eta_vocabulary: int
    N_length: int
    V_volume: float


def _strip_comments(source_code: str) -> str:
    # Remove block comments first, then line comments
    without_block = _RE_BLOCK_COMMENT.sub(" ", source_code)
    without_line = _RE_LINE_COMMENT.sub(" ", without_block)
    return without_line


def _extract_and_mask_literals(source: str) -> Tuple[str, List[str]]:
    literals: List[str] = []

    def _store(match: re.Match) -> str:
        literals.append(match.group(0))
        return f" __LIT{len(literals) - 1}__ "

    # Replace strings and chars with placeholders
    masked = _RE_STRING.sub(_store, source)
    masked = _RE_CHAR.sub(_store, masked)
    return masked, literals


def _restore_literal(token: str, literals: List[str]) -> str:
    if token.startswith("__LIT") and token.endswith("__"):
        idx = int(token[5:-2])
        return literals[idx]
    return token


def _tokenize(source_code: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Tokenize F# code into a list of tokens, returning tokens and a special_counts
    dict for paired parentheses occurrences to be recorded as a single operator '()'.
    """
    code = _strip_comments(source_code)
    code_masked, literals = _extract_and_mask_literals(code)

    # Build regex alternation for multi-char operators to match first
    multi_ops_pattern = "|".join(map(re.escape, sorted(FSHARP_MULTI_CHAR_OPERATORS, key=len, reverse=True)))
    single_ops_pattern = "|".join(map(re.escape, FSHARP_SINGLE_CHAR_OPERATORS))

    token_pattern = re.compile(
        rf"\s+|({multi_ops_pattern})|({single_ops_pattern})|({_RE_NUMBER.pattern})|({_RE_IDENTIFIER.pattern})|(__LIT\d+__)",
        re.UNICODE,
    )

    tokens: List[str] = []
    paren_open = 0
    paren_close = 0

    pos = 0
    while pos < len(code_masked):
        m = token_pattern.match(code_masked, pos)
        if not m:
            # Unrecognized character, skip
            pos += 1
            continue
        pos = m.end()
        tok = m.group(0)
        if tok.isspace():
            continue
        # Parentheses are handled as pairs; count but don't emit individually
        if tok == "(":
            paren_open += 1
            continue
        if tok == ")":
            paren_close += 1
            continue
        tokens.append(tok)

    pairs = min(paren_open, paren_close)
    special_counts = {"()": pairs} if pairs > 0 else {}

    # Restore literal placeholders to their original values for operand identity
    restored: List[str] = []
    for t in tokens:
        if t.startswith("__LIT"):
            restored.append(_restore_literal(t, literals))
        else:
            restored.append(t)

    return restored, special_counts


def _classify_tokens(tokens: List[str], special_counts: Dict[str, int]) -> Tuple[Counter, Counter]:
    operator_counts: Counter = Counter()
    operand_counts: Counter = Counter()

    # Count special paired parentheses as a single operator type '()'
    for op, cnt in special_counts.items():
        operator_counts[op] += cnt

    # Prepare operator symbol set for quick membership
    multi_ops = set(FSHARP_MULTI_CHAR_OPERATORS)
    single_ops = set(FSHARP_SINGLE_CHAR_OPERATORS)

    for tok in tokens:
        lower_tok = tok.lower()

        # Numbers: operands
        if _RE_NUMBER.fullmatch(tok):
            operand_counts[tok] += 1
            continue

        # String/char literals begin and end with quotes: operands
        if (len(tok) >= 2 and ((tok[0] == '"' and tok[-1] == '"') or (tok[0] == "'" and tok[-1] == "'"))):
            operand_counts[tok] += 1
            continue

        # Multi and single character operators
        if tok in multi_ops or tok in single_ops:
            # Do not count '(' or ')' here; they were handled via pairs
            if tok not in {"(", ")"}:
                operator_counts[tok] += 1
            continue

        # Identifiers and keywords
        if _RE_IDENTIFIER.fullmatch(tok):
            if lower_tok in FSHARP_KEYWORDS:
                operator_counts[lower_tok] += 1
            else:
                operand_counts[tok] += 1
            continue

        # Fallback: if something slips through, treat punctuation as operator
        operator_counts[tok] += 1

    return operator_counts, operand_counts


def analyze_fsharp_source(source_code: str) -> HalsteadResult:
    tokens, special_counts = _tokenize(source_code)
    op_counts, opd_counts = _classify_tokens(tokens, special_counts)

    # Merge in counts for '.' and ';' to align with typical Halstead definitions
    # (already counted by tokenizer if present)

    # Sort by frequency desc then lexicographically for stable display
    operators_sorted = sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    operands_sorted = sorted(opd_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    eta1 = len(op_counts)
    eta2 = len(opd_counts)
    N1 = sum(op_counts.values())
    N2 = sum(opd_counts.values())

    eta = eta1 + eta2
    N = N1 + N2
    V = float(N) * (math.log2(eta) if eta > 0 else 0.0)

    return HalsteadResult(
        operator_frequencies=operators_sorted,
        operand_frequencies=operands_sorted,
        eta1_unique_operators=eta1,
        eta2_unique_operands=eta2,
        N1_total_operators=N1,
        N2_total_operands=N2,
        eta_vocabulary=eta,
        N_length=N,
        V_volume=V,
    )


