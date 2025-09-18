import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set


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
    "**", "&&", "||", "&&&", "|||", "^^^", "~~~", "<<<", ">>>", "@",
]

FSHARP_SINGLE_CHAR_OPERATORS = list(
    {
        "+", "-", "*", "/", "%", "=", "<", ">", ".", ",",
        ":", "|", "&", "^", "~", "?", "!", "[", "]", "{", "}", "(", ")",
    }
)


_RE_BLOCK_COMMENT = re.compile(r"\(\*[\s\S]*?\*\)")
_RE_LINE_COMMENT = re.compile(r"//.*?(?=\n|$)")

_RE_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_RE_CHAR = re.compile(r"'(?:[^'\\]|\\.)'")
_RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_RE_IDENTIFIER = re.compile(r"\b[_A-Za-z][A-Za-z0-9_']*\b")

# Printf-format specifiers inside strings: normalize by final letter (e.g., %f, %d, %s)
_RE_PRINTF_SPEC = re.compile(r"%(?:\d+\$)?(?:\.\d+)?([a-zA-Z])")


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


def _tokenize(source_code: str) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
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
    bracket_open = 0
    bracket_close = 0

    pos = 0
    while pos < len(code_masked):
        m = token_pattern.match(code_masked, pos)
        if not m:
            # skip
            pos += 1
            continue
        pos = m.end()
        tok = m.group(0)
        if tok.isspace():
            continue
        
        # Check for attributes before processing brackets: [<Identifier>]
        if tok == "[" and pos <= len(code_masked):
            temp_pos = pos
            next_tokens = []
            for _ in range(5):  # Check next 5 tokens
                if temp_pos < len(code_masked):
                    m = token_pattern.match(code_masked, temp_pos)
                    if m and not m.group(0).isspace():
                        next_tokens.append(m.group(0))
                        temp_pos = m.end()
                    else:
                        break
                else:
                    break
            
            if (len(next_tokens) >= 5 and 
                next_tokens[0] == "[" and 
                next_tokens[1] == "<" and 
                _RE_IDENTIFIER.fullmatch(next_tokens[2]) and 
                next_tokens[3] == ">" and 
                next_tokens[4] == "]"):
                # Add attribute as single token
                tokens.append(f"[<{next_tokens[2]}>]")
                pos = temp_pos
                continue
        
        # Parentheses are handled as pairs; count but don't emit individually
        if tok == "(":
            paren_open += 1
            continue
        if tok == ")":
            paren_close += 1
            continue

        # Square brackets treated as a single operand '[]' per balanced pair
        if tok == "[":
            bracket_open += 1
            continue
        if tok == "]":
            bracket_close += 1
            continue
        tokens.append(tok)

    readkey_calls = code.count('Console.ReadKey()')
    special_counts = {"()": readkey_calls} if readkey_calls > 0 else {}
    bracket_pairs = min(bracket_open, bracket_close)
    special_operand_counts = {}

    # Restore literal placeholders to their original values for operand identity
    restored: List[str] = []
    for t in tokens:
        if t.startswith("__LIT"):
            restored.append(_restore_literal(t, literals))
        else:
            restored.append(t)

    return restored, special_counts, special_operand_counts


def _classify_tokens(tokens: List[str], special_counts: Dict[str, int], special_operand_counts: Dict[str, int]) -> Tuple[Counter, Counter]:
    operator_counts: Counter = Counter()
    operand_counts: Counter = Counter()

    # Count special paired parentheses as a single operator type '()'
    for op, cnt in special_counts.items():
        operator_counts[op] += cnt

    # Count special operand for list brackets '[]'
    for opd, cnt in special_operand_counts.items():
        operand_counts[opd] += cnt


    # Prepare operator symbol set for quick membership
    multi_ops = set(FSHARP_MULTI_CHAR_OPERATORS)
    single_ops = set(FSHARP_SINGLE_CHAR_OPERATORS)

    # Track function names discovered via definitions
    known_functions: Set[str] = set()
    
    # System function patterns - any function matching these patterns is an operator
    def is_system_function(name: str) -> bool:
        
        # Single word system functions
        single_system_functions = {
            "printf", "printfn", "sprintf", "failwith", "sqrt", "ignore",
            "abs", "sin", "cos", "tan", "map", "iter", "filter",
            "head", "tail", "isEmpty", "fold", "foldBack", "reduce", "sum",
            "max", "min", "sort", "sortBy", "groupBy", "distinct", "exists",
            "forall", "find", "tryFind", "choose", "collect", "concat"
        }
        
        # Special system properties that should be operators
        special_system_properties = {
            "Math.E", "Math.Tau", "Math.Sqrt2", "Math.Sqrt1_2",
            "Math.LN2", "Math.LN10", "Math.LOG2E", "Math.LOG10E"
        }
        
        if name in single_system_functions:
            return True
            
        if name in special_system_properties:
            return True
            
        # Pattern: Module.Function
        if '.' in name:
            parts = name.split('.')
            if len(parts) == 2:
                module, func = parts
                # System modules (capitalized)
                if module[0].isupper():
                    return True
                # Special system prefixes
                if module in {"System", "Microsoft", "FSharp", "Int32", "Int64", 
                             "Double", "Single", "String", "Char", "Boolean",
                             "List", "Array", "Seq", "Console", "DateTime"}:
                    return True
                    
        return False

    # When a 'match-with' is detected, suppress counting of '|', '->', and 'when' that follow shortly
    suppress_match_markers_window = 0

    processed_indices = set()
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] == "<" and 
            tokens[i + 1] == "EntryPoint" and 
            tokens[i + 2] == ">"):
            # Count as [<EntryPoint>] operator
            operator_counts["[<EntryPoint>]"] += 1
            # Mark constituent tokens as processed
            for j in range(i, i + 3):
                processed_indices.add(j)
            i += 3
        else:
            i += 1

    i = 0
    while i < len(tokens) - 1:
        if (tokens[i] == "|" and 
            tokens[i + 1] == "|"):
            # Count as [] operand
            operand_counts["[]"] += 1
            # Mark constituent tokens as processed
            for j in range(i, i + 2):
                processed_indices.add(j)
            i += 2
        else:
            i += 1

    # Track compound operators
    i = 0
    while i < len(tokens):
        # Skip tokens that were already processed as part of EntryPoint attribute
        if i in processed_indices:
            i += 1
            continue
        tok = tokens[i]
        lower_tok = tok.lower()
        
        # Check for compound operators
        if lower_tok == "for" and i + 2 < len(tokens):
            # Look for "for ... in ... do" or "for ... to ... do" patterns
            found_in = False
            found_to = False
            found_do = False
            j = i + 1
            while j < len(tokens) and j < i + 10:  # lookahead
                if tokens[j].lower() == "in":
                    found_in = True
                elif tokens[j].lower() == "to":
                    found_to = True
                elif tokens[j].lower() == "do" and (found_in or found_to):
                    found_do = True
                    break
                j += 1
            
            if found_in and found_do:
                operator_counts["for-in-do"] += 1
                # skip the 'for' keyword itself
                i += 1
                continue
            elif found_to and found_do:
                operator_counts["for-to-do"] += 1
                # skip the 'for' keyword itself
                i += 1
                continue
        
        if lower_tok == "match" and i + 1 < len(tokens):
            # Look for "match ... with" pattern
            found_with = False
            j = i + 1
            while j < len(tokens) and j < i + 10:  # lookahead
                if tokens[j].lower() == "with":
                    found_with = True
                    break
                j += 1
            
            if found_with:
                operator_counts["match-with"] += 1
                # skip the 'match' keyword itself
                i += 1
                suppress_match_markers_window = 50  # suppress subsequent '|', '->', 'when'
                continue

        # Function definition: let <name> <params> ... =
        if lower_tok == "let" and i + 1 < len(tokens):
            # handle "let rec" case
            if i + 2 < len(tokens) and tokens[i + 1] == "rec":
                name_tok = tokens[i + 2]
            else:
                name_tok = tokens[i + 1]
            if _RE_IDENTIFIER.fullmatch(name_tok):
                # look ahead to see '=' present soon
                # for "let rec", start searching from position after "rec"
                if i + 2 < len(tokens) and tokens[i + 1] == "rec":
                    j = i + 3
                else:
                    j = i + 2
                has_eq = False
                has_params_before_eq = False
                while j < len(tokens) and j < i + 15:
                    if tokens[j] == "=":
                        has_eq = True
                        break
                    # treat identifiers or '(' as parameters prior to '='
                    if _RE_IDENTIFIER.fullmatch(tokens[j]) or tokens[j] == "(":
                        has_params_before_eq = True
                    j += 1
                # Only count as function declaration if there are parameters before '='
                if has_eq and has_params_before_eq:
                    # Count let as operator for function declaration
                    operator_counts["let"] += 1
                    known_functions.add(name_tok)
                    # Skip the function name token to avoid double counting
                    i += 1
                    continue
        
        if lower_tok == "while" and i + 1 < len(tokens):
            # Look for "while ... do" pattern
            found_do = False
            j = i + 1
            while j < len(tokens) and j < i + 10:  # lookahead
                if tokens[j].lower() == "do":
                    found_do = True
                    break
                j += 1
            
            if found_do:
                operator_counts["while-do"] += 1
                # Only skip the 'while' keyword itself, let other tokens be processed normally
                i += 1
                continue
        
        if lower_tok == "if" and i + 1 < len(tokens):
            
            found_then = False
            j = i + 1
            while j < len(tokens) and j < i + 50:  # lookahead
                if tokens[j].lower() == "then":
                    found_then = True
                    break
                j += 1
            
            if found_then:
                operator_counts["if-then-elif-else"] += 1
                i += 1
                continue
        
        if lower_tok == "try" and i + 1 < len(tokens):
            found_with = False
            found_finally = False
            j = i + 1
            while j < len(tokens) and j < i + 15:  # lookahead
                if tokens[j].lower() == "with":
                    found_with = True
                    break
                elif tokens[j].lower() == "finally":
                    found_finally = True
                    break
                j += 1
            
            if found_with:
                operator_counts["try-with"] += 1
                # Skip to after 'with'
                k = i + 1
                while k < len(tokens):
                    if tokens[k].lower() == "with":
                        i = k + 1
                        break
                    k += 1
                continue
            elif found_finally:
                operator_counts["try-finally"] += 1
                # Skip to after 'finally'
                k = i + 1
                while k < len(tokens):
                    if tokens[k].lower() == "finally":
                        i = k + 1
                        break
                    k += 1
                continue

        # num operands
        if _RE_NUMBER.fullmatch(tok):
            operand_counts[tok] += 1
            i += 1
            continue

        if (len(tok) >= 2 and ((tok[0] == '"' and tok[-1] == '"') or (tok[0] == "'" and tok[-1] == "'"))):
            normalized_tok = tok
            if tok[0] == '"':
                for m in _RE_PRINTF_SPEC.finditer(tok):
                    spec_letter = m.group(1).lower()
                    operator_counts[f"%{spec_letter}"] += 1
                normalized_tok = _RE_PRINTF_SPEC.sub("", tok)
            operand_counts[normalized_tok] += 1
            i += 1
            continue

        # check for empty array patterns: [| |] and | |
        if tok == "[" and i + 4 < len(tokens) and tokens[i + 1] == "|" and tokens[i + 2] == " " and tokens[i + 3] == "|" and tokens[i + 4] == "]":
            operand_counts["[| |]"] += 1
            i += 5
            continue
            
        if tok == "|" and i + 2 < len(tokens) and tokens[i + 1] == " " and tokens[i + 2] == "|":
            operand_counts["| |"] += 1
            i += 3
            continue

        # Look for "when ... - >" pattern 
        if lower_tok == "when":
            j = i + 1
            found_arrow = False
            while j < len(tokens) - 1 and j < i + 10:
                if tokens[j] == "-" and tokens[j + 1] == ">":
                    found_arrow = True
                    break
                j += 1
            if found_arrow:
                operator_counts["when->"] += 1
                i += 1
                continue
            
        if lower_tok == "fun":
            j = i + 1
            found_arrow = False
            while j < len(tokens) - 1 and j < i + 5:
                if tokens[j] == "-" and tokens[j + 1] == ">":
                    found_arrow = True
                    break
                j += 1
            if found_arrow:
                operator_counts["fun->"] += 1
                i += 1
                continue
        
        # check for range operator: .. (tokenized as . .)
        if tok == "." and i + 1 < len(tokens) and tokens[i + 1] == ".":
            operator_counts[".."] += 1
            # Skip both . tokens
            i += 2
            continue
        
        # Check for arrow operator: -> (tokenized as - >)
        if tok == "-" and i + 1 < len(tokens) and tokens[i + 1] == ">":
            operator_counts["->"] += 1
            # Skip both - and > tokens
            i += 2
            continue

        # Multi and single character operators
        if tok in multi_ops or tok in single_ops:
            # Do not count '(' or ')' here; they were handled via pairs
            if tok not in {"(", ")"}:
                if suppress_match_markers_window > 0 and (tok in {"|"} or tok == "->"):
                    pass
                else:
                    operator_counts[tok] += 1
            i += 1
            continue

        # Special operand: underscore (wildcard)
        if tok == "_":
            operand_counts["_"] += 1
            i += 1
            continue

        # Identifiers and keywords
        if _RE_IDENTIFIER.fullmatch(tok):
            if lower_tok in FSHARP_KEYWORDS:
                if lower_tok not in {"for", "in", "do", "match", "with", "while", "if", "then", "elif", "else", "try", "finally", "to", "fun", "when"}:
                    operator_counts[lower_tok] += 1
            else:
                # Detect member access: A . B ...
                full_name = tok
                consumed = 1
                if i + 2 < len(tokens) and tokens[i + 1] == "." and _RE_IDENTIFIER.fullmatch(tokens[i + 2]):
                    full_name = f"{tok}.{tokens[i + 2]}"
                    consumed = 3
                    is_call = (i + 3 < len(tokens) and tokens[i + 3] == "(")
                    if is_call:
                        operator_counts[full_name] += 1
                        i += consumed
                        continue
                    else:
                        if full_name == "Math.PI":
                            operand_counts[full_name] += 1
                            i += consumed
                            continue
                        if is_system_function(full_name):
                            operator_counts[full_name] += 1
                            i += consumed
                            continue
                        operand_counts[full_name] += 1
                        i += consumed
                        continue

                if i + 1 < len(tokens) and tokens[i + 1] == "(":
                    operator_counts[tok] += 1
                elif tok in known_functions or is_system_function(tok):
                    operator_counts[tok] += 1
                else:
                    operand_counts[tok] += 1
            i += 1
            continue

        operator_counts[tok] += 1
        i += 1


        if suppress_match_markers_window > 0:
            suppress_match_markers_window -= 1

    return operator_counts, operand_counts


def analyze_fsharp_source(source_code: str) -> HalsteadResult:
    tokens, special_counts, special_operand_counts = _tokenize(source_code)
    op_counts, opd_counts = _classify_tokens(tokens, special_counts, special_operand_counts)

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


