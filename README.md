# Halstead Metrics Analyzer for F#

GUI application that tokenizes F# code and computes Halstead metrics (6 base + 3 extended) per `docs/req.md`.

## Features

- Open any `.fs` / `.fsx` file
- Displays operator and operand frequency tables (f₁ⱼ and f₂ᵢ)
- Base metrics: η₁, η₂, N₁, N₂ (in Base Metrics tab)
- Extended metrics: η (vocabulary), N (length), V (volume) (in Extended tab)

## Run

Requirements: Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.gui
```

## Sample

A sample F# file is provided at `sample/FSharpSample.fs`. Open it in the app and click Analyze.

## Notes

- Parentheses are counted as a single operator `()` per pair.
- F# keywords are treated as operators (Holstead interpretation), identifiers/constants as operands.
