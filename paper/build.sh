#!/usr/bin/env bash
set -euo pipefail
pandoc paper.md -s -o paper.tex --number-sections --citeproc --bibliography=refs.bib --csl=ieee.csl \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=0.92in \
  -V colorlinks=true -V linkcolor=citeblue -V urlcolor=citeblue -V citecolor=citeblue \
  --include-in-header=header.tex
# Keep only the Figure 1 teaser pinned right under the abstract; let all other figures float naturally
python3 -c '
from pathlib import Path
p = Path("paper.tex")
s = p.read_text()
s = s.replace("\\begin{figure}", "\\begin{figure}[H]", 1)
p.write_text(s)
'
tectonic -X compile paper.tex
