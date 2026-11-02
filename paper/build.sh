set -e
pandoc paper.md -s -o paper.tex --number-sections --citeproc --bibliography=refs.bib --csl=ieee.csl \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=0.95in -V colorlinks=true -V linkcolor=piblue -V urlcolor=piblue -V citecolor=piblue \
  --include-in-header=header.tex
# longtable cannot see floats placed on its page later; flush pending floats before each table
sed -i '' 's/^\\begin{longtable}/\\FloatBarrier\n\\begin{longtable}/' paper.tex
# the teaser stays right under the abstract
perl -0pi -e 's/\\begin\{figure\}/\\begin{figure}[H]/' paper.tex
tectonic -X compile paper.tex 2>&1 | grep -i "error\|Overfull \\\\vbox" || true
