#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/paper"

echo "Building Template 1 (Custom Single-Column Arial): paper/norm.pdf..."
xelatex -interaction=nonstopmode norm.tex > /dev/null 2>&1
bibtex norm > /dev/null 2>&1
xelatex -interaction=nonstopmode norm.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode norm.tex > /dev/null 2>&1

echo "Building Template 2 (Springer LNCS): paper/springer.pdf..."
xelatex -interaction=nonstopmode springer.tex > /dev/null 2>&1
bibtex springer > /dev/null 2>&1
xelatex -interaction=nonstopmode springer.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode springer.tex > /dev/null 2>&1

echo "Cleaning up auxiliary compilation files..."
rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg missfont.log

echo "Build complete!"
echo "Generated PDFs:"
ls -lh norm.pdf springer.pdf
