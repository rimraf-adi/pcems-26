#!/bin/bash
echo "Building Template 1 (Custom Single-Column Arial): norm.pdf..."
xelatex -interaction=nonstopmode norm.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode norm.tex > /dev/null 2>&1

echo "Building Template 2 (Springer LNCS): springer.pdf..."
xelatex -interaction=nonstopmode springer.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode springer.tex > /dev/null 2>&1

echo "Build complete!"
echo "Generated PDFs:"
ls -lh norm.pdf springer.pdf
