#!/bin/bash
# Build the two-capital report PDF from report.md via MyST (md -> LaTeX) + tectonic.
#
# Toolchain (one-time, user-local, no project-space inodes):
#   pip install --user mystmd        # MyST CLI (auto-installs Node to ~/.local/share/myst)
#   tectonic 0.15.0 binary on PATH   # static LaTeX engine (~/.local/bin/tectonic)
#
# Two cosmetic post-edits to MyST's generated LaTeX:
#   1. starred \section*/\subsection* -> drop LaTeX's auto section numbers so only the
#      manual "0./1./2.x" numbers in the headings show (keeps in-text "Section 2.4" refs valid).
#   2. remove the default Curvenote title-page logo.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report.md --tex
TEX=_build/exports/index_tex/index.tex

# 1. suppress LaTeX section numbering (starred headings are unnumbered)
sed -i 's/\\section{/\\section*{/g; s/\\subsection{/\\subsection*{/g' "$TEX"
# 2. drop the Curvenote logo
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"

( cd _build/exports/index_tex && tectonic index.tex )
cp _build/exports/index_tex/index.pdf two_capital_report.pdf
cp two_capital_report.pdf ../two_capital_report.pdf
echo "PDF -> $(pwd)/two_capital_report.pdf  and  benchmarks/two_capital_report.pdf"
