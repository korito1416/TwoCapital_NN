#!/bin/bash
# Build the one-jump adjcost report PDF from report.md via MyST (md -> LaTeX) + tectonic.
# Toolchain (user-local): pip install --user mystmd ; tectonic binary on ~/.local/bin.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report.md --tex
TEX=_build/exports/index_tex/index.tex

# cosmetic: starred headings (use the manual "N." numbers, drop LaTeX auto-numbering); drop Curvenote logo
sed -i 's/\\section{/\\section*{/g; s/\\subsection{/\\subsection*{/g' "$TEX"
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"

# running header: left = topic, right = date (fancyhdr injected before \begin{document})
sed -i 's#\\begin{document}#\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small One-Jump Climate Model: Adjustment-Cost Sensitivity}\\fancyhead[R]{\\small 2026-06-27}\\renewcommand{\\headrulewidth}{0.4pt}\n\\begin{document}#' "$TEX"

# drop the title block (\maketitle) and logo — the running header already carries topic + date
sed -i 's/\\maketitle//; s|\\begin{center}\\logo\\end{center}||' "$TEX"

( cd _build/exports/index_tex && tectonic index.tex )
cp _build/exports/index_tex/index.pdf one_jump_adjcost_report.pdf
cp one_jump_adjcost_report.pdf ../one_jump_adjcost_report.pdf
echo "PDF -> $(pwd)/one_jump_adjcost_report.pdf  and  benchmarks/one_jump_adjcost_report.pdf"
