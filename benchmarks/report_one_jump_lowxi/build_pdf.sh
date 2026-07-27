#!/bin/bash
# Build the one-jump robustness (xi) sensitivity report PDF: MyST (md -> LaTeX) + tectonic.
# Toolchain (user-local): pip install --user mystmd ; tectonic binary on ~/.local/bin.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report.md --tex
TEX=_build/exports/index_tex/index.tex

# cosmetic: starred headings (drop LaTeX auto-numbering); drop Curvenote logo
sed -i 's/\\section{/\\section*{/g; s/\\subsection{/\\subsection*{/g' "$TEX"
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"

# running header: left = topic, right = date
sed -i 's#\\begin{document}#\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small One-Jump Climate Model: Robustness ($\\xi$) Sensitivity}\\fancyhead[R]{\\small 2026-06-29}\\renewcommand{\\headrulewidth}{0.4pt}\n\\begin{document}#' "$TEX"

# drop the title block + logo — the running header carries topic + date
sed -i 's/\\maketitle//; s|\\begin{center}\\logo\\end{center}||' "$TEX"

( cd _build/exports/index_tex && tectonic index.tex )
cp _build/exports/index_tex/index.pdf one_jump_lowxi_report.pdf
cp one_jump_lowxi_report.pdf ../one_jump_lowxi_report.pdf
echo "PDF -> $(pwd)/one_jump_lowxi_report.pdf  and  benchmarks/one_jump_lowxi_report.pdf"
