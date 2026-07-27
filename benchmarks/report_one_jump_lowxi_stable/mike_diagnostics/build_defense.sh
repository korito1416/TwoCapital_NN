#!/bin/bash
# Build the Mike-defense report PDF: MyST (md -> LaTeX) + tectonic.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report_mike_defense.md --tex
TEX="../_build/exports/report-mike-defense_tex/report-mike-defense.tex"

# drop Curvenote logo + center-logo block
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"
sed -i 's|\\begin{center}\\logo\\end{center}||' "$TEX"

# match the Lars report format: NO title page — strip \maketitle (running header carries topic+date)
sed -i 's/\\maketitle//' "$TEX"

# running header: left = topic (no bare &), right = date
grep -q "fancyhdr" "$TEX" || sed -i 's#\\begin{document}#\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small One-Jump Climate --- Worst-Case Distortions}\\fancyhead[R]{\\small 2026-07-01}\\renewcommand{\\headrulewidth}{0.4pt}\n\\begin{document}#' "$TEX"

( cd ../_build/exports/report-mike-defense_tex && tectonic report-mike-defense.tex )
cp ../_build/exports/report-mike-defense_tex/report-mike-defense.pdf ./report_mike_defense.pdf
echo "PDF -> $(pwd)/report_mike_defense.pdf"
