#!/bin/bash
# Build the low-ξ structural-defect report PDF: MyST (md -> LaTeX) + tectonic.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report.md --tex
TEX=_build/exports/report_tex/report.tex

# drop Curvenote logo/link AND the title block/title page — the running header carries topic + date
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"
sed -i 's|\\begin{center}\\logo\\end{center}||' "$TEX"
sed -i 's/\\maketitle//' "$TEX"

# running header: left = topic, right = date
sed -i 's#\\begin{document}#\\usepackage{amssymb}\\usepackage{amsmath}\\usepackage{setspace}\\setstretch{1.18}\\usepackage{amsthm}\\theoremstyle{plain}\\newtheorem{thm}{Theorem}\\newtheorem{lem}{Lemma}\\newtheorem{prop}{Proposition}\\newtheorem{coro}{Corollary}\\theoremstyle{definition}\\newtheorem{defn}{Definition}\\newtheorem{assu}{Assumption}\\usepackage{pdflscape}\\usepackage{tikz}\\usetikzlibrary{positioning,arrows.meta,calc}\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small Uniqueness of the HJB residual-minimization solution}\\fancyhead[R]{\\small 2026-07-09}\\renewcommand{\\headrulewidth}{0.4pt}\n\\begin{document}#' "$TEX"

( cd _build/exports/report_tex && tectonic report.tex )
cp _build/exports/report_tex/report.pdf solution_uniqueness_2026-07-09.pdf
echo "PDF -> $(pwd)/solution_uniqueness_2026-07-09.pdf"
