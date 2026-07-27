#!/bin/bash
# Build the welfare report PDF: MyST (md -> LaTeX) + tectonic.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report_welfare.md --tex
TEX=_build/exports/report-welfare_tex/report-welfare.tex

sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"
sed -i 's|\\begin{center}\\logo\\end{center}||' "$TEX"
sed -i 's/\\maketitle//' "$TEX"
sed -i 's#\\begin{document}#\\usepackage{amssymb}\\usepackage{float}\\usepackage{capt-of}\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small The welfare cost of robustness}\\fancyhead[R]{\\small 2026-07-20}\\renewcommand{\\headrulewidth}{0.4pt}\\setlength{\\abovedisplayskip}{4pt}\\setlength{\\belowdisplayskip}{4pt}\\setlength{\\parskip}{3pt}\n\\begin{document}#' "$TEX"

# figures appear exactly where referenced (un-float)
python3 - "$TEX" <<'PY2'
import re, sys
p = sys.argv[1]; t = open(p).read()
def unfloat(m):
    region = m.group(0)
    region = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?\s*\\centering", r"\\begin{center}", region)
    region = region.replace("\\end{figure}", "\\end{center}")
    region = region.replace("\\caption*{", "\\captionof{figure}{")
    region = region.replace("\\caption[]{", "\\captionof{figure}{")
    return region
t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", unfloat, t, flags=re.S)
open(p, "w").write(t)
print("figures un-floated")
PY2

( cd _build/exports/report-welfare_tex && tectonic report-welfare.tex )
cp _build/exports/report-welfare_tex/report-welfare.pdf welfare_cost_of_robustness_2026-07-20.pdf
echo "PDF -> $(pwd)/welfare_cost_of_robustness_2026-07-20.pdf"
