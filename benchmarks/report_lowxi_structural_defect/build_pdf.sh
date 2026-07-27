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
sed -i 's#\\begin{document}#\\usepackage{amssymb}\\usepackage{pdflscape}\\usepackage{float}\\usepackage{capt-of}\\usepackage{tikz}\\usetikzlibrary{positioning,arrows.meta,calc}\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small Same solution-level loss, divergent policies}\\fancyhead[R]{\\small 2026-07-11}\\renewcommand{\\headrulewidth}{0.4pt}\n\\begin{document}#' "$TEX"

# rotate the two full-page loss figures (Lars reads these) onto landscape pages
python3 - "$TEX" <<'PY'
import re,sys
p=sys.argv[1]; t=open(p).read()
keys=("K5_loss_composition","K5b_loss_magnitude")
blocks=[m for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", t, re.S)
        if any(k in m.group(0) for k in keys)]
if len(blocks)==2 and blocks[0].end()<=blocks[1].start():
    region=t[blocks[0].start():blocks[1].end()]
    # non-floating inside landscape: figure envs defer to the next rotated page and leave a blank one
    region=re.sub(r"\\begin\{figure\}(\[[^\]]*\])?\s*\\centering", r"\\begin{center}", region)
    region=region.replace("\\end{figure}","\\end{center}")
    region=region.replace("\\caption[]{","\\captionof{figure}{")
    t=t[:blocks[0].start()]+"\\begin{landscape}\n"+region+"\n\\end{landscape}"+t[blocks[1].end():]
open(p,"w").write(t)
PY

( cd _build/exports/report_tex && tectonic report.tex )
cp _build/exports/report_tex/report.pdf solution_identification_2026-07-11.pdf
echo "PDF -> $(pwd)/solution_identification_2026-07-11.pdf"
