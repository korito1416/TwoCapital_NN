#!/bin/bash
# Build the warm-start RCT report PDF: MyST (md -> LaTeX) + tectonic.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

printf 'y\n' | myst build report.md --tex
TEX=_build/exports/report_tex/report.tex

# drop Curvenote logo/link and the title page; running header carries topic + date
sed -i 's|\\href{https://curvenote.com}{\\includegraphics\[width=2cm\]{curvenote.png}}||' "$TEX"
sed -i 's|\\begin{center}\\logo\\end{center}||' "$TEX"
sed -i 's/\\maketitle//' "$TEX"
sed -i 's#\\begin{document}#\\usepackage{amssymb}\\usepackage{pdflscape}\\usepackage{float}\\usepackage{capt-of}\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\fancyhead[L]{\\small Two contradictions with the model}\\fancyhead[R]{\\small 2026-07-20}\\renewcommand{\\headrulewidth}{0.4pt}\\setlength{\\abovedisplayskip}{3pt}\\setlength{\\belowdisplayskip}{3pt}\\setlength{\\abovedisplayshortskip}{1pt}\\setlength{\\belowdisplayshortskip}{2pt}\\setlength{\\parskip}{2pt}\n\\begin{document}\\small#' "$TEX"

# each wide 1xN strip figure gets its own landscape page (non-floating: figure envs
# inside landscape defer to the next rotated page and leave a blank one)
python3 - "$TEX" <<'PY'
import re, sys
p = sys.argv[1]; t = open(p).read()
# MyST truncates figure basenames to 20 chars before the content hash
keys = ()
for key in keys:
    blocks = [m for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", t, re.S)
              if key in m.group(0)]
    assert len(blocks) == 1, (key, len(blocks))
    b = blocks[0]
    region = b.group(0)
    region = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?\s*\\centering", r"\\begin{center}", region)
    region = region.replace("\\end{figure}", "\\end{center}")
    region = region.replace("\\caption*{", "\\captionof{figure}{")
    region = region.replace("\\caption[]{", "\\captionof{figure}{")
    region = region.replace("\\includegraphics[width=0.7\\linewidth]", "\\includegraphics[width=\\linewidth]")
    t = t[:b.start()] + "\\begin{landscape}\n" + region + "\n\\end{landscape}" + t[b.end():]
open(p, "w").write(t)
print("landscape-wrapped:", ", ".join(keys))
PY

# tables: equal-width p{} columns force needless wrapping -> natural-width l columns
python3 - "$TEX" <<'PY3'
import re, sys
p = sys.argv[1]; t = open(p).read()
t = re.sub(r"\\begin\{tabular\}\{((?:p\{[^{}]*\})+)\}",
           lambda m: "\\begin{tabular}{" + "l" * m.group(1).count("p{") + "}", t)
open(p, "w").write(t)
print("table columns -> natural width")
PY3

# remaining floats -> non-floating (figures appear exactly where referenced)
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
print("all remaining figures un-floated")
PY2

( cd _build/exports/report_tex && tectonic report.tex )
cp _build/exports/report_tex/report.pdf belief_defect_2026-07-20.pdf
echo "PDF -> $(pwd)/belief_defect_2026-07-20.pdf"
