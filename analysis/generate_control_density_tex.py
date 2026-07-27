#!/usr/bin/env python3
"""Generate LaTeX reports for stochastic control density figures."""

from __future__ import annotations

from pathlib import Path


MODEL_SPECS = [
    {
        "label": "One technology jump, direct final technology",
        "folder": Path(
            "output_001/OneTechJump_Pi_1p0_TechIntensityScale_1p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
        ),
        "plot_folder": "OneJump1_controls",
        "events": [
            ("tech0to2", "tech_0_to_2", "Technology jump: pre-tech to final tech"),
            ("damage", "damage_jump", "Damage jump"),
        ],
    },
    {
        "label": "Two-stage technology jump, baseline intensity",
        "folder": Path(
            "output_001/TwoStageTech_LR_warmup_cosine_10e-6,10e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
        ),
        "plot_folder": "TwoJump1_controls",
        "events": [
            ("tech0to1", "tech_0_to_1", "Technology jump: pre-tech to intermediate tech"),
            ("tech1to2", "tech_1_to_2", "Technology jump: intermediate tech to final tech"),
            ("tech0to2", "tech_0_to_2", "Technology jump: pre-tech to final tech"),
            ("damage", "damage_jump", "Damage jump"),
        ],
    },
    {
        "label": "Two-stage technology jump, doubled technology intensity",
        "folder": Path(
            "output_001/TwoStageTech_TechIntensityScale_2p0_LR_warmup_cosine_10e-6,40e-4_128_neurons_32_#HiddenLayer_4_num_iterations1000000"
        ),
        "plot_folder": "TwoJump2_controls",
        "events": [
            ("tech0to1", "tech_0_to_1", "Technology jump: pre-tech to intermediate tech"),
            ("tech1to2", "tech_1_to_2", "Technology jump: intermediate tech to final tech"),
            ("tech0to2", "tech_0_to_2", "Technology jump: pre-tech to final tech"),
            ("damage", "damage_jump", "Damage jump"),
        ],
    },
]

XIS = [
    ("148.6", "148p6"),
    ("0.1", "0p1"),
    ("0.05", "0p05"),
]

FIGURE_GROUPS = [
    ("rate", "i_d", r"$i^d$"),
    ("rate", "i_g", r"$i^g$"),
    ("rate", "i_r", r"$i^r$"),
    ("level", "I_d", r"$I^d$"),
    ("level", "I_g", r"$I^g$"),
    ("level", "I_r", r"$I^r$"),
    ("marginal", "dV_dKd", r"$V_{K^d}$"),
    ("marginal", "dV_dKg", r"$V_{K^g}$"),
    ("marginal", "climate_marginal_cost", r"$-V_Y$"),
]


def latex_header(title: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{float}}
\usepackage{{placeins}}
\usepackage[hidelinks]{{hyperref}}

\captionsetup{{font=small}}
\captionsetup[subfigure]{{font=footnotesize}}
\renewcommand{{\floatpagefraction}}{{0.85}}
\renewcommand{{\topfraction}}{{0.9}}
\renewcommand{{\bottomfraction}}{{0.8}}
\renewcommand{{\textfraction}}{{0.05}}
\setlength{{\textfloatsep}}{{0.6em}}
\setlength{{\floatsep}}{{0.6em}}
\setlength{{\intextsep}}{{0.6em}}

\title{{Stochastic Simulation Density Summary\\{title}}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Capital Dynamics and Control Definitions}}

Dirty and green capital evolve according to
\begin{{equation}}
dK_t^j
=K_t^j\left[\alpha_j+\Gamma_j\log\left(1+\theta_j i_t^j\right)\right]dt
+\sigma_j K_t^j\,dW_t^j,
\qquad j\in\{{d,g\}} .
\end{{equation}}
Let total productive capital be
\begin{{equation}}
K_t=K_t^d+K_t^g,\qquad
Z_t=\frac{{K_t^g}}{{K_t}} .
\end{{equation}}
The normalized controls are investment-to-capital ratios:
\begin{{equation}}
i_t^d=\frac{{I_t^d}}{{K_t^d}},\qquad
i_t^g=\frac{{I_t^g}}{{K_t^g}},\qquad
i_t^r=\frac{{I_t^r}}{{K_t}} .
\end{{equation}}
The corresponding investment levels are
\begin{{equation}}
I_t^d=i_t^dK_t^d,\qquad
I_t^g=i_t^gK_t^g,\qquad
I_t^r=i_t^rK_t .
\end{{equation}}
Knowledge capital satisfies
\begin{{equation}}
dR_t
=\left[-\zeta R_t+\psi_0\left(I_t^r\right)^{{\psi_1}}R_t^{{1-\psi_1}}\right]dt
+\sigma_r R_t\,dW_t^r .
\end{{equation}}
The resource constraint, normalized by total capital, is
\begin{{equation}}
\frac{{C_t}}{{K_t}}
=\left(A_d-i_t^d\right)(1-Z_t)
+\left(A_g-i_t^g\right)Z_t
-i_t^r .
\end{{equation}}

\section{{Marginal Values}}

Because the neural networks use $(\log K,Z)$ instead of $(K^d,K^g)$, the marginal values of the two productive capital stocks follow from the chain rule:
\begin{{align}}
\frac{{\partial V}}{{\partial K^d}}
&=\frac{{1}}{{K}}\left(V_{{\log K}}-ZV_Z\right),\\
\frac{{\partial V}}{{\partial K^g}}
&=\frac{{1}}{{K}}\left(V_{{\log K}}+(1-Z)V_Z\right).
\end{{align}}
These objects measure the marginal value of one additional unit of dirty or green capital.

The positive marginal cost induced by temperature is reported in value units as
\begin{{equation}}
\mathcal{{M}}_Y\equiv -V_Y.
\end{{equation}}
The implementation reconstructs the derivative of the original value function from the transformed value-network output $v$:
\begin{{equation}}
\mathcal{{M}}_Y
=\frac{{\partial\log N(Y)}}{{\partial Y}}-v_Y.
\end{{equation}}
Before the damage jump,
\begin{{equation}}
\frac{{\partial\log N(Y)}}{{\partial Y}}=\lambda_1+\lambda_2Y,
\end{{equation}}
while after the damage jump,
\begin{{equation}}
\frac{{\partial\log N(Y;\lambda_3)}}{{\partial Y}}
=\lambda_1+\lambda_2Y+\lambda_3(Y-\bar y).
\end{{equation}}
Thus, the climate-cost density incorporates both the value-network response to temperature and the additional curvature revealed by the damage jump.

\paragraph{{Reading the density figures.}}
Each figure compares the empirical kernel density immediately before a jump with the empirical kernel density immediately after that jump, using stochastic paths initialized at $Y_0=1.2$.
The rate figures report $(i^d,i^g,i^r)$, the level figures report $(I^d,I^g,I^r)$, the marginal-value figures report $(V_{{K^d}},V_{{K^g}},-V_Y)$, and the value-function figures report $V$ itself.
The parameter $\xi$ indexes robustness concerns; smaller $\xi$ corresponds to stronger model-uncertainty aversion.
The figures report $\xi\in\{{0.05,0.1,\infty\}}$, where $\xi=148.6$ is used as the uncertainty-neutral benchmark and is displayed first.

\section{{Density Figures}}

"""


def tex_escape_text(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def xi_label(xi_value: str) -> str:
    if xi_value == "148.6":
        return r"$\xi=\infty$"
    return rf"$\xi={xi_value}$"


def figure_path(plot_folder: str, xi_value: str, group: str, variable: str, event_name: str) -> str:
    return (
        f"{plot_folder}/paths_ξ_{xi_value}/ControlDensities/"
        f"{group}_pre-post/{variable}_kde_{event_name}_pre-post.png"
    )


def compact_group_figure(
    event_token: str,
    event_name: str,
    event_title: str,
    plot_folder: str,
    group: str,
) -> str:
    if group == "rate":
        variables = [("i_d", r"$i^d$"), ("i_g", r"$i^g$"), ("i_r", r"$i^r$")]
        group_title = "Investment rates"
    elif group == "level":
        variables = [("I_d", r"$I^d$"), ("I_g", r"$I^g$"), ("I_r", r"$I^r$")]
        group_title = "Investment levels"
    elif group == "marginal":
        variables = [
            ("dV_dKd", r"$\partial V/\partial K^d$"),
            ("dV_dKg", r"$\partial V/\partial K^g$"),
            ("climate_marginal_cost", r"$-\partial V/\partial Y$"),
        ]
        group_title = "Marginal values and climate cost"
    else:
        raise ValueError(group)
    column_spec = "ccc"
    image_width = "0.315\\textwidth"
    image_height = "0.145\\textheight"

    header_cells = " & ".join(rf"\textbf{{{label}}}" for _variable, label in variables)
    rows = [
        r"\begin{figure}[p]",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        header_cells + r" \\[-0.25em]",
    ]

    for xi_value, _xi_token in XIS:
        figure_cells = []
        for variable, _label in variables:
            fname = figure_path(plot_folder, xi_value, group, variable, event_name)
            figure_cells.append(
                rf"\includegraphics[width={image_width},height={image_height},keepaspectratio]{{{fname}}}"
            )
        rows.append(" & ".join(figure_cells) + r" \\[-0.2em]")
        rows.append(rf"\multicolumn{{{len(variables)}}}{{c}}{{\small {xi_label(xi_value)}}} \\[0.25em]")

    rows.extend(
        [
            r"\end{tabular}",
            rf"\caption{{{group_title} around {event_title.lower()}. Rows correspond to robustness specifications; the top row is the uncertainty-neutral benchmark. Each panel compares the empirical density immediately before the jump with the empirical density immediately after the jump.}}",
            rf"\label{{fig:{event_token}:{group}}}",
            r"\end{figure}",
            "",
        ]
    )
    return "\n".join(rows)


def value_figure(
    event_token: str,
    event_name: str,
    event_title: str,
    plot_folder: str,
) -> str:
    rows = [
        r"\begin{figure}[p]",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{ccc}",
    ]

    figure_cells = []
    label_cells = []
    for xi_value, _xi_token in XIS:
        fname = figure_path(plot_folder, xi_value, "value", "V", event_name)
        figure_cells.append(
            rf"\includegraphics[width=0.315\textwidth,height=0.42\textheight,keepaspectratio]{{{fname}}}"
        )
        label_cells.append(rf"\small {xi_label(xi_value)}")
    rows.append(" & ".join(figure_cells) + r" \\[-0.2em]")
    rows.append(" & ".join(label_cells) + r" \\[0.35em]")

    rows.extend(
        [
            r"\end{tabular}",
            rf"\caption{{Value-function densities around {event_title.lower()}. Each panel compares the empirical distribution of $V$ immediately before the jump with its distribution immediately after the jump.}}",
            rf"\label{{fig:{event_token}:value}}",
            r"\end{figure}",
            "",
        ]
    )
    return "\n".join(rows)


def event_section(event_token: str, event_name: str, event_title: str, plot_folder: str) -> str:
    return "\n".join(
        [
            rf"\subsection{{{event_title}}}",
            (
                "The figures in this subsection condition on stochastic paths in which this jump occurs. "
                "They compare investment rates, investment levels, marginal values, and the value function immediately before the jump "
                "with the distribution immediately after the jump. "
                "A visible horizontal movement between the before- and after-jump densities indicates a policy or valuation response to the information revealed by the jump."
            ),
            "",
            compact_group_figure(event_token, event_name, event_title, plot_folder, "rate"),
            compact_group_figure(event_token, event_name, event_title, plot_folder, "level"),
            compact_group_figure(event_token, event_name, event_title, plot_folder, "marginal"),
            value_figure(event_token, event_name, event_title, plot_folder),
            r"\FloatBarrier",
            "",
        ]
    )


def latex_body(spec: dict) -> str:
    chunks = [latex_header(spec["label"])]
    chunks.append(
        r"""\paragraph{Organization.}
The figures below follow the layout used in jump-response density plots.
For each jump event, the first figure reports investment rates, the second reports investment levels, the third reports the two capital marginal values and the marginal climate cost, and the fourth reports the value function.
Within each figure, rows report different robustness specifications and columns report the relevant policy or valuation object.
Comparing the before- and after-jump densities shows the discontinuous change at the realized jump time.
Comparing rows shows how the same jump response changes as robustness concerns weaken.

"""
    )
    for event_token, event_name, event_title in spec["events"]:
        chunks.append(event_section(event_token, event_name, event_title, spec["plot_folder"]))

    chunks.append(
        r"""\section{Notes}

Investment rates and levels are shown separately because the same rate can imply a different investment quantity when the associated capital stock changes.
The climate marginal cost $-V_Y$ is measured in value units.
It is not converted into consumption units or monetary damages in these figures.

\end{document}
"""
    )
    return "\n".join(chunks)


def main() -> None:
    for spec in MODEL_SPECS:
        folder = spec["folder"]
        if not folder.exists():
            raise FileNotFoundError(folder)
        tex_path = folder / "control_density_report.tex"
        tex_path.write_text(latex_body(spec), encoding="utf-8")
        print(f"{tex_path}: wrote report using {spec['plot_folder']}")


if __name__ == "__main__":
    main()
