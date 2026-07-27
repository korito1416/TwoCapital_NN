import matplotlib.pyplot as plt
import sys
import os
import shutil
import glob
import numpy as np

FolderName = sys.argv[1]

def read_values(file_path):
    return np.loadtxt(file_path, ndmin=1)

# Original data folder
data_dir = os.path.join(FolderName, 'SimulationDeterministic')

# New plot folder
plot_dir = os.path.join(FolderName, 'SimulationDeterministicPlot')
os.makedirs(plot_dir, exist_ok=True)

def xi_label(xi):
    if abs(xi - 148.6) < 1e-10:
        return r'$\xi$ = $\infty$'
    return rf'$\xi$ = {xi:g}'

def xi_folder_name(xi):
    return f"SimulationOutputs_ξ_{xi:.3f}"

xi_values = [
    float(value)
    for value in os.environ.get("SIMULATION_XIS", "0.01,0.05,0.1,148.6").split(",")
    if value.strip()
]
colors = ['purple', 'orange', 'b', 'r', 'g', 'k', 'c', 'm']
xi_configs = [
    (xi_folder_name(xi), colors[i % len(colors)], xi_label(xi))
    for i, xi in enumerate(xi_values)
]

def write_conditional_jump_probabilities(xi_folder):
    """Derive first-jump probabilities and densities under both conditionings."""
    damage_path = os.path.join(xi_folder, "dmg_jump_subdensity.txt")
    tech_path = os.path.join(xi_folder, "tech_jump_subdensity.txt")
    if not os.path.exists(damage_path) or not os.path.exists(tech_path):
        damage_path = os.path.join(xi_folder, "dmg_jump_density.txt")
        tech_path = os.path.join(xi_folder, "tech_jump_density.txt")
    if not os.path.exists(damage_path) or not os.path.exists(tech_path):
        return

    damage = read_values(damage_path)
    tech = read_values(tech_path)
    count = min(damage.size, tech.size)
    damage = damage[:count]
    tech = tech[:count]
    total = damage + tech
    type_damage = np.divide(
        damage, total, out=np.zeros_like(damage), where=total > 0
    )
    type_tech = np.divide(
        tech, total, out=np.zeros_like(tech), where=total > 0
    )
    np.savetxt(os.path.join(xi_folder, "first_jump_type_dmg_prob.txt"), type_damage)
    np.savetxt(os.path.join(xi_folder, "first_jump_type_tech_prob.txt"), type_tech)

    damage_cumulative_path = os.path.join(xi_folder, "dmg_jump_prob.txt")
    tech_cumulative_path = os.path.join(xi_folder, "tech_jump_prob.txt")
    if not os.path.exists(damage_cumulative_path) or not os.path.exists(tech_cumulative_path):
        return
    damage_cumulative = read_values(damage_cumulative_path)[:count]
    tech_cumulative = read_values(tech_cumulative_path)[:count]
    horizon_probability = damage_cumulative[-1] + tech_cumulative[-1]
    if horizon_probability <= 0:
        conditional_damage_cumulative = np.zeros_like(damage_cumulative)
        conditional_tech_cumulative = np.zeros_like(tech_cumulative)
        conditional_damage_density = np.zeros_like(damage)
        conditional_tech_density = np.zeros_like(tech)
    else:
        conditional_damage_cumulative = damage_cumulative / horizon_probability
        conditional_tech_cumulative = tech_cumulative / horizon_probability
        conditional_damage_density = damage / horizon_probability
        conditional_tech_density = tech / horizon_probability
    np.savetxt(
        os.path.join(xi_folder, "conditional_dmg_jump_prob.txt"),
        conditional_damage_cumulative,
    )
    np.savetxt(
        os.path.join(xi_folder, "conditional_tech_jump_prob.txt"),
        conditional_tech_cumulative,
    )
    np.savetxt(
        os.path.join(xi_folder, "conditional_dmg_jump_density.txt"),
        conditional_damage_density,
    )
    np.savetxt(
        os.path.join(xi_folder, "conditional_tech_jump_density.txt"),
        conditional_tech_density,
    )
    t_path = os.path.join(xi_folder, "t.txt")
    if os.path.exists(t_path):
        t = read_values(t_path)[:count]
        dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
        damage_mass = float(np.sum(conditional_damage_density[:count] * dt))
        tech_mass = float(np.sum(conditional_tech_density[:count] * dt))
        total_mass = damage_mass + tech_mass
        with open(os.path.join(xi_folder, "first_jump_density_accounting.txt"), "w") as f:
            f.write(f"horizon_first_jump_probability: {horizon_probability:.12g}\n")
            f.write(f"damage_conditional_mass: {damage_mass:.12g}\n")
            f.write(f"technology_conditional_mass: {tech_mass:.12g}\n")
            f.write(f"total_conditional_mass: {total_mass:.12g}\n")
            f.write(f"damage_percent: {100.0 * damage_mass:.12g}\n")
            f.write(f"technology_percent: {100.0 * tech_mass:.12g}\n")
            f.write(f"integration_error: {total_mass - 1.0:.12g}\n")


for subfolder, _, _ in xi_configs:
    write_conditional_jump_probabilities(os.path.join(data_dir, subfolder))

def write_consumption_output_ratio(xi_folder):
    """Derive total consumption, output, and C/Y for legacy simulations."""
    c_per_capital_path = os.path.join(xi_folder, "c.txt")
    capital_path = os.path.join(xi_folder, "K.txt")
    output_path = os.path.join(xi_folder, "Output.txt")
    if not os.path.exists(output_path):
        output_path = os.path.join(xi_folder, "y_consumption.txt")
    if not all(os.path.exists(path) for path in (c_per_capital_path, capital_path, output_path)):
        return

    c_per_capital = read_values(c_per_capital_path)
    capital = read_values(capital_path)
    output = read_values(output_path)
    count = min(c_per_capital.size, capital.size, output.size)
    consumption = c_per_capital[:count] * capital[:count]
    output = output[:count]
    ratio = np.divide(
        consumption, output, out=np.zeros_like(consumption), where=output != 0
    )
    np.savetxt(os.path.join(xi_folder, "C.txt"), consumption)
    np.savetxt(os.path.join(xi_folder, "Output.txt"), output)
    np.savetxt(os.path.join(xi_folder, "ConsumptionOutputRatio.txt"), ratio)
    plt.figure()
    plt.plot(read_values(os.path.join(xi_folder, "t.txt"))[:count], ratio * 100.0)
    plt.xlabel("Years")
    plt.ylabel("Percent")
    plt.title("Consumption as % of Output (C/Y)")
    plt.tight_layout()
    plt.savefig(os.path.join(xi_folder, "ConsumptionOutputRatio.png"))
    plt.close()


for subfolder, _, _ in xi_configs:
    write_consumption_output_ratio(os.path.join(data_dir, subfolder))

title_map = {
    "E":               "Emission",
    "RD":              r'R&D investment as % of Output $(I_k/Y)$',
    "I_d":             r'Dirty investment $(I_d)$',
    "DirtyInvestment": r'Dirty investment as % of Output $(I_d/Y)$',
    "I_g":             r'Green investment $(I_g)$',
    "GreenInvestment": r'Green investment as % of Output $(I_g/Y)$',
    "ConsumptionOutputRatio": r'Consumption as % of Output $(C/Y)$',
    "tech_jump_prob":  "Technology first-jump cumulative incidence",
    "dmg_jump_prob":   "Damage first-jump cumulative incidence",
    "any_jump_prob":   "Probability that any first jump has occurred",
    "tech_jump_density": "Technology first-jump density conditional on first jump by horizon",
    "dmg_jump_density":  "Damage first-jump density conditional on first jump by horizon",
    "tech_jump_subdensity": "Technology first-jump subdensity",
    "dmg_jump_subdensity":  "Damage first-jump subdensity",
    "conditional_tech_jump_density": "Technology density conditional on a first jump by year 60",
    "conditional_dmg_jump_density": "Damage density conditional on a first jump by year 60",
    "conditional_tech_jump_prob": "Conditional cumulative probability: technology first jump",
    "conditional_dmg_jump_prob": "Conditional cumulative probability: damage first jump",
    "first_jump_type_tech_prob": "Probability first jump is technology, conditional on its time",
    "first_jump_type_dmg_prob": "Probability first jump is damage, conditional on its time",
    "tech_jump_intensity": "Grouped distorted technology jump intensity",
    "dmg_jump_intensity":  "Grouped distorted damage jump intensity",
    "total_jump_intensity": "Total grouped distorted jump intensity",
}

pct_names = {"RD", "DirtyInvestment", "GreenInvestment", "ConsumptionOutputRatio"}
density_area_names = {
    "tech_jump_density",
    "dmg_jump_density",
    "conditional_tech_jump_density",
    "conditional_dmg_jump_density",
}

names = ["E", "RD", "I_d", "DirtyInvestment", "I_g", "GreenInvestment",
         "ConsumptionOutputRatio",
         "tech_jump_prob", "dmg_jump_prob", "any_jump_prob",
         "tech_jump_density", "dmg_jump_density",
         "tech_jump_subdensity", "dmg_jump_subdensity",
         "conditional_tech_jump_density", "conditional_dmg_jump_density",
         "conditional_tech_jump_prob", "conditional_dmg_jump_prob",
         "first_jump_type_tech_prob", "first_jump_type_dmg_prob",
         "tech_jump_intensity", "dmg_jump_intensity", "total_jump_intensity"]

# ── 1. Plot figures and save them in SimulationDeterministicPlot ──
for txt_name in names:
    plt.figure(figsize=(10, 6))
    plotted = False
    for subfolder, color, label in xi_configs:
        xi_folder = os.path.join(data_dir, subfolder)
        file_path = os.path.join(data_dir, subfolder, f'{txt_name}.txt')
        t_path = os.path.join(xi_folder, 't.txt')
        if not os.path.exists(file_path) or not os.path.exists(t_path):
            continue
        values = read_values(file_path)
        times = read_values(t_path)
        count = min(values.size, times.size)
        values = values[:count]
        times = times[:count]
        if txt_name in pct_names:
            values = values * 100
        curve_label = label
        if txt_name in density_area_names and count > 1:
            dt = float(np.mean(np.diff(times)))
            area = float(np.sum(values * dt))
            curve_label = f"{label} (area={area:.4f})"
        plt.plot(times, values, color=color, label=curve_label, linewidth=5)
        plotted = True

    if not plotted:
        plt.close()
        continue

    plt.title(title_map.get(txt_name, txt_name), fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Percent' if txt_name in pct_names else 'Value', fontsize=16)
    if txt_name in {
        "conditional_tech_jump_prob",
        "conditional_dmg_jump_prob",
        "first_jump_type_tech_prob",
        "first_jump_type_dmg_prob",
    }:
        plt.ylim(0, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f'{txt_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")

# ── 2. Copy all figures from each xi subfolder to SimulationDeterministicPlot ──
for subfolder, color, label in xi_configs:
    src_folder = os.path.join(data_dir, subfolder)
    dest_folder = os.path.join(plot_dir, subfolder)
    os.makedirs(dest_folder, exist_ok=True)

    for fig_file in glob.glob(os.path.join(src_folder, '*.png')):
        shutil.copy2(fig_file, dest_folder)

with open(os.path.join(plot_dir, "first_jump_density_accounting.txt"), "w") as summary:
    summary.write("xi,horizon_first_jump_probability,damage_percent,technology_percent,total_conditional_mass,integration_error\n")
    for subfolder, _, label in xi_configs:
        accounting_path = os.path.join(data_dir, subfolder, "first_jump_density_accounting.txt")
        if not os.path.exists(accounting_path):
            continue
        values = {}
        with open(accounting_path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                try:
                    values[key.strip()] = float(value.strip())
                except ValueError:
                    pass
        summary.write(
            f"{label},"
            f"{values.get('horizon_first_jump_probability', np.nan):.12g},"
            f"{values.get('damage_percent', np.nan):.12g},"
            f"{values.get('technology_percent', np.nan):.12g},"
            f"{values.get('total_conditional_mass', np.nan):.12g},"
            f"{values.get('integration_error', np.nan):.12g}\n"
        )
        print(
            f"First-jump density accounting {label}: "
            f"damage={values.get('damage_percent', np.nan):.6f}%, "
            f"technology={values.get('technology_percent', np.nan):.6f}%, "
            f"total_mass={values.get('total_conditional_mass', np.nan):.12f}, "
            f"integration_error={values.get('integration_error', np.nan):.3e}"
        )
