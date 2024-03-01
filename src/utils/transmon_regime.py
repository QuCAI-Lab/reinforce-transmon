import matplotlib.pyplot as plt
import scqubits as scq
import numpy as np
import os

'''Plot the transmon regime curve for different ng values.'''

def relative_anharmonicity(ej, ec, ng):
    transmon = scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=31)
    energies = transmon.eigenvals(evals_count=3)
    E01 = energies[1] - energies[0]
    E12 = energies[2] - energies[1]
    return (E12 - E01) / E01


def plot_anharmonicity(labels, x_values, save_dir):
    fig, ax = plt.subplots()
    ax.set_title("Transmon Regime")
    ax.set_xlabel(r"$E_j/E_c$")
    ax.set_ylabel('Relative Anharmonicity')

    for ng in labels:
        y = [relative_anharmonicity(ej, ec=1, ng=ng) for ej in x_values]
        ax.plot(x_values, y, label=f"$ng$ = {ng:.2f}")

    ax.legend()
    plt.savefig(os.path.join(save_dir, "transmon_regime.png"))
    print(f'Transmon regime plot saved at {save_dir}')

def main():
    save_dir = './assets/plots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ng_values = np.linspace(-2, 2, 10)
    ej_values = np.linspace(0, 80, 100)

    plot_anharmonicity(ng_values, ej_values, save_dir)

if __name__ == "__main__":
    main()
