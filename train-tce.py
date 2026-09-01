from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
from tce.calculator import TCECalculator
import matplotlib.pyplot as plt


def main():

    configurations = []
    for path in Path("samples").glob("*/relax/OUTCAR"):

        trajectory = io.read(path, format="vasp-out", index=slice(None))

        initial_config = trajectory[0]

        positions = initial_config.get_positions()
        ordering = np.lexsort((
            positions[:, 0],
            positions[:, 1],
            positions[:, 2]
        ))

        initial_config = initial_config[ordering]
        initial_config.calc = SinglePointCalculator(atoms=initial_config, energy=trajectory[-1].get_potential_energy())
        configurations.append(initial_config)

    train, test = train_test_split(configurations, test_size=0.2, random_state=0)

    calc = TCECalculator(
        neighbor_cutoffs=[2.5, 3.25, 4.0, 4.5, 5.0, 5.5],
        many_body_features=[
            (0, 0, 1),
            (1, 1, 1),
            (0, 1, 2),
            (1, 2, 2),
            (0, 0, 0, 0, 1, 1),
            (0, 0, 0, 1, 1, 1),
            (0, 0, 1, 1, 1, 2),
            #(1, 1, 1, 1, 1, 5)
        ],
        species=["W", "Mo", "Ta", "S", "Se", "Te"],
        neighbor_tolerance=0.05,
        models={
            "energy": Pipeline([
                ("rescale", StandardScaler()), 
                ("fit", LassoCV(max_iter=10_000))
            ])
        }
    ).train(train)

    actual = []
    predicted = []
    for t in train:
        actual.append(t.get_potential_energy() / len(t))
        predicted.append(calc.get_potential_energy(t) / len(t))
    plt.scatter(actual, predicted, edgecolor="black", label="train", zorder=7, marker="o")

    actual = []
    predicted = []
    for t in test:
        actual.append(t.get_potential_energy() / len(t))
        predicted.append(calc.get_potential_energy(t) / len(t))
    plt.scatter(actual, predicted, edgecolor="black", label="test", zorder=8, marker="s")

    xlim = plt.gca().get_xlim()
    plt.plot(xlim, xlim, linestyle="--", color="black", zorder=6)
    plt.grid()
    plt.xlabel("DFT energy (eV / atom)")
    plt.ylabel("TCE energy (eV / atom)")
    plt.legend()
    plt.savefig("parity.pdf", bbox_inches="tight")

    calc.save("tmd-tce.pkl")


if __name__ == "__main__":
    
    main()
