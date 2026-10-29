import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_estimates
import plot_flatness_walk
import plot_simulation
import plot_spectra

MODULES = [plot_estimates, plot_spectra, plot_flatness_walk, plot_simulation]


def main():
    for module in MODULES:
        print(module.__name__)
        module.main()


if __name__ == '__main__':
    main()
