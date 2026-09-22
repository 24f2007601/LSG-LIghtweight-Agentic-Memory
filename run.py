import os

os.environ["LSG_RESULTS_DIR"] = "results2"

from lsg.train import run, run_adaptation_curve, run_rigorous
from lsg.analysis import run_analysis
from plot_results import plot_all_results


def main() -> None:
    os.makedirs("results2", exist_ok=True)

    print("--- 1. Running Model Training ---")
    run()

    print("\n--- 2. Running Few-Shot Adaptation Curve ---")
    run_adaptation_curve()

    print("\n--- 3. Running Rigorous Diagnostics ---")
    run_rigorous()

    print("\n--- 4. Running Error Analysis ---")
    run_analysis()

    print("\n--- 5. Plotting Graph Assets ---")
    plot_all_results(results_dir="results2", assets_dir="assets")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
