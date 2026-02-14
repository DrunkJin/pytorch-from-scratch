"""Train all models sequentially and save result plots to assets/."""

import time
import importlib


def main():
    print("=" * 60)
    print("  pytorch-from-scratch — Training All Models")
    print("=" * 60)

    modules = [
        ("01 Linear Regression", "01_linear_regression"),
        ("02 Logistic Regression", "02_logistic_regression"),
        ("03 MLP", "03_mlp"),
        ("04 CNN", "04_cnn"),
        ("05 RNN/LSTM", "05_rnn"),
        ("06 Autoencoder", "06_autoencoder"),
        ("07 GAN", "07_gan"),
        ("08 Transformer", "08_transformer"),
    ]

    results = {}
    for name, mod_name in modules:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        start = time.time()
        mod = importlib.import_module(mod_name)
        mod.train()
        elapsed = time.time() - start
        results[name] = elapsed
        print(f"  Done in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for name, t in results.items():
        print(f"  {name:<25} {t:>8.1f}s")
    total = sum(results.values())
    print(f"  {'─' * 35}")
    print(f"  {'Total':<25} {total:>8.1f}s")
    print(f"\n  All plots saved to assets/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
