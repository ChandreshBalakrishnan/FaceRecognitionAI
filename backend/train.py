"""Fit the boredom model to your own scrolling.

    python train.py            # fit, report, and save if it is actually good
    python train.py --dry-run  # report only
    python train.py --stats    # just show how much data you have

It will refuse to save a model that is not measurably better than guessing, and
it reports how it compares with the hand-tuned fusion it would replace. A model
that loses to the hand-tuned rules is not worth having, and you should be told
so rather than quietly shipped it.
"""

import argparse
import json

import numpy as np

from learning import EXAMPLES_PATH, FEATURE_NAMES, WEIGHTS_PATH, LogisticModel
from server import load_config

MIN_EXAMPLES = 60
MIN_PER_CLASS = 20
MIN_AUC = 0.60


def load_examples(path=EXAMPLES_PATH):
    xs, ys = [], []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    x, y = row["x"], row["y"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if len(x) == len(FEATURE_NAMES) and y in (0, 1):
                    xs.append(x)
                    ys.append(y)
    except OSError:
        pass
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def auc(y_true, scores):
    """Area under the ROC curve, via rank statistics. 0.5 is coin-flipping."""
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positives, negatives]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum = ranks[: len(positives)].sum()
    return (rank_sum - len(positives) * (len(positives) + 1) / 2) / (
        len(positives) * len(negatives)
    )


def fit(x, y, l2=1.0, iterations=4000, learning_rate=0.1):
    """Plain gradient descent. The problem is tiny; nothing fancier is warranted."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    z = (x - mean) / std

    weights = np.zeros(z.shape[1])
    bias = 0.0
    n = len(z)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ weights + bias, -30, 30)))
        error = p - y
        weights -= learning_rate * ((z.T @ error) / n + l2 * weights / n)
        bias -= learning_rate * error.mean()
    return weights, bias, mean, std


def soft_or_scores(x, config):
    """What the hand-tuned fusion would have said, for comparison."""
    gains = config.get("fusion_gains", {})
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    parts = [
        (x[:, index["attention_loss"]], gains.get("attention", 0.85)),
        (x[:, index["drowsiness"]], gains.get("drowsiness", 0.55)),
        (x[:, index["negative"]], gains.get("emotion", 0.85)),
    ]
    product = np.ones(len(x))
    for values, gain in parts:
        product *= 1.0 - np.clip(values, 0, 1) * gain
    return 1.0 - product


def main():
    parser = argparse.ArgumentParser(description="Fit the boredom model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    x, y = load_examples()
    bored, engaged = int((y == 1).sum()), int((y == 0).sum())

    print(f"examples        {len(y)}  ({bored} skipped early, {engaged} watched through)")
    if args.stats:
        return 0

    if len(y) < MIN_EXAMPLES or bored < MIN_PER_CLASS or engaged < MIN_PER_CLASS:
        print(
            f"\nNot enough yet. Need at least {MIN_EXAMPLES} examples with "
            f"{MIN_PER_CLASS} of each kind.\n"
            "Keep scrolling normally with the service running - every video you skip\n"
            "fast or watch through adds one. An evening of normal use is usually plenty."
        )
        return 1

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    x, y = x[order], y[order]
    cut = int(len(y) * 0.75)
    x_train, y_train, x_test, y_test = x[:cut], y[:cut], x[cut:], y[cut:]

    weights, bias, mean, std = fit(x_train, y_train, l2=args.l2)
    model = LogisticModel(weights, bias, mean, std)

    test_scores = np.array([model.predict(row) for row in x_test])
    test_auc = auc(y_test, test_scores)
    accuracy = ((test_scores > 0.5).astype(float) == y_test).mean()

    baseline_auc = auc(y_test, soft_or_scores(x_test, load_config()))

    print(f"\ntrained on {len(y_train)}, tested on {len(y_test)}")
    print(f"  learned model   AUC {test_auc:.3f}   accuracy {accuracy:.0%}")
    print(f"  hand-tuned rules AUC {baseline_auc:.3f}   <- what it would replace")

    print("\nwhat it learned (positive = pushes toward 'you will skip this'):")
    for name, weight in sorted(
        zip(FEATURE_NAMES, weights), key=lambda kv: -abs(kv[1])
    ):
        bar = "#" * int(min(20, abs(weight) * 10))
        print(f"  {name:16s} {weight:+.3f}  {bar}")

    if np.isnan(test_auc) or test_auc < MIN_AUC:
        print(
            f"\nRefusing to save: AUC {test_auc:.3f} is below {MIN_AUC}, which means it\n"
            "barely beats guessing. Either there is not enough signal in these\n"
            "features for your behaviour, or more data is needed. The hand-tuned\n"
            "rules stay in use."
        )
        return 1

    if not np.isnan(baseline_auc) and test_auc < baseline_auc:
        print(
            "\nRefusing to save: it does worse than the hand-tuned rules it would\n"
            "replace. Collect more examples and try again."
        )
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing saved.")
        return 0

    model.meta = {
        "examples": int(len(y)),
        "test_auc": round(float(test_auc), 4),
        "test_accuracy": round(float(accuracy), 4),
        "baseline_auc": None if np.isnan(baseline_auc) else round(float(baseline_auc), 4),
    }
    model.save(WEIGHTS_PATH)
    print(f"\nSaved {WEIGHTS_PATH}")
    print("Restart server.py and it will use this instead of the hand-tuned rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
