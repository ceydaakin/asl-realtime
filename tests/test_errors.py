import numpy as np

from asl_realtime.errors import per_sign_accuracy, top_confusions

SIGNS = ["apple", "bed", "cat"]
Y = np.array([0, 0, 0, 0, 1, 1, 2])
PRED = np.array([0, 1, 1, 2, 1, 1, 0])


def test_per_sign_accuracy_is_sorted_worst_first():
    rows = per_sign_accuracy(Y, PRED, SIGNS)

    assert [(r["sign"], r["clips"], r["top1"]) for r in rows] == [("cat", 1, 0.0), ("apple", 4, 0.25), ("bed", 2, 1.0)]


def test_top_confusions_counts_wrong_pairs_only():
    rows = top_confusions(Y, PRED, SIGNS, n=2)

    assert rows[0] == {"true": "apple", "predicted": "bed", "count": 2, "share_of_true": 0.5}
    assert len(rows) == 2 and all(r["true"] != r["predicted"] for r in rows)


def test_top_confusions_with_no_mistakes():
    assert top_confusions(Y, Y, SIGNS) == []
