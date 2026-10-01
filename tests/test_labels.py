from asl_realtime.labels import SIGNS
from asl_realtime.landmarks import NUM_CLASSES


def test_one_unique_name_per_class_in_dataset_order():
    assert len(SIGNS) == NUM_CLASSES
    assert len(set(SIGNS)) == NUM_CLASSES
    assert list(SIGNS) == sorted(SIGNS)
    assert SIGNS[0] == "TV" and SIGNS[-1] == "zipper"
