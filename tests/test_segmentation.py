import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from yeast_vacuole_pipeline.unet_segmentation.dataset import (  # noqa: E402
    CellMaskDataset,
    find_pairs,
    split_pairs,
)
from yeast_vacuole_pipeline.unet_segmentation.metrics import (  # noqa: E402
    bce_dice_loss,
    dice_score,
    iou_score,
)
from yeast_vacuole_pipeline.unet_segmentation.training import run_epoch  # noqa: E402


def square_mask(lo, hi, size=8):
    m = torch.zeros(1, size, size)
    m[:, lo:hi, lo:hi] = 1
    return m


def test_perfect_prediction_scores_one():
    mask = square_mask(2, 6)
    assert dice_score(mask, mask).item() == pytest.approx(1.0)
    assert iou_score(mask, mask).item() == pytest.approx(1.0)


def test_partial_overlap_scores():
    # Two 4x4 squares offset by 2 px along both axes: overlap 2x2 = 4 px.
    a = square_mask(0, 4)
    b = torch.zeros(1, 8, 8)
    b[:, 2:6, 2:6] = 1
    assert dice_score(a, b).item() == pytest.approx(2 * 4 / (16 + 16))
    assert iou_score(a, b).item() == pytest.approx(4 / (16 + 16 - 4))


def test_empty_prediction_and_target_count_as_match():
    empty = torch.zeros(1, 8, 8)
    assert dice_score(empty, empty).item() == pytest.approx(1.0)


def test_loss_is_lower_for_better_prediction():
    target = square_mask(2, 6)
    good = target * 0.9 + 0.05
    bad = 1 - good
    assert bce_dice_loss(good, target) < bce_dice_loss(bad, target)


@pytest.fixture
def toy_data(tmp_path):
    """Two classes x 5 images of 32x32 px, each with a bright disk and its mask."""
    for cls in ("a", "b"):
        (tmp_path / "Mask" / cls).mkdir(parents=True)
        (tmp_path / "Images" / cls).mkdir(parents=True)
        for i in range(5):
            mask = np.zeros((32, 32), np.uint8)
            cv2.circle(mask, (16, 16), 4 + i, 255, -1)
            img = cv2.merge([mask // 2 + 40] * 3)
            cv2.imwrite(str(tmp_path / "Mask" / cls / f"{i}.tif"), mask)
            cv2.imwrite(str(tmp_path / "Images" / cls / f"{i}.tif"), img)
    return tmp_path


def test_find_pairs_matches_images_to_masks(toy_data):
    pairs = find_pairs(toy_data / "Mask", toy_data / "Images")
    assert len(pairs) == 10
    for img, mask in pairs:
        assert img.relative_to(toy_data / "Images") == mask.relative_to(toy_data / "Mask")


def test_find_pairs_reports_missing_image(toy_data):
    (toy_data / "Images" / "a" / "0.tif").unlink()
    with pytest.raises(FileNotFoundError, match="no image"):
        find_pairs(toy_data / "Mask", toy_data / "Images")


def test_split_is_disjoint_and_complete(toy_data):
    pairs = find_pairs(toy_data / "Mask", toy_data / "Images")
    train, val, test = split_pairs(pairs)
    assert (len(train), len(val), len(test)) == (8, 1, 1)
    assert set(train) | set(val) | set(test) == set(pairs)


def test_dataset_returns_normalized_tensors(toy_data):
    ds = CellMaskDataset(find_pairs(toy_data / "Mask", toy_data / "Images"))
    img, mask = ds[0]
    assert img.shape == (3, 32, 32) and img.dtype == torch.float32
    assert 0 <= img.min() and img.max() <= 1
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_training_step_runs_end_to_end(toy_data):
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

    # Small encoder without pretrained weights: exercises the loop, not the real model.
    model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation="sigmoid")
    loader = DataLoader(CellMaskDataset(find_pairs(toy_data / "Mask", toy_data / "Images")), 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = run_epoch(model, loader, optimizer)
    val = run_epoch(model, loader)
    for scores in (train, val):
        assert set(scores) == {"loss", "dice", "iou"}
        assert np.isfinite(scores["loss"])
        assert 0 <= scores["dice"] <= 1 and 0 <= scores["iou"] <= 1
