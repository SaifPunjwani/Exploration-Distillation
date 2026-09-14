from tmx_jax import data


def test_dapo_loader_pins_the_training_snapshot(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    data._load_raw()

    assert calls == [
        (
            (data.DAPO_DATASET_ID, data.DAPO_DATASET_CONFIG),
            {
                "split": "train",
                "revision": data.DAPO_DATASET_REVISION,
            },
        )
    ]
    assert len(data.DAPO_DATASET_REVISION) == 40


def test_deepscaler_loader_pins_the_training_snapshot(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    data._load_raw_deepscaler()

    assert calls == [
        (
            (data.DEEPSCALER_DATASET_ID,),
            {
                "split": "train",
                "revision": data.DEEPSCALER_DATASET_REVISION,
            },
        )
    ]
    assert len(data.DEEPSCALER_DATASET_REVISION) == 40
