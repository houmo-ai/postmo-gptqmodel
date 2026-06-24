from auto_round.compressors.mllm import compressor


class FakeParam:
    def numel(self):
        return 1

    def element_size(self):
        return 1


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=[1])
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class FakeTokenizer:
    padding_side = "right"
    pad_token = None
    eos_token = "<eos>"

    def __init__(self):
        self.inputs = FakeInputs()

    def __deepcopy__(self, memo):
        return self

    def __call__(self, *args, **kwargs):
        return self.inputs


class FakeModel:
    def __init__(self):
        self.moves = []
        self.called = 0

    def parameters(self):
        return [FakeParam()]

    def to(self, device):
        self.moves.append(str(device))
        return self

    def __call__(self, **kwargs):
        self.called += 1
        return object()


def setup_cuda_probe(monkeypatch):
    monkeypatch.setattr(compressor, "detect_device", lambda device: device)
    monkeypatch.setattr(compressor.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(compressor.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(compressor.torch.cuda, "mem_get_info", lambda idx: (1024, 2048))


def test_only_text_test_low_gpu_mem_usage_stays_on_cpu(monkeypatch):
    setup_cuda_probe(monkeypatch)
    model = FakeModel()
    tokenizer = FakeTokenizer()

    assert compressor._only_text_test(model, tokenizer, "cuda", "fake_mllm", low_gpu_mem_usage=True) is True

    assert "cuda" not in model.moves
    assert model.moves == ["cpu"]
    assert tokenizer.inputs.moves == ["cpu"]


def test_only_text_test_gpu_success_restores_cpu_and_clears_cache(monkeypatch):
    setup_cuda_probe(monkeypatch)
    empty_cache_calls = []
    monkeypatch.setattr(compressor.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    model = FakeModel()
    tokenizer = FakeTokenizer()

    assert compressor._only_text_test(model, tokenizer, "cuda", "fake_mllm") is True

    assert model.moves == ["cuda", "cpu"]
    assert tokenizer.inputs.moves == ["cuda", "cpu"]
    assert empty_cache_calls == [True]
