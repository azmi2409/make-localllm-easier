from localllm import catalog, runtime


def test_pick_by_vram():
    assert catalog.pick(8) is None
    assert catalog.pick(12) == "qwen3.8-27b-iq2"
    assert catalog.pick(16, "th") == "gemma4-26b-a4b-qat"


def test_server_args_mtp_and_device():
    a = runtime.server_args(runtime.Path("m.gguf"), "Vulkan0", 8080, 4096, mtp=True)
    assert a[a.index("-dev") + 1] == "Vulkan0" and "draft-mtp" in a and a[a.index("--spec-draft-n-max") + 1] == "2"
    assert "draft-mtp" not in runtime.server_args(runtime.Path("m.gguf"), None, 8080, 4096, mtp=False)


def test_env_disables_host_visible_vidmem_unless_set(monkeypatch):
    monkeypatch.delenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", raising=False)
    assert runtime.server_env()["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "1"
    monkeypatch.setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "0")
    assert runtime.server_env()["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] == "0"


def test_best_device_skips_igpu():
    devs = [{"id": "Vulkan0", "name": "AMD Radeon RX 9070 XT", "total_gb": 15.9},
            {"id": "Vulkan1", "name": "Intel(R) UHD Graphics 770", "total_gb": 15.9}]
    assert runtime.best_device(devs)["id"] == "Vulkan0"


def test_bench_coverage():
    from localllm import bench
    assert bench.available("en") == ["global"]
    assert bench.available("th") == ["regional"]
    assert bench.available("ja") == ["global", "regional"]
    assert bench.available("xx") == []


def test_sizing_tiers_16gb():
    from localllm import sizing
    rows = {r["shape"]: r for r in sizing.tiers(15.9, 32, "AMD Radeon RX 9070 XT")}
    assert rows["8B"]["status"] == "fits" and rows["24-32B"]["quant"] == "Q3"
    assert rows["30B MoE (3B active)"]["status"] == "offload-moe"
    assert rows["70B"]["status"] == "too-big"
    assert sizing.bandwidth("NVIDIA GeForce RTX 4060 Ti") == 288 and sizing.bandwidth("Mystery GPU") is None


def test_pick_prefers_accuracy_then_speed_on_ties():
    assert catalog.pick(15.9, "zh") == "qwen3.8-27b-q3"      # 5.5 points better in Chinese
    assert catalog.pick(15.9, "en") == "gemma4-26b-a4b-qat"  # tie on accuracy, faster
