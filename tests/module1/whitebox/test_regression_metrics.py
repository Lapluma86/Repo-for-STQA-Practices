"""AI辅助回归：时延精度及非法执行次数，使用可控时钟排除机器噪声。"""
import pytest
import torch
from eval import metrics_engineering as metrics

pytestmark = [pytest.mark.module1, pytest.mark.whitebox]


def test_latency_uses_monotonic_high_resolution_clock(monkeypatch):
    ticks = iter([10.0, 10.002])
    monkeypatch.setattr(metrics.time, "time", lambda: 100.0)
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(ticks))
    calls = []
    class Model(torch.nn.Module):
        def forward(self, x):
            calls.append(1)
            return x
    result = metrics.measure_inference_latency(Model(), (torch.ones(1),), torch.device("cpu"), 2, 4)
    assert result == pytest.approx(0.5)
    assert len(calls) == 6


@pytest.mark.parametrize("runs", [0, -1, 1.5, True])
def test_latency_invalid_runs(runs):
    with pytest.raises(ValueError, match="n_run"):
        metrics.measure_inference_latency(torch.nn.Identity(), (torch.ones(1),), torch.device("cpu"), n_run=runs)


@pytest.mark.parametrize("warmup", [-1, 0.5, True])
def test_latency_invalid_warmup(warmup):
    with pytest.raises(ValueError, match="n_warmup"):
        metrics.measure_inference_latency(torch.nn.Identity(), (torch.ones(1),), torch.device("cpu"), n_warmup=warmup)
