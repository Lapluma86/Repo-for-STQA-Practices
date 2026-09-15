"""AI辅助回归：扫描之后深度文件损坏或丢失时应报告文件读取错误。"""
from pathlib import Path
import pytest

pytestmark = [pytest.mark.module1, pytest.mark.blackbox]


@pytest.mark.parametrize("broken", ["missing", "corrupt"])
def test_depth_read_failure(rail_factory, broken):
    dataset = rail_factory()
    depth = Path(dataset.samples[0]["depth_path"])
    if broken == "missing":
        depth.unlink()
    else:
        depth.write_bytes(b"not a tiff")
    with pytest.raises(FileNotFoundError) as error:
        dataset[0]
    assert str(depth) in str(error.value)
