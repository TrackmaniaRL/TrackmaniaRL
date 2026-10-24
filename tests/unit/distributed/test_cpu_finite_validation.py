import pytest
import torch

from trackmaniarl.distributed.coordinator_validation import _validate_numeric_array


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, float("nan"), float("inf"), -float("inf")])
def test_cpu_finiteness_keeps_torch_acceptance(dtype: torch.dtype, value: float) -> None:
    data = torch.tensor([value, 0.0], dtype=dtype, requires_grad=True)[::2]
    expected = bool(torch.isfinite(data).all())
    before = data.detach().clone()
    if expected:
        _validate_numeric_array(data, "observation")
    else:
        with pytest.raises(ValueError, match="contains non-finite"):
            _validate_numeric_array(data, "observation")
    torch.testing.assert_close(data.detach(), before, equal_nan=True)


def test_empty_and_noncontiguous_cpu_views_remain_valid() -> None:
    for data in (torch.empty(0), torch.arange(20.0).reshape(4, 5).t(), torch.tensor([True])):
        _validate_numeric_array(data, "observation")
