import struct
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.unit.trackmania.test_immediate_environment import capture
from trackmaniarl.trackmania.native_headless._runtime import codec_v7
from trackmaniarl.trackmania.native_headless._runtime.continuous import ContinuousControl
from trackmaniarl.trackmania.native_headless._runtime.immediate import (
    ImmediateSession,
    qualify_continuous_immediate,
)
from trackmaniarl.trackmania.native_headless.full_qr.continuous import (
    ExperimentalContinuousEnvironment,
    ExperimentalContinuousEnvironmentFactory,
    ExperimentalContinuousProvider,
)
from trackmaniarl.trackmania.native_headless.full_qr.environment import FullQrEnvironment


def test_native_record_has_sgb_order_and_float32_values() -> None:
    control = ContinuousControl(0.25, 0.75, -0.5)
    assert control.wire_record().hex() == "000000bf0000803e0000403f"
    assert control.receipt_identity()["record_hex"] == control.wire_record().hex()


@pytest.mark.parametrize("values", [(float("nan"), 0, 0), (0, 1.1, 0), (0, 0, -1.1)])
def test_invalid_values_fail_closed(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="finite action domain"):
        ContinuousControl(*values)


def test_existing_record_evidence_can_be_checked_with_new_identity() -> None:
    before, _, packet = capture()
    control = ContinuousControl(1, 0, 0)
    packet["authorization"]["actionId"] = control.receipt_identity()
    assert len(qualify_continuous_immediate(packet, before, control, 1)) == 1
    altered = deepcopy(packet)
    altered["authorization"]["actionId"]["record_hex"] = "00" * 12
    with pytest.raises(ValueError, match="action/anchor"):
        qualify_continuous_immediate(altered, before, control, 1)
    with pytest.raises(ValueError, match="action/anchor"):
        qualify_continuous_immediate(packet, before, ContinuousControl(0.5, 0, 0), 1)


def test_experimental_actions_are_rejected_without_explicit_session_enablement() -> None:
    session = ImmediateSession.__new__(ImmediateSession)
    for control in (ContinuousControl(0.25, 0, 0), ContinuousControl(0, 0.25, 0)):
        with pytest.raises(ValueError, match="not enabled"):
            session.advance_continuous(control, 1)


def test_analog_receipt_requires_explicit_decoder_opt_in() -> None:
    _, _, packet = capture()
    raw = bytearray.fromhex(packet["responseFramesHex"][0])
    struct.pack_into("<f", raw, 320, 0.25)
    with pytest.raises(ValueError, match="outside native domain"):
        codec_v7.decode_v7_response(bytes(raw))
    assert codec_v7.decode_v7_response(bytes(raw), allow_continuous_pedals=True)["kind"] == 6


def test_environment_converts_gas_brake_steer_without_discrete_quantization() -> None:
    environment = ExperimentalContinuousEnvironment.__new__(ExperimentalContinuousEnvironment)
    with patch.object(FullQrEnvironment, "step", return_value="ok") as step:
        assert environment.step(np.array([0.25, 0.75, -0.5], dtype=np.float32)) == "ok"
    assert step.call_args.args[0] == ContinuousControl(0.25, 0.75, -0.5)
    with pytest.raises(ValueError, match="shape"):
        environment.step(np.zeros((1, 3)))


def test_factory_requires_explicit_experimental_opt_in() -> None:
    with pytest.raises(ValueError, match="opt-in"):
        ExperimentalContinuousEnvironmentFactory({})


def test_factory_checks_pinned_continuous_shim(tmp_path: Path) -> None:
    shim = tmp_path / "shim.so"
    shim.write_bytes(b"not the expected binary")
    factory = ExperimentalContinuousEnvironmentFactory.__new__(
        ExperimentalContinuousEnvironmentFactory
    )
    factory._continuous_shim_sha256 = "0" * 64
    settings = SimpleNamespace(assets=SimpleNamespace(paths={"shim": shim}))
    with pytest.raises(ValueError, match="shim pin differs"):
        factory._create_provider(settings, None)


def test_provider_repeats_continuous_command_and_uses_issued_steer() -> None:
    provider = ExperimentalContinuousProvider.__new__(ExperimentalContinuousProvider)
    command = ContinuousControl(0.25, 0.75, -0.5)
    assert provider._control_for_tick(command, 4) is command
    assert provider._decision_steering(command) == -0.5
