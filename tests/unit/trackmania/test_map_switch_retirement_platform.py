import os
from pathlib import Path

import pytest

from trackmaniarl.trackmania.map_switch_retirement import (
    RetirementAttestation,
    retire_quarantined_endpoint,
)


@pytest.mark.skipif(os.name == "nt", reason="Windows has read-locked journal support")
def test_non_windows_retirement_is_refused_before_files_are_accessed() -> None:
    with pytest.raises(RuntimeError, match="requires Windows"):
        retire_quarantined_endpoint(
            old_journal=Path("missing-old"),
            old_instance_id="old",
            new_journal=Path("missing-new"),
            new_instance_id="new",
            attestation=RetirementAttestation(True, True, True, True, Path("missing-evidence")),
        )
