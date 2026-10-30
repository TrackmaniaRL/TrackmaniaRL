"""Compatibility wrapper for the installed V7 provisioning command."""

from trackmaniarl.trackmania.native_headless.provision_immediate import (
    copy_selected,
    main,
    provision,
    save,
)

__all__ = ["copy_selected", "main", "provision", "save"]

if __name__ == "__main__":
    main()
