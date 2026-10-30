# Repository layout

TrackmaniaRL is a reusable Python package. Keep your maps, training outputs and
custom components in a separate project created with `trackmaniarl init`.

| Path | Purpose |
| --- | --- |
| `trackmaniarl/` | Published library and bundled project templates |
| `examples/` | Runnable Python examples and portable configurations |
| `readme/` | User guides and API documentation |
| `tests/` | Automated checks |
| `docs/` | Architecture, provenance and development notes |
| `scripts/` | Maintenance and release tools |
| `experiments/` | Historical research and reproducible demonstrations |

Start with the [Python quickstart](../readme/python-quickstart.md) and
[public API](../readme/public-api.md). The [documentation index](../readme/README.md)
links to the game setup and extension guides.

Local `assets/`, `demonstrations/`, `artifacts/` and campaign configurations are
working data, not prerequisites for using the library. Existing experiment paths
are preserved for reproducibility. New application code belongs in its own project.

## Keeping the library focused

`readme/README.md` is the entry point for user documentation. Use `docs/` for
implementation details and dated research evidence, and link to the user guide
rather than copying its instructions. Historical reports describe their original
revision; they are not current setup instructions.

The full QR implementation has two boundaries:

- `trackmaniarl/trackmania/native_headless/full_qr/` owns native observations,
  action execution, event decoding and reward delivery. Separate capture formats
  retain their own validation and provenance.
- `trackmaniarl/experiments/full_qr/` owns the training profile and campaign
  lifecycle. Campaign policy groups clocks, storage reserves and schedule checks;
  workers own their counter reporting. The watchdog and process supervisor remain
  separate because they have independent shutdown responsibilities.

Keep tests beside related behavior instead of creating a new test module for
every helper. Extract a module when it has a distinct responsibility or dependency
boundary, rather than a target line count. Add maintenance tools under `scripts/`
only when they can be reused from a checkout; campaign-specific probes and results
belong outside the repository.
