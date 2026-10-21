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
