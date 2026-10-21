# Native dependency licenses

The local crates under `vendor/trackmania-tas/` originate from Christopher
Chedeau's trackmania-tas and retain its MIT license. `SOURCE-PROVENANCE.json`
records the exact upstream base, license commit and copied source hashes.
TrackmaniaRL additions are MIT; see `LICENSE` and the repository `NOTICE`.

Cargo downloads registry dependencies separately. This repository does not
vendor their source or distribute their compiled libraries. The lockfile also
contains development and macro dependencies, not just code linked into a driver.

| Crate | Locked version | Declared license |
| --- | --- | --- |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| memchr | 2.8.3 | Unlicense OR MIT |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 |
| serde_json | 1.0.145 | MIT OR Apache-2.0 |
| syn | 3.0.9 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |

`OR` permits choosing an offered license; `AND` retains both obligations.
In particular, unicode-ident also requires Unicode-3.0, and ryu is not MIT.
Preserve the applicable licenses/notices when vendoring dependencies or
distributing compiled artifacts. This source-only release is not clearance
for a bundled server, assets, wheel collection or CUDA runtime.

The Python SDK installs dependencies through uv/pip. orjson 3.11.7 includes
MPL-2.0 AND (Apache-2.0 OR MIT) code; using that separate dependency does not
relicense this SDK. Copying/modifying covered files or shipping its binaries
requires complying with those licenses. Keep its original distribution notices.
