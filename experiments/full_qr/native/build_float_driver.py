"""Build an isolated diagnostic driver without narrowing mode-12 steering.

The full original locate, tape identity, ownership and process guards remain
mandatory. This does not qualify the assumed float32 engine interpretation:
the caller must compare the decoded native buffer with recorded RPC bits.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.diagnostic_inputs import configured_path

PINNED = {
    "tools/fk/fk/src/tape.rs": "1a25038e77a5bca00cd14e09c1871ca258ecf3892dc29999ee8233b546c81ebf",
    "tools/fk/fk/src/session.rs": (
        "c6e666477cc4bf255c684b301e99054fc98365aec300e54a03491e3bf1afba61"
    ),
    "tools/search/forkoracle/src/forksrv.rs": (
        "1fb9e5c06972be4cc6f2da52f5ab959cd38a0e1bd489aa7b6b2e740c08e3400b"
    ),
    "tools/search/forkoracle/src/layout.rs": (
        "48d3ec53fa5cbe64e0e08f6ee4ec94e8f8bfa3f66918d9d73a310dd7df986820"
    ),
}

STEERS = r"""
    /// Literal channels for this diagnostic; unsupported modes fail closed.
    pub fn engine_steers(&self) -> Result<Vec<f32>, String> {
        self.inner.archives[0].packets.iter().map(|p| {
            let value = match p.mode {
                2 | 4 => p.steer_i8() as f32 / 127.0,
                12 | 13 => f32::from_bits(p.steer),
                0 => 0.0,
                // The pinned tape has idle non-car packets. Their channels
                // must be literal zeros, and full native identity still has
                // to prove this interpretation before any action is issued.
                15 if p.steer == 0 && p.accel == 0 && p.brake == 0
                    && p.tri == Some([0, 0, 0, 0]) => 0.0,
                other => return Err(format!("unsupported diagnostic input mode {}", other)),
            };
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err("invalid diagnostic steering channel".into());
            }
            Ok(value)
        }).collect()
    }
"""

KEY = r"""
/// Same key format and exact locator, retaining every float32 bit.
pub fn write_key_f32(path: &Path, steer: &[f32]) -> Result<(), String> {
    const M: usize = 24;
    if steer.len() < M || steer.len() > u32::MAX as usize
        || steer.iter().any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v)) {
        return Err("invalid float32 key".into());
    }
    let mut t0 = 0usize;
    let mut best = 0usize;
    for t in 0..=steer.len()-M {
        let distinct = steer[t..t+M].iter().map(|v| v.to_bits())
            .collect::<std::collections::BTreeSet<_>>().len();
        if distinct > best { best = distinct; t0 = t; }
    }
    let mut out = Vec::with_capacity(12 + 4*steer.len());
    out.extend_from_slice(&(steer.len() as u32).to_le_bytes());
    out.extend_from_slice(&(t0 as u32).to_le_bytes());
    out.extend_from_slice(&(M as u32).to_le_bytes());
    for value in steer { out.extend_from_slice(&value.to_le_bytes()); }
    std::fs::write(path, out).map_err(|e| e.to_string())
}

#[cfg(test)]
mod float_key_tests {
    use super::*;
    #[test]
    fn preserves_non_byte_steering_and_rejects_invalid_channels() {
        let p = std::env::temp_dir().join(format!("tmrl-float-key-test-{}", std::process::id()));
        assert!(!p.exists());
        let values: Vec<f32> = (0..48).map(|i| (i%13) as f32/6.0-1.0).collect();
        write_key_f32(&p, &values).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len(), 12+4*values.len());
        for (i, value) in values.iter().enumerate() {
            assert_eq!(&bytes[12+4*i..16+4*i], &value.to_le_bytes());
        }
        std::fs::remove_file(&p).unwrap();
        assert!(write_key_f32(&p, &[0.0;23]).is_err());
        assert!(write_key_f32(&p, &[f32::NAN;24]).is_err());
        assert!(write_key_f32(&p, &[2.0;24]).is_err());
        assert!(!p.exists());
    }
}
"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"source anchor mismatch: {path}: {old[:70]}")
    path.write_text(text.replace(old, new))


def main() -> None:
    source, build = map(Path, sys.argv[1:])
    if build.exists():
        raise ValueError("fresh build required")
    if (
        shutil.disk_usage("/mnt/c").free < 10 * 1024**3
        or shutil.disk_usage(source).free < 4 * 1024**3
    ):
        raise RuntimeError("disk reserve")
    for relative, digest in PINNED.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"pinned source mismatch: {relative}")
    staged = build / "source"
    build.mkdir()
    staged.mkdir()
    for directory in ("tools", "driver"):
        shutil.copytree(
            source / directory, staged / directory, ignore=shutil.ignore_patterns("target", ".git")
        )
    for manifest in staged.rglob("Cargo.toml"):
        manifest.write_text(manifest.read_text().replace(str(source), str(staged)))
    tape = staged / "tools/fk/fk/src/tape.rs"
    replace_once(
        tape, "    pub fn n(&self) -> usize {", STEERS + "\n    pub fn n(&self) -> usize {"
    )
    replace_once(
        tape,
        "        self.write_candidate(&self.steer, &self.accel, &self.brake, out)",
        "        check_oracle_readable_name(out)?;\n"
        "        let body = self.inner.splice_into(&self.body, Encoding::Explicit)?;\n"
        "        let g = gbx::container::Gbx::parse("
        "&std::fs::read(&self.path).map_err(|e|e.to_string())?);\n"
        "        gbx::container::write_gbx(&g, body, &out.to_string_lossy())",
    )
    replace_once(
        tape,
        "        (from..self.n())\n"
        "            .map(|t| forkoracle::forksrv::rec_of("
        "self.steer[t], self.accel[t], self.brake[t]))\n            .collect()",
        '        let steer = self.engine_steers().expect("validated diagnostic channels");\n'
        "        (from..self.n()).map(|t| forkoracle::forksrv::Rec { steer: steer[t],\n"
        "            gas: if self.accel[t] != 0 {1.0} else {0.0},\n"
        "            brake: if self.brake[t] != 0 {1.0} else {0.0} }).collect()",
    )
    fork = staged / "tools/search/forkoracle/src/forksrv.rs"
    fork.write_text(fork.read_text() + KEY)
    layout = staged / "tools/search/forkoracle/src/layout.rs"
    text = layout.read_text()
    start = text.index("pub fn verify_tape(")
    end = text.index("/// What a whole-run self-check found.", start)
    function = (
        text[start:end]
        .replace("pub fn verify_tape(", "pub fn verify_tape_f32(")
        .replace("steer: &[u8]", "steer: &[f32]")
    )
    function = function.replace(
        "let want = crate::forksrv::rec_of(steer[t], accel[t], brake[t]);",
        "let want = Rec {steer: steer[t], gas: if accel[t] != 0 {1.0} else {0.0}, "
        "brake: if brake[t] != 0 {1.0} else {0.0}};",
    )
    function = function.replace(
        "g(0) != want.steer || g(4) != want.gas || g(8) != want.brake",
        "g(0).to_bits() != want.steer.to_bits() || g(4).to_bits() != want.gas.to_bits() "
        "|| g(8).to_bits() != want.brake.to_bits()",
    )
    layout.write_text(
        text + "\n// Float32 diagnostic uses the same complete identity control.\n" + function
    )
    session = staged / "tools/fk/fk/src/session.rs"
    text = session.read_text().replace(
        "write_key(&key, &tape.steer);",
        "forkoracle::forksrv::write_key_f32(&key, &tape.engine_steers()?)?;",
    )
    text = text.replace(
        "forkoracle::layout::verify_tape(", "forkoracle::layout::verify_tape_f32("
    ).replace("&self.tape.steer,", "&self.tape.engine_steers()?,")
    session.write_text(text)
    env = os.environ.copy()
    rust = str(configured_path("TRACKMANIARL_RUST_TOOLS"))
    env.update(
        PATH=f"{rust}/rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:" + env["PATH"],
        CARGO_HOME=f"{rust}/cargo",
        RUSTUP_HOME=f"{rust}/rustup",
        CARGO_TARGET_DIR=str(build / "target"),
    )
    commands = [
        (["cargo", "build", "--offline", "--release"], staged / "driver", "build.log"),
        (
            ["cargo", "test", "--offline", "--lib", "float_key_tests"],
            staged / "tools/search/forkoracle",
            "test.log",
        ),
    ]
    for command, cwd, name in commands:
        with (build / name).open("wb") as log:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        if result.returncode:
            raise RuntimeError((build / name).read_text()[-6000:])
    driver = build / "target/release/tmre-reset-driver"
    receipt = {
        "schema": "tmrl.diagnostic-float32-driver-build/v1",
        "source_pins": PINNED,
        "driver": str(driver),
        "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "source_file_hashes": {
            p: hashlib.sha256((staged / p).read_bytes()).hexdigest() for p in PINNED
        },
        "engine_interpretation_qualified": False,
        "training_qualified": False,
    }
    (build / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
