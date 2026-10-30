# Źródło eksperymentalnego protokołu V7

Ten katalog zawiera wyłącznie patch i dokumenty pochodzenia. Nie zawiera gry,
Packs, ghosta ani biblioteki wykonywalnej. V6 i jego hash pozostają oddzielne.
Pliki dowodowe mają przypięte bajty, dlatego lokalne `.gitattributes` wyłącza
konwersję ich zakończeń linii.

Źródłowe archiwum prywatnej dostawy:
`TrackmaniaRL-RE-zrodla-i-kontekst-20261021.zip`, SHA-256
`863ae95ff8a8d1e5339aea255b2675fd2d09eb652f8f6d9af4f437317bd346b9`.
Wybierz dokładnie 179 plików z prefiksu
`Trackmania2020RE/build/forkoracle/native-long-v6-001/source/tools/`.
Prywatny, oryginalny `original-source-manifest.json` podaje względne ścieżki,
rozmiary i hashe; nie jest dołączony do paczki.
Wypakuj je do nowego katalogu; sprawdź każdy wpis przed zastosowaniem patcha.

Patch zmienia wyłącznie `search/forkshim/src/post_observe/native_ipc.rs`.
Oryginał ma SHA-256
`94f864828b0000b3061b070fe08794bb99697a292b6a1ff51070378b9da18b40`.
Przy `SOURCE_TOOLS` wskazującym wypakowany katalog `tools`:

```sh
patch --strip-trailing-cr -p1 -d "$SOURCE_TOOLS/search/forkshim/src" < v6-to-v7.patch
```

Zapis użytego w pomiarze pliku Rust ma CRLF. Po zastosowaniu patcha znormalizuj
tylko ten plik do CRLF i sprawdź SHA-256:

```sh
python3 - "$SOURCE_TOOLS/search/forkshim/src/post_observe/native_ipc.rs" <<'PY'
from pathlib import Path
import hashlib, sys
p = Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
assert hashlib.sha256(p.read_bytes()).hexdigest() == "159cb143e15594f60258dbbfff358622a0b7a8a1fc813611eb7dccc9a52df51b"
PY
cd "$SOURCE_TOOLS/search"
cargo build --locked --release -p forkshim
```

Pomiar używał `cargo 1.99.1 (5f94df478 2026-08-27)` i
`rustc 1.99.1 (58cf8f9c3 2026-10-13)`, Linux x86-64, istniejącego profilu
release: opt-level 3, LTO, codegen-units 1. Żaden pozostały plik źródłowy ani
Cargo.lock nie zmienił hasha podczas tego builda. Weryfikowany wynik ma
594648 bajtów i SHA-256
`fb041afb982e7cc7a5922d49c3dd0ed49c2e6ae698dd0fd103415fbeddf98004`.
Hash nowego builda trzeba porównać; inna ścieżka lub toolchain może dać inne
bajty i nie dziedziczy automatycznie kwalifikacji tego artefaktu.

Prywatny `build-receipt.json` zapisuje faktyczny wynik i pierwotną lokalizację.
Nie jest dystrybuowany w paczce ani archiwum źródłowym. Dostarcz oryginalne,
niezmienione bajty manifestu i receipt w jawnym `--source-evidence-dir` dla
`python -m trackmaniarl.trackmania.native_headless.provision_immediate`; przypięty hash w `immediate_profile.json`
pozostaje obowiązkowy. Skrypt przyjmuje bibliotekę osobno, weryfikuje jej tożsamość
i tworzy osobny runtime na ext4 z jawnej konfiguracji V6. Nie nadpisuje instalacji
źródłowej i nie akceptuje zanonimizowanego receipt jako dowodu oryginalnego builda.

V7 jest nową ramką TMA7/TMR7/version7. Op6 zapisuje dokładnie `n` rekordów
`p..p+n-1`, dla `1 <= n <= 5`, przed ich załadowaniem przez oryginalny reader.
Receipt ma `P=p`, `k=p`, `length=n`, `mode=2`; odpowiedź kind6 ma flags4.
TMAUD003 zapisuje przed/po dla bieżącego rekordu, zakresu oraz shadow steer.
TMAUD002 i zakaz zapisu `k<=p` pozostają w odrębnej ścieżce op4. Nie należy
przedstawiać V7 jako niezmienionego protokołu V6.

Granice kwalifikacji i wymagania wdrożenia opisuje przewodnik
`readme/headless-full-qr.md` w repozytorium.
