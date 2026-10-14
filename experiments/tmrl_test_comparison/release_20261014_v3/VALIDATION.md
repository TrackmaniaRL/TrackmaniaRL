# Weryfikacja wersji v3 — 14 października 2026

104 testy PASS: reset/fokus Windows, reset Linux z atrapą X11, odmowa Wayland/
headless/SSH/utraty fokusu, zwolnienie klawisza przy STOP, blokady źródeł/konfiguracji,
równe budżety/15 przydziałów, STOP przed spawnem, bounded save/closure,
niezasygnalizowanie ponownie użytego PID, checkpointy i pomiary ocen.
Ruff check/format PASS; kontrola typów nowych modułów PASS na profilach Windows i Linux.

Osobno rzeczywiste testy blokad:
- Windows: drugi proces nie uzyskał Global\TrackmaniaRL.ComparisonController;
  po zwolnieniu ponowne uzyskanie PASS.
- Ubuntu/Python3.12.3 pod WSL, wyłącznie test mechanizmu systemowego: drugi proces
  nie uzyskał flock; ponowne uzyskanie po release i odmowa Wayland PASS.
  Nie uruchamiano tam gry. WSL nie jest wspieranym hostem treningu.

Nowy plan v3 i wszystkie15 YAML-i zweryfikowane bez gry. PPO ma persistent recovery
z oryginalnym updaterem, pozostałe algorytmy bez zmiany modelu/nagrody. Timing
report-only według jawnej decyzji człowieka; błędy i niepełne pomiary nadal fatal.
Nowy launcher jest Pythonem i nie wymaga wykonywania skryptów PowerShell.

Nie wykonano treningu, benchmarku, game preflightu ani kalibracji w tej sesji.
Nie potwierdzono gamepada/force feedback, resetu w Proton ani stabilności gry na
komputerach kolegów. Linux X11/Steam/Proton pozostaje eksperymentalną integracją,
z obowiązkową lokalną kalibracją i check-game. Wayland, WSL i headless nieobsługiwane.

Archiwalne plany v1/v2/piloty/frozen/STOP/CP zachowane. Wspólny moduł resetu ma
nową ścieżkę Linux; stare piny celowo nie pasują do nowego źródła. Stare próby
należą do własnych commitów. Nie repinowano żadnej historycznej kampanii.
Nowe identyfikatory v3 zapobiegają pomieszaniu wyników i resume zmienionego kodu.
Nadzór jest procesem lokalnym, nie niezależną usługą odporną na SIGKILL/awarię OS.
