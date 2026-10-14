Cześć, aktualna paczka treningowa to weeks-20261014-v3 — działa przez ten sam
launcher Python na Windows i Linux X11. Poprzedniej instrukcji v2 już nie używajcie
przy nowych startach. Jeśli ktoś już trenuje v2, niech nie zmienia kodu w trakcie.

Przydział: Jakub QR, potem IQN; Borys SAC; Kamil TQC; Kuba P. PPO recovery.
SD-SAC pomijamy. Każdy algorytm: seedy 17/29/43, po 8 192 000 kroków i 30 ocen.
Plan to około 2 tygodnie samej jazdy na 3 seedy, u Jakuba około 4 tygodnie,
plus uczenie, resetowanie i zapisy.

Kod: branch codex/tmrl-algorithm-runs, dokładny commit w RELEASE.json.
Instrukcja: experiments/tmrl_test_comparison/release_20261014_v3/START_HERE.md.
Są osobne kroki instalacji gry na Windows i Linux oraz wspólne polecenia startu.
Windows nie wymaga już odpalania skryptów PowerShell. Linux: Steam/Proton,
sesja X11, xdotool, libevdev/uinput; nie Wayland/WSL/headless. Linuxowy reset
wymaga lokalnego check-game — nie wykonano tu treningu w grze na Linuksie.

Przed startem odeślijcie hash commita oraz raporty host-check i check-game;
po pierwszych metrykach link W&B. Własny klucz i kalibracja na każdym PC.
Nie wysyłajcie klucza/.env. Minimum 100 GiB wolnego miejsca plus zapas i backup.
Jeden komputer = jeden kontroler; terminal pozostaje uruchomiony.
Timing ponad 100 ms raportujemy, sam nie blokuje; błędy i niepełne wyniki blokują.
STOP i wyniki do oddania są opisane w instrukcji. Bez zmiany kodu/YAML-i w trakcie,
bez samodzielnych retry/resume i wybierania najlepszego checkpointu.
