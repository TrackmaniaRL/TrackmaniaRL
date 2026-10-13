# Przekazanie treningów na cztery komputery

13 października: lokalny pilot `tmrl-ppo-p13-prefix-r2-s17` wystartował 10:04:27 UTC,
preflight PASS (33 pola, poprawna mapa, zegar 10 ms), W&B `puhyz84h`.
Został zatrzymany po awarii zapisu statusu launchera (Windows WinError5):
4096 kroków/2 aktualizacje, checkpoint zachowany, wszystkie procesy zamknięte,
mutex wolny. Oceny jazdy nie wykonano. Naprawa nadzoru przeszła 57 testów,
w tym rzeczywistą blokadę pliku Windows. Nie uruchamiaj ponownie ani kolejki r2, ani wcześniejszej
`ppo-pilot-20261013`, która zakończyła się przed treningiem.
Nie uruchamiaj kampanii/full na podstawie samego startu pilota.
Szczegóły: [PPO_PILOT_20261013.md](PPO_PILOT_20261013.md).
Aktualnie działa wyłącznie świeży `tmrl-ppo-p13-prefix-r3-s17`, start 10:14:51 UTC,
preflight PASS, W&B `brczyy2x`. SAVE 14:04:51 UTC, HARD 14:14:51 UTC.
Nie restartuj; używa frozen runtime r3 z naprawioną obsługą statusu i błędów.

| Osoba | Zadanie | Seedy |
| --- | --- | --- |
| Jakub | QR, następnie IQN | 17, 29, 43 każdego |
| Borys | ciągły SAC | 17, 29, 43 |
| Kamil | TQC | 17, 29, 43 |
| Kuba P. | PPO, po lokalnym pilocie | 17, 29, 43 |

Pełna instrukcja instalacji, budżetu, poleceń bez jazdy, późniejszego startu,
STOP, przerwy, backupu i oddawanych wyników jest w aktualnym planie poniżej.
Domyślne `run_campaign.ps1 -Worker ...` tylko pokazuje kolejkę; dopiero `-Launch`
uruchamia grę. Przygotowano 15 nowych runów, żadnego nie uruchomiono.
Wszyscy pracują na tym samym czystym commicie brancha `codex/tmrl-algorithm-runs`.
Nie kopiuj `.env`, checkpointów ani kalibracji autora do nowego treningu.

Aktualny plan: [MULTI_COMPUTER_TRAINING.md](MULTI_COMPUTER_TRAINING.md).
SD-SAC jest **experimental**, wyłączony ze standardowych kolejek; scheduler
napraw jest **PAUSED**. Raport i pełny prompt dla Fable 5.1:
[SD_SAC_FABLE_5_1_REPORT.md](SD_SAC_FABLE_5_1_REPORT.md).

Historia: [archiwum HANDOFF.md](archive/20261008/HANDOFF.md).
