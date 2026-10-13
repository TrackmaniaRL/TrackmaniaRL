# Przekazanie treningów na cztery komputery

Nowe poprawkiPPO przygotowane, NIE uruchomione: opt-in pełnyrollback epoki
przekraczającejKL oraz osobny pipelinepodtrzymujący recovery przystaniu przybandzie.
14 testów kontrolnych/Ruff/mypyPASS, zerorealoptimizersteps. Oddzielne konfiguracje
REVIEWONLY, żadnej nowej kolejki/gry/full/monitora. Wymagają freshfreeze/planu,
bezresume starychCP. [Szczegóły](PPO_REPAIR_CANDIDATES_20261013.md).

Diagnoza regresji bez treningu: CP100 zmienił komendy na477 wspólnych wejściach
i traci stabilność nawrotu78–82%; trzy zatrzymania przy zewnętrznej krawędzi,
gaz pozostaje, no_progress100 kończy próby. Jeden120ms spike nie wyjaśnia
pozostałych dwóch porażek. Nie ustalono konkretnego błędu optymalizatora.
[Raport i rysunek](PPO_REGRESSION_20261013.md). Monitor nadalPAUSED, bez nowych prób.

AKTUALNE: kontynuacjaPPO COMPLETE20:59:10Warszawa, pełne204800/100,
końcowe7/10mean62,207s/median63,14s.QualitygateFAIL, osobno jeden120ms spike/12147.
10 obserwowanych tożsamości closed/mutexfree/sourcehelperSTOPpins i finalCPSHA PASS.
MonitorPAUSED, fullBLOCKED; wcześniejszeACTIVE poniżej to historia.

AKTUALNE: kontynuacjaPPO r2 od19:16:52Warszawa DZIAŁA, preflightREADY,
eventtrain/resumed102400/50, targetTOTAL204800/100. W&B `65msv47w`.
MonitorACTIVE15min; SAVE21:06:52/HARD21:16:52 Warszawa. Tylko końcowa ocenaCP100,
bez zmiany frozenPPO/config/fingerprint, bezfull. Poniższa awariaforeground jest historią.

Najnowsza instrukcja autoryzuje dokończeniePPO z CP50 z nowym limitem2h.
Próba kontynuacji19:11 Warszawa zakończona PRZED treningiem: foreground preflight
failed.6 tożsamości zamknięte/mutex wolny/CP50SHA i351pins zgodne.0 nowych aktualizacji.
Potrzebna interwencja użytkownika: Trackmania na pierwszym planie/odblokowany pulpit.
Nie restartuj użytej kolejki kontynuacji. Monitor nadalPAUSED, fullBLOCKED.

Lokalny pilot `tmrl-ppo-trend-s17` ZAMKNIĘTY17:10 Warszawa:102400/204800 kroków,
50/100 aktualizacji. Ocena pośrednia10/10, średnia51,963s, lecz max130ms>100ms;
zatrzymany przez walidację czasu, bez końcowej oceny. W&B `mbf6v73g`.
Wszystkie14 obserwowanych tożsamości zamknięte, mutex wolny,351pins zgodne.
Monitor PAUSED. Bez restart/resume/nowej próby/full — bezpośrednia instrukcja człowieka.
Aktualny plan: [PPO_TREND_PILOT_20261013.md](PPO_TREND_PILOT_20261013.md).
Poniższe nieaktywne r3 nie jest bieżącą próbą.

13 października: lokalny pilot `tmrl-ppo-p13-prefix-r2-s17` wystartował 10:04:27 UTC,
preflight PASS (33 pola, poprawna mapa, zegar 10 ms), W&B `puhyz84h`.
Został zatrzymany po awarii zapisu statusu launchera (Windows WinError5):
4096 kroków/2 aktualizacje, checkpoint zachowany, wszystkie procesy zamknięte,
mutex wolny. Oceny jazdy nie wykonano. Naprawa nadzoru przeszła 57 testów,
w tym rzeczywistą blokadę pliku Windows. Nie uruchamiaj ponownie ani kolejki r2, ani wcześniejszej
`ppo-pilot-20261013`, która zakończyła się przed treningiem.
Nie uruchamiaj kampanii/full na podstawie samego startu pilota.
Szczegóły: [PPO_PILOT_20261013.md](PPO_PILOT_20261013.md).
R3 `tmrl-ppo-p13-prefix-r3-s17` już zakończony 12:31:46 UTC (14:31 Warszawa):
145408 kroków/71 aktualizacji, 10 kompletnych ocen i 3 mety. Bramka 8/10 FAIL;
runtime/timing/checkpoint/pins/closure PASS. Wszystkie zapisane procesy zamknięte,
mutex wolny. Nie restartuj ani nie uruchamiaj full; brak aktywnego pilota PPO.
Przed dalszym PPO stosuj [PPO_DIAGNOSIS_20261013.md](PPO_DIAGNOSIS_20261013.md):
poprawa widoczności metryk/śladu jazdy, kontrola trendu, następnie dopiero
uzasadniona pojedyncza zmiana i kwalifikacja seedów. Nowej próby jeszcze nie przygotowano.

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
