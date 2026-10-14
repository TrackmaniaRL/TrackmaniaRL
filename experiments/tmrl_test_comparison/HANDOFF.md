# Przekazanie treningów na cztery komputery

AKTUALNE: używaj **weeks-20261014-v3**, wspólnego launchera Python dla Windows/Linux X11. [Instrukcja i podział dla zespołu](release_20261014_v3/START_HERE.md). V3 zastępuje paczkę v2 przy nowych startach; stare plany/pins pozostają historyczne i wymagają własnego commita. Nie aktualizuj aktywnych treningów. Linux/Proton wymaga lokalnego check-game; Wayland/WSL/headless nie są obsługiwane. Timing raportowany bez progu, SD-SAC wyłączone.

AKTUALNE14 października: człowiek jawnie zniósł blokadę za pojedyncze opóźnienia. Nowa kampania **weeks-20261014-v2** obejmuje QR/IQN/SAC/TQC oraz PPO persistent recovery. SD-SAC wyłączone. Timing nadal raportowany, próg100ms nie blokuje v2; błędy/niepełne pomiary nadal blokują. Przygotowano15świeżych runów, żadnego nie uruchomiono. [Instrukcja dla kolegów i polecenia startowe](release_20261014/START_HERE.md). Używaj wyłącznie skryptów z release_20261014; wszystkie wcześniejsze BLOCKED/ACTIVE/stare polecenia poniżej są historyczne.

AKTUALNE14 października: potwierdzenie PPO recoveryseed29 COMPLETE9/10, median51.980s; seed17 10/10median51.835s. Oba qualityPASS, timingFAIL140/180ms. Seed29 jeden no_progress79.75% przy max50ms; outlier180ms w ukończonej próbie. Wszystkie21ownedclosed/mutexfree/CP100/traces/pinsverified. MonitorPAUSED po zakończeniu; fullBLOCKED. [Wynik i następna decyzja](PPO_CONFIRMATION_RESULT_20261014.md). Poniżej historyczne stany.

AKTUALNE14 października09:40Warszawa: człowiek polecił kontynuować PPO.
Nowe potwierdzenie seed29-r2 uruchomione od07:40:50UTC, preflightREADY20ms,
ComputerUse normalImprove przywróciłoactiveplayer z RaceValidationfinishmenu.
Źródło01459dbb, freshqueue/run/noresume, SAVE13:30/HARD13:40Warszawa,
monitorACTIVE15min. FullBLOCKED do niezależnejoceny/timingu i zgodnego planu.
[Aktualny kontrakt](PPO_CONFIRMATION_20261014.md). StarszePAUSEDhistoryczne.

AKTUALNE14 października06:17Warszawa: recoveryseed17 final10/10median51.835s,
timingFAIL140ms, checkpoint/closureverified. Seed29nieprzetestowany: preflight
correctUID→player_not_ready before_reset,0updates, brakrun; wszystkieownedclosed,
mutexfree/360pins/predecessorCP i STOPverified. MonitorPAUSED po raporcie, fullBLOCKED.
Przywróć grywalny stan wyścigu przed przyszłym świeżym potwierdzeniem; nie ponawiaj
użytej kolejki i nie obchodź readiness. [Raport](PPO_OVERNIGHT_REPORT_20261014.md).
WcześniejszeACTIVEponiżej historyczne.

AKTUALNE14 października02:46Warszawa: KLvariant COMPLETE0/10, odrzucony,
CPfinitefingerprint204800/100/traces/allownedclosed/mutexfree/sourcepins verified;
timing osobnoFAIL130ms. Świeży isolatedrecoverypilot od00:46:40UTC, preflightREADY,
oryginalny PPOupdater/freshseed17/frozen1fd74717, monitorACTIVE15min,
SAVE06:36/HARD06:46Warszawa. Bez dowodu naprawyjazdy/full. Ledger i aktualny
[plan](PPO_OVERNIGHT_20261013.md); stareACTIVEKL poniżej historyczne.

AKTUALNE13 października23:18Warszawa: autoryzowana nocna pętlaPPO do14 października
11:04Warszawa. Aktywny świeży wariantKLguard `tmrl-ppo-overnight-kl-s17-r2`,
preflightREADY, frozen87832312, independentguardREADY, SAVE03:08/HARD03:18Warszawa.
MonitorACTIVE15min. Brak jeszcze wynikujazdy; fullBLOCKED. Ledger w
BASE/ppo-overnight-20261013/repair-state.json, [kontrakt](PPO_OVERNIGHT_20261013.md).
Pierwszy start zamknięty po naprawie forwarding-wrapper guarda, bezCP/events;
nie wznowiony, STOP i closureproof zachowane. Starsze stany poniżej historyczne.

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
