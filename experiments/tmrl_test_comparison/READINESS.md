# Gotowość — 8 października 2026

AKTUALNE: używaj **weeks-20261014-v3**, wspólnego launchera Python dla Windows/Linux X11. [Instrukcja i podział dla zespołu](release_20261014_v3/START_HERE.md). V3 zastępuje paczkę v2 przy nowych startach; stare plany/pins pozostają historyczne i wymagają własnego commita. Nie aktualizuj aktywnych treningów. Linux/Proton wymaga lokalnego check-game; Wayland/WSL/headless nie są obsługiwane. Timing raportowany bez progu, SD-SAC wyłączone.

AKTUALNE14 października: człowiek jawnie zniósł blokadę za pojedyncze opóźnienia. Nowa kampania **weeks-20261014-v2** obejmuje QR/IQN/SAC/TQC oraz PPO persistent recovery. SD-SAC wyłączone. Timing nadal raportowany, próg100ms nie blokuje v2; błędy/niepełne pomiary nadal blokują. Przygotowano15świeżych runów, żadnego nie uruchomiono. [Instrukcja dla kolegów i polecenia startowe](release_20261014/START_HERE.md). Używaj wyłącznie skryptów z release_20261014; wszystkie wcześniejsze BLOCKED/ACTIVE/stare polecenia poniżej są historyczne.

AKTUALNE14 października: potwierdzenie PPO recoveryseed29 COMPLETE9/10, median51.980s; seed17 10/10median51.835s. Oba qualityPASS, timingFAIL140/180ms. Seed29 jeden no_progress79.75% przy max50ms; outlier180ms w ukończonej próbie. Wszystkie21ownedclosed/mutexfree/CP100/traces/pinsverified. MonitorPAUSED po zakończeniu; fullBLOCKED. [Wynik i następna decyzja](PPO_CONFIRMATION_RESULT_20261014.md). Poniżej historyczne stany.

AKTUALNE14 października09:40Warszawa: po nowym poleceniu człowieka świeże
potwierdzenie recoveryseed29 `tmrl-ppo-overnight-recovery-s29-r2` DZIAŁA,
preflightREADY20ms/33fields/55reads. ComputerUse zobaczył RaceValidationfinishmenu
i przywrócił grę normalnymImprove. Nowyfrozen01459dbb/nowaqueue-r2/noresume,
actualSTART07:40:50UTC,SAVE13:30/HARD13:40Warszawa; monitorACTIVE15min.
To nowy boundedpilot, nie wydłużenie zakończonej nocy. FullBLOCKED.
[Aktualne potwierdzenie](PPO_CONFIRMATION_20261014.md); starszePAUSEDhistoryczne.

AKTUALNE14 października06:17Warszawa: recoveryseed17 COMPLETE10/10,
mean51.841s/median51.835s, CPfinitefingerprint204800/100/traces/closureverified.
TimingFAILjedna140ms/10368decyzji, fullBLOCKED. Seed29potwierdzenie zatrzymane
PRZEDresetem/treningiem: correctUID, player_not_ready;0updates, runnieutworzony.
Wszystkieownedclosed/mutexfree/pins/priorCP/STOPverified. MonitorPAUSED po raporcie;
brak ponowienia lub obchodzenia readiness. [Raport](PPO_OVERNIGHT_REPORT_20261014.md).
StarszeACTIVEponiżej historyczne. Nie ogłaszamy gotowości na wiele tygodni.

AKTUALNE14 października02:46Warszawa: KLvariant zakończony0/10, odrzucony;
mid9/10→final0/10 mimo KL<=.02. Full204800/100, CP/traces/closure/mutex/pins verified.
Osobno timingFAIL130ms. Nowy izolowany recoverypilot `tmrl-ppo-overnight-recovery-s17`
start00:46:40UTC, preflightREADY10ms/33fields/60reads; oryginalny updaterPPO,
freshseed17, frozen1fd74717. SAVE06:36/HARD06:46Warszawa, monitorACTIVE15min.
Brak jeszcze wyniku jazdy nowej próby. FullBLOCKED; wcześniejszeACTIVEKL historyczne.

AKTUALNE 13 października23:18Warszawa: człowiek autoryzował nocną pętlę
PPO do14 października11:04Warszawa. Świeży izolowany KLguard pilot
`tmrl-ppo-overnight-kl-s17-r2` wystartował, preflightREADY10ms/33fields/59reads.
Frozen87832312; independentdeadlineguardREADY. SAVE03:08/HARD03:18Warszawa,
204800/100, monitorACTIVE15min, fullBLOCKED. Start nie dowodzi poprawyjazdy.
Pierwszy krótki start zamknięty po diagnozie guard forwarding-wrapper; brakCP/events,
allobservedclosed/mutexfree/360pins, STOP i proof zachowane. Poniżej historia.
[Aktualny plan](PPO_OVERNIGHT_20261013.md).

PrzygotowaneOFFLINE opt-in kandydatyPPO: transactionalpostepochKL rollback
oraz persistentincidentmemory tylko przy trwającym utknięciu.14 kontrolnych testów,
Ruff/mypyPASS;20historycznychśladów: recoverygate12/9/13→100/100/100, bez zmian
cech dobrychCP50jazd ani ruchomych klatek. To nie dowód naprawyjazdy.
Bez treningu/optimizersteps, monitorPAUSED. KonfiguracjeREVIEWONLY wymagają nowego
freeze/planu przed przyszłą próbą. [Uzasadnienie](PPO_REPAIR_CANDIDATES_20261013.md).

DiagnozaCP50→100 offline: trzy zatrzymania przy81% silnie wskazują kontakt
z zewnętrzną bandą nawrotu i brak odzyskania ruchu.477 wspólnych obserwacji
potwierdza drift sterowania, błąd odtworzeniaakcji<0,000002. Spowolnienie zaczyna
się wcześniej na trasie; brak dowodu resetuLR/załamania eksploracji/konkretnego
błędu optymalizatora. Bez nowego treningu. Raport:
[PPO_REGRESSION_20261013.md](PPO_REGRESSION_20261013.md).

AKTUALNE13 października21:03Warszawa: PPO COMPLETE, pełne204800/100.
Końcowa ocena7/10mean62,207s/median63,14s; trzy no_progress około81%.
Qualitygate8/10FAIL. Osobno jedna decyzja120ms/12147, timinggateFAIL.
Wszystkie10 obserwowanych tożsamości zamknięte/mutex wolny/pins i finalCP zweryfikowane.
MonitorPAUSED, fullBLOCKED, bez automatycznego restartu. StanyACTIVE poniżej historyczne.

AKTUALNE13 października19:18Warszawa: kontynuacjaPPO r2 DZIAŁA. PreflightREADY,
train/resumed potwierdza102400/50 z CP50; ta sama frozenkonfiguracja, targetTOTAL204800/100.
W&B `65msv47w`; monitorACTIVE15min. SAVE21:06:52/HARD21:16:52 Warszawa.
Pierwotna próba kontynuacji poniżej jest zamkniętą historią. Full nadalBLOCKED.

Najnowszy stan13 października19:11 Warszawa: człowiek autoryzował dokończeniePPO
z CP50 z nowym limitem2h. Osobna kontynuacja zatrzymana w preflight przed treningiem:
nie można ustawić Trackmanii na pierwszym planie. Wszystkie procesy zamknięte,
0 nowych aktualizacji, CP50 i351pins zgodne. Potrzebna gotowa gra na pierwszym
planie; monitor nadalPAUSED. Szczegóły w dokumencie pilota.

**tmrl-ppo-trend-s17 zamknięty 13 października o17:10 Warszawa** po102400 krokach/50 aktualizacjach.
Ocena pośrednia10/10, średnia51,963s, ale jedna decyzja130ms przekroczyła limit100ms.
Końcowej oceny ani pełnego budżetu204800 nie wykonano; full nadal BLOCKED.
Wszystkie14 obserwowanych tożsamości procesów zamknięte, mutex wolny,351pins zgodne.
Monitor PAUSED. Człowiek zakazał wznowienia; tylko diagnoza i naprawy offline.
Szczegóły: [PPO_TREND_PILOT_20261013.md](PPO_TREND_PILOT_20261013.md).
Nie restartować, nie resume/extend/full. Poniższy r3 to zakończony predecessor.

Aktualizacja 13 października: zainstalowano PowerShell 7.6.6 dla bieżącego konta
i potwierdzono działający podgląd kolejki przez ten silnik. Przygotowano osobny,
świeży pilot PPO; wynik jazdy nadal nie istnieje. Diagnoza PowerShell, środowisko,
zakres i polecenie weryfikacji: [PPO_PILOT_20261013.md](PPO_PILOT_20261013.md).
Stary STOP zachowano w archiwum pierwszej kolejki po autoryzacji człowieka.
Pierwsza próba zatrzymała się na timeout telemetrii przed treningiem, procesy
zamknięte. Po trzech kontrolach resetu PASS wystartował świeży pilot
`tmrl-ppo-p13-prefix-r2-s17` o 10:04:27 UTC (12:04 Warszawa); preflight PASS,
W&B `puhyz84h`. Następnie launcher zakończył się przez Windows WinError5 przy
podmianie statusu. Interwencja STOP zamknęła trening z checkpointem 4096 kroków/
2 aktualizacji; wszystkie zarejestrowane procesy zamknięte, mutex wolny.
Oceny jazdy nie wykonano. Poprawiono bounded retry zapisu i sprzątanie dzieci
przy wyjątku; 57 testów PASS, w tym rzeczywisty Windows reader lock.
Źródło r2 niezmienione, kolejka wykorzystana i zatrzymana; nie resume/restart.
R3 `tmrl-ppo-p13-prefix-r3-s17` zakończony normalnie 12:31:46 UTC (14:31 Warszawa):
145408 kroków/71 aktualizacji, 10 kompletnych ocen, **3/10 met — FAIL bramki 8/10**.
Brak błędu runtime/telemetrii/kontrolera, poprawne UID, max odstęp 50 ms.
Checkpoint finite/fingerprint/budget/SHA PASS, pins PASS, wszystkie 15 zapisanych
oryginalnych procesów zamknięte, mutex wolny. Pełny trening PPO nadal zablokowany.
Dowód: `evidence/ppo-pilot-20261013-r3/post-closure-verification.json`.
Diagnoza i plan przed dalszym treningiem: [PPO_DIAGNOSIS_20261013.md](PPO_DIAGNOSIS_20261013.md).
Porażki skupione przy5/41/81% trasy; pierwsza meta treningowa w rolloucie42,
10 met w ostatnich11 rolloutach nie jest wynikiem oceny. Lokalny KL early-stop63/71;
W&B pomija te metryki PPO. Najpierw pełniejsze pomiary i świeży ograniczony
test trendu z góry ustalonego budżetu; bez automatycznego full lub siatki hiperparametrów.
Próg dysku dla długiej kampanii nadal nie jest spełniony.

Kampania na czterech komputerach jest **PREPARED_NOT_LAUNCHED**: pięć algorytmów,
trzy seedy, 8 192 000 kroków/seed. Nie uruchomiono długich treningów. Gotowość
oprogramowania nie oznacza sprawdzenia sprzętu i gry u czterech operatorów.

| Algorytm | Zachowana ocena lokalna | Ograniczenie |
| --- | --- | --- |
| IQN | 29/30, mediana 49,4 s, max 50 ms | lokalna kalibracja/preflight przed startem |
| QR | 29/30, mediana 54,3 s, max 60 ms | lokalna kalibracja/preflight przed startem |
| SAC ciągły | 30/30, mediana 44,5 s, max 50 ms | zastępuje SD-SAC u Borysa |
| TQC | 29/30, mediana 44,3 s, max 60 ms | lokalna kalibracja/preflight przed startem |
| PPO | stary baseline 8/10; późniejsza próba 3/10 | nowy ograniczony pilot przed wielodniowym startem |
| SD-SAC | ostatni 0/10; poprzedni 2/10 | experimental, poza kampanią |

Pierwsze cztery oceny są kompletne i bez błędów. Są dowodami historycznych
konfiguracji na lokalnym komputerze, nie powtórzeniem na nowym commicie/hostach.
Ścieżki `evaluation.json` względem lokalnego `artifacts/tmrl-test-comparison/`:

- IQN: `tmrl-overnight-diag-iqn-s17-benchmark-20261005T054442205511/`.
- QR: `tmrl-repair-diag-qr-loss-sum-s17-retry02-benchmark-20261005T222217918890/`.
- SAC: `tmrl-overnight-diag-sac-s17-benchmark-20261005T063406463907/`.
- TQC: `tmrl-overnight-diag-tqc-s17-benchmark-20261005T061051859378/`.
- PPO baseline: `tmrl-overnight-diag-ppo-baseline-s17-benchmark-20261005T031138975953/`.
- PPO późniejsza próba: `tmrl-repair-diag-ppo-s17-benchmark-20261005T130105216657/`.

Walidacja bieżącego przygotowania jest zapisana osobno w
[evidence/campaign-weeks-20261008/validation.json](evidence/campaign-weeks-20261008/validation.json).
Nie nazywamy syntetycznych aktualizacji dowodem jazdy. Nowe checkouty muszą być
czyste, identyczne i mieć własne klucze W&B/kalibrację. Zachowany lokalny STOP nie
jest automatycznie usuwany. Stare checkpointy wymagają swojego zamrożonego kodu.

Na komputerze autora wykonanie skryptów blokuje polityka PowerShell; nie zmieniano
jej ani nie stosowano obejścia. Kontrole planu w Pythonie i testy przeszły, lecz
skryptowy podgląd kolejki został zablokowany przed wykonaniem. `host-check` zgłasza
taką blokadę przed przydzieleniem runu. Każdy host wymaga zatwierdzonej konfiguracji
wykonywania/podpisywania skryptów oraz lokalnej gotowości gry.

Aktualny plan: [MULTI_COMPUTER_TRAINING.md](MULTI_COMPUTER_TRAINING.md).
SD-SAC jest **experimental**, wyłączony ze standardowych kolejek; scheduler
napraw jest **PAUSED**. Raport i pełny prompt dla Fable 5.1:
[SD_SAC_FABLE_5_1_REPORT.md](SD_SAC_FABLE_5_1_REPORT.md).

Historia: [archiwum READINESS.md](archive/20261008/READINESS.md).
