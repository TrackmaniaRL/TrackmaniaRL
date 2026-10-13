# Gotowość — 8 października 2026

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
