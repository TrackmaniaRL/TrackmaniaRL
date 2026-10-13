# Świeży pilot trendu PPO — 13 października 2026

Bezpośrednia autoryzacja człowieka: „to rob i odpalaj”, po raporcie
[PPO_DIAGNOSIS_20261013.md](PPO_DIAGNOSIS_20261013.md). Jeden nowy ograniczony
pilot, bez resume, restartu starych kolejek lub automatycznego full.

Run `tmrl-ppo-trend-s17`; kolejka `BASE/ppo-trend-20261013`, nowy frozen runtime
`BASE/ppo-trend-runtime-20261013`. BASE to
`H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison`.
Predecessor: zakończony `tmrl-ppo-p13-prefix-r3-s17`, 3/10 met, pełne closure,
checkpoint71 SHA `f9b5a4309f9e9ce5bce77a2b08c77cb4212400ed7c47d94e9e206d884532ab96`.

## Pytanie i z góry ustalona decyzja

Czy identyczne PPO przy dalszej ekspozycji poprawia deterministyczną jazdę,
czy nadal zawodzi w rejonach5/41/81%? Nowa inicjalizacja seed17;204800 kroków,
100 kompletnych rolloutów2048. Ten sam LR0,0003 z horyzontem8192000, gamma0,995,
GAE0,95, entropy0,01, clipping, Adam, model/GNN/V5/reward/actions i rollout resets.
Nie zwiększamy gamma, nie rozdzielamy optimizerów, nie włączamy normalizacji.

Oceny: dokładnie10 deterministycznych prób po krokach102400 i204800.
Pośrednia służy wyłącznie diagnozie trendu. Tylko końcowa, związana z checkpoint100,
stanowi bramkę>=8/10 plus pełne kontrole runtime/SHA/fingerprint/finite/closure.
Niski wynik pośredni nie zatrzymuje ani nie kwalifikuje treningu. Błąd runtime,
niekompletny ślad, niezgodne UID/checkpoint lub STOP przerywa próbę.
Bez wybierania checkpointu po wyniku, dodatkowej oceny lub przedłużania limitu.

## Zmiany wyłącznie w osobnych adapterach

- `pilot_tools.ppo_diagnostics.InstrumentedPrefixPPO`: takie same operacje Adam,
  clipping i drawy losowe; dodatkowo rzeczywista liczba epok/minibatchy,
  full-batch KL po aktualizacji, explained variance/return/value i normy gradientów.
  Metryki o zerowej wariancji targetów mają osobny znacznik valid, bez NaN.
- `PpoWandbTracker`: jawna allowlista `learner/ppo_*`; te same metryki pozostają
  w lokalnych `events.jsonl`. Nie wysyła dowolnych pól ani sekretów.
- `pilot_tools.train_trend`: jeden hook dokładnie po102400 krokach, używający
  istniejącego mechanizmu checkpoint/evaluation. Zamyka kolektor przed oceną,
  zachowuje RNG Python/NumPy/Torch, po ocenie wraca do zwykłego trenera.
- `pilot_tools.ppo_trace.TraceEvaluator`: bufor maksymalnie3000 kroków na próbę,
  zapis gzip między przejazdami. Reset snapshot, causal raw_current/raw_next,
  issued action, pozycja/postęp/prędkość/geometria/timing i powód zakończenia.
  Dane po akcji są wyraźnie oznaczone, nie są podawane jako current input.

Obie oceny i10 śladów każdej zachowane oddzielnie w
`RUN/evaluations/update-00000050/` i `RUN/evaluations/update-00000100/`.
Pośredni checkpoint kopiowany do `RUN/evaluation-checkpoints/`, aby retencja
zwykłych checkpointów go nie usunęła. Root `evaluation.json` dotyczy tylko finału.
Liczba kroków każdego śladu musi odpowiadać wynikowi próby, indeksy ciągłe,
UID oraz ścieżka/SHA checkpointu zgodne. Nie nadpisujemy wcześniejszych śladów.

## Nadzór i ograniczenia

Cap14400 s od rzeczywistego startu, SAVE+13800 s, HARD+14400 s. W limit wliczone
obie oceny, inicjalizacja i trening. Bez przedłużania lub nowej próby po awarii.
PowerShell7.6.6; CUDA RTX4090; min20 GiB na dysku artefaktów. STOP globalny,
STOP kolejki i nowe/tknięte STOP zawsze wygrywają; bez automatycznych wyjątków.
Jeden controller mutex. Bez UI podczas treningu/oceny.

Nadzór launchera zachowuje bounded retry Windows sharing errors i zamyka własne
procesy przy wyjątku. To proces launchera, nie niezależna usługa systemowa.
Zgodnie z prośbą człowieka osobny monitor TEJ próby co15min, cichy przy normalnym
postępie, powiadomienie po treningu i końcowej ocenie lub awarii. Nie wznawia
schedulera SD-SAC ani innych kolejek, nie uruchamia nowych prób. Przy utracie
launchera ma zatrzymać własny trening i zachować checkpoint. Po zakończeniu
weryfikuje closure i wyłącza własne kolejne kontrole.

## Walidacja przed startem

65 testów PASS: adaptery PPO, realna konfiguracja GNN, guardy launchera, preflight
i lifecycle ewaluacji. CPU1thread porównuje wagi, Adam, stare metryki i RNG po
identycznej aktualizacji bazowego oraz instrumentowanego PPO — wszystko identyczne.
Testy obejmują archiwum checkpoint50 po retencji, oddzielenie finału,
bufor śladu, causal order i zachowanie RNG przez hook oceny.

Rzeczywisty kontrolny forward2048 próbek na CUDA: poprawne kształty logp/value,
finite, CUDA RNG bez zmian;0,594 s cold forward, peak około246 MiB.
Nie wykonano w nim kroku optimizerów ani sterowania grą. Pomiar dotyczy diagnostyki
między rolloutami, nie latency jazdy. Ruff/mypy oraz przygotowanie/pins wymagane.

Wspólne źródła trackmaniarl i experiments helpers, stara kampania i wszystkie
frozen runtime'y pozostają niezmienione. Kod nowych adapterów wyłącznie editable;
plan przypina ich źródła oraz zainstalowany plugin, assety, kalibrację i fingerprint.

## Zamknięcie i diagnoza — 13 października 2026

### Aktualna kontynuacja r2 — ACTIVE

Po bezpośrednim „no rób” osobna kolejka `ppo-trend-continuation-20261013-r2`
wystartowała17:16:52.330600 UTC=19:16:52 Warszawa. PreflightREADY10ms/33fields/58reads,
właściwyUID. Event `train/resumed`17:17:18 UTC potwierdza CP50,102400 kroków/50updates,
nowy segment `cae6d86b511b41b39afdc936a78610ef`; bez nowego losowego startu.
W&B kontynuacji `65msv47w` (pierwszy segment `mbf6v73g` pozostaje historyczny).
Nowy jawnie autoryzowany SAVE21:06:52/HARD21:16:52 Warszawa, targetTOTAL204800/100.
Monitor `ppo-pilot-trendu-co-15-minut` ponownieACTIVE15min, raport po końcowej ocenie
lub awarii. Bezfull/ponownego midpoint50/zmiany algorytmu,config,fingerprint,Adama.
Frozen nadal `d8f2440b22f0dc595fe3946815f5d53de22ec1f9`; helperr2 tylko deleguje
oryginalnemu nadzorcy do NOWEJ kolejki po weryfikacji closure pierwszej próby.
Użyte helpery i zachowaneSTOP mają SHA+mtime w `source-identity.json`; nie edytować.
Wrapper40912ctime1791911807.998136, launcher36212ctime1791911808.043826.
Actual receipts w kolejce, snapshots w `evidence/ppo-trend-continuation-20261013-r2/`.
RuffPASS; pierwszy mypy zgłosił wyłącznie dynamiczne przypisanie QUEUE do modułu
załadowanego przez importlib (attr-defined); nie jest to test treningu ani claimmypyPASS.

### Autoryzowana kontynuacja: preflight zatrzymany, trening nie ruszył

Późniejsza instrukcja człowieka „dokoncz obecny pilot”, następnie jawne
„Tak, kontynuuj z limitem 2 h” autoryzowała pozostałe50 rolloutów z CP50 na
oryginalnym frozen/runtime/config/fingerprint, bez powtórzenia oceny50.
Pierwotny deadline został jawnie zastąpiony nowym limitem7200s, SAVE6600s.
Osobny supervisor `pilot_tools/continue_ppo_trend.py` i kolejka
`ppo-trend-continuation-20261013` zachowują oryginalneSTOP/receipts; tylko dokładny
supervisorSTOP z poprzedniej próby był superseded po jawnej zgodzie człowieka,
z kontrolą jegoSHA co tick. InneSTOP nadal zatrzymują. Bez zmianyPPO/replay.

Wywołanie17:11:24 UTC (19:11 Warszawa) zatrzymało się17:11:35 UTC PRZED treningiem:
`SetForegroundWindow=0`, target593400, foreground1574646,20 prób/5s.
Nie obchodzono zabezpieczenia i nie ponowiono startu. Wszystkie6 obserwowanych
tożsamości zamknięte, mutex wolny,351pins zgodne, CP50SHA niezmieniony,
0 dodatkowych aktualizacji. Monitor pozostajePAUSED, bo nie ma aktywnego treningu.
Do kontynuacji potrzebna interwencja: odblokowana sesja i Trackmania na pierwszym
planie. Użyta kolejka nie może być restartowana; następny start wymaga osobnej
kolejki i świeżego preflight. Dowody: `evidence/ppo-trend-continuation-20261013/`.
Supervisor przeszedł Ruff/mypy i read-only validate przed uruchomieniem.

Pilot zakończył się błędem walidacji pośredniej15:10:10 UTC (17:10 Warszawa),
po102400 krokach/50 aktualizacjach, bez wznowienia i bez końcowej oceny.
Wszystkie10 przejazdów ukończone: średnia51,963s, mediana51,885s, najlepszy51,12s.
To wynik pośredni; nie kwalifikuje pełnej kampanii.

Na10392 decyzje jedna przekroczyła100ms: trial06, krok78, zegar3910→4040ms.
Wywołanie policy.act trwało129,475ms, controller.apply0,364ms;18 pominiętych ramek.
Pozostałe cztery odchylenia od50ms wynosiły60ms. Wszystkie delty surowego zegara
zgadzają się z zapisanym step_race_time_ms. Zatrzymanie było prawidłowe.
Ślad nie rozdziela czasu modelu, synchronizacjiGPU, garbage collection i planistyOS;
nie wolno przypisać źródła ani obiecywać naprawy opóźnień na podstawie tego pomiaru.
Nie uzasadnia to zmianyPPO/LR/clip/entropy ani podniesienia limitu100ms.

Checkpoint50 SHA `7fba0d9bf019aecc65ae6a7924a1614caa0841c9da0d30295e38522934a6dcb6`,
finite/fingerprint/processed102400/fractional_updates0 i wszystkie10 śladów zweryfikowane.
Wszystkie14 obserwowanych tożsamości zamknięte, mutex wolny,351pins oraz predecessorCP
zgodne; własnySTOP zachowany. Monitor PAUSED. Bezpośrednia instrukcja człowieka:
nie wznawiać treningu; przemyśleć i naprawić uzasadnione problemy offline.

Naprawa w editable: komunikat bramki nie nazywa oceny pośredniej „Final evaluation”;
podaje checkpoint, numer próby, rzeczywiste maksimum i limit oraz stan tożsamości,
UID, pomiarów i przerwań. Warunki odrzucenia pozostają identyczne.
31 testów guardów PASS, Ruff PASS. Frozen/usedplan/CP/replay niezmienione.
Dowody: `evidence/ppo-trend-20261013/timing-diagnosis.json` i
`post-closure-verification.json`. Dalsza profilacja wymaga osobnego, uzasadnionego
testu; obecne polecenie nie autoryzuje wznowienia ani nowego pilota.

## Historyczny rzeczywisty start i monitor

Pilot JUŻ uruchomiony 2026-10-13T13:28:06.412619Z (15:28 Warszawa).
Frozen commit `d8f2440b22f0dc595fe3946815f5d53de22ec1f9`, fingerprint
`a0b31a8c2cba88c1bc0f80dd68b4c986600ba99a12c20e2bb027464eaaa461c4`.
Preflight PASS20ms/33fields/55reads/właściwy UID. W&B `mbf6v73g`:
https://wandb.ai/dsc-pjatk-warsaw/my-trackmania-agent/runs/mbf6v73g.
Wrapper31804/ctime1791898081.1019006, launcher44196/ctime1791898081.6307912;
aktualnych potomków czytaj z receipts i monitor-state.json, nie z dawnych PID.

Kontrola13:32:27 UTC:2048 kroków/1 aktualizacja;10 epok/80 minibatchy,
full-batch KL0,00664, EV0,1977 valid; nowe metryki zachowane lokalnie,
brak błędu launchera/STOP. To postęp treningu, nie dowód poprawy jazdy.
SAVE17:18:06 UTC=19:18 Warszawa, HARD17:28:06 UTC=19:28 Warszawa.
Nie uruchamiaj launchera ponownie, nie extend/resume.

Monitor `ppo-pilot-trendu-co-15-minut` rzeczywiście utworzony ACTIVE co15min,
w tym czacie; stary SD-SAC pozostaje PAUSED. Powiadomienie po końcowym wyniku
lub awarii/interwencji, bez rutynowych raportów. Monitor wyłącza własne przyszłe
kontrole po zamknięciu i końcowym raporcie. Pierwsze wywołanie tworzenia monitora
zwróciło błąd brakującego destination; poprawione jawne destination=thread
utworzyło jeden monitor, bez duplikatu.

Dowody przygotowania/startu/monitoringu: `evidence/ppo-trend-20261013/` oraz
oryginalne receipts w kolejce. Pełna kampania nadal BLOCKED do bramki końcowej,
kwalifikacji kolejnych seedów i wymogów infrastruktury.
