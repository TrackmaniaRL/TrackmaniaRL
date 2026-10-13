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

## Rzeczywisty start i monitor

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
