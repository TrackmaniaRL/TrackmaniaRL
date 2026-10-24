> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R5: rzeczywista meta przez publiczny adapter V7

**Próba `public-source-reference001` zakończyła mapę przez niezmieniony publiczny
adapter: 1780 rzeczywistych kroków action78/N1, wynik 17715 ms, jeden checkpoint,
zero respawnów.** `NativeImmediateEnvironment` zwrócił `terminated=true`,
`truncated=false`, EOF w fazie `reader`, ostatni POST native 20000. Cleanup ma
returncode 0, brak błędów i `remainingOwned=[]`.

Walidator zachował `IsValid=false`. Ta eksperymentalna taśma deklaruje 40000 ms,
a obliczony czas 17715 ms jest lepszy; w `Desc` jest także informacja o walltime.
To potwierdzenie obliczonej fizycznej mety przez oryginalny silnik. Nie jest to
rankingowa akceptacja ghosta. Produkcyjny classifier zachowuje tę różnicę.
Nie uruchomiono modelu, optymalizacji ani replaya learnera.

## Przyczyna wcześniejszego braku mety

Przypięta taśma 4152 zawiera pięć zmienionych początkowych rekordów Jimmy’ego.
Porównano wszystkie 1922 paczki z ustaloną semantyką SGB z dostarczonego eksportu:
1917 jest bitowo zgodnych, a indeksy 151..155 / input race 0..40 mają w runtime
steer −0.5039370059967041 zamiast źródłowego 0. Gaz 1 i hamulec 0 są takie same.
Paczka 0 ma mode 15 i nie otrzymała zgadniętej semantyki SGB.

| Źródło | SHA256 |
| --- | --- |
| `TrackmaniaRL-RE-zrodla-i-kontekst-20261021.zip` | `863ae95ff8a8d1e5339aea255b2675fd2d09eb652f8f6d9af4f437317bd346b9` |
| `export-001/jimmy-17715.json` w archiwum | `214b54f8b79c40af5db23375bd88ab9ff817d2de98315245de2a9c6792887777` |
| `assets-001/jimmy-17715.Ghost.Gbx` w archiwum | `2890dcbd55356441304097afd94608cc4b166c89ef3fda2fe33dd817dabc8911` |

Nie poprawiano pliku taśmy, binarki, produkcyjnych pinów ani resetu. Zwykłe
wydanie prawidłowych akcji przez V7 zastępuje zmienione rekordy przed odczytem.

## Trzy osobne próby

Wszystkie katalogi znajdują się pod
`<USER_HOME>/tmrl-headless-parity-20261021-005/`.

| Próba | Rzeczywisty wynik |
| --- | --- |
| `drain-original001` | Jawny op2 przy POST 2190 bez zmiany wejść. Pełny oryginalny JSON, ale `ValidatedResult=null`, `Desc=wrong simu…`. 4298 kolejnych POST 2030..45000, 43 TMRESULT. Brak dowodu mety. |
| `drain-source-repair001` | Dokładnie pięć źródłowych SGB `[0,1,0]` przez op6 N5, potwierdzony POST 2240, następnie op2. Pełny JSON i TMRESULT zgodnie potwierdzają 17715 / CP1 / respawn 0. 1798 kolejnych POST 2030..20000. |
| `public-source-reference001` | Standardowy neutralny reset 2200, potem publiczne action78/N1. 1780 przejść zakończonych naturalnym reader EOF i produkcyjnym `terminated=true`. Op2 nie był używany. |

We wszystkich trzech próbach normalny reap potwierdza zachowanie rodzica,
a końcowy cleanup ma returncode 0 i zero pozostałych właścicieli. Czasy hosta
3,16 s / 3,41 s / 28,00 s są zapisami diagnostycznymi z równolegle używanego komputera,
bez kwalifikacji benchmarku. Sesje mają limit 120 s i jawny budżet 64 MiB; runner
publiczny ma dodatkowy limit 2000 decyzji. Nie zmieniano kontraktu V7 ani jego
bezpiecznego odrzucania ostatniego readera.

W wariancie drain celowe zamknięcie IPC po op2 **nie oznacza mety**. O wyniku
decydują dopiero normalny reap, zamknięta tablica oryginalnego validatora i własny
TMRESULT. Autonomiczne POST po release mają zero kredytu decyzji. Osobny
`production-outcome002.json` podał rzeczywisty ostatni POST 20000 i te dowody do
niezmienionej `ImmediateSession.classify_terminal`: wynik `terminated=true`,
`validationAcceptance=false`, `physicalFinishNativeT=null`. Nie przedstawiono
autonomicznego drain jako przejścia RL.

Publiczny runner korzysta z dokładnych N1 SGB. Cały źródłowy przejazd używa tylko
ośmiu takich wartości, obecnych w istniejących action IDs
`[0,1,3,4,37,39,72,75]`. Zduplikowane N1 znaczenia mają deterministycznie wybrany
najmniejszy indeks; tabela 78 pozostaje niezmieniona. Nie ma kwantyzacji do
najbliższego skrętu. Po resecie p152 wykonano 1771 źródłowych wejść i dziewięć
literalnie neutralnych rekordów istniejącego ogona. Wszystkie otrzymały własne
produkcyjne receipt/audit/POST i surowy journal. Learner otrzymał zero doświadczeń.

## Zegar i referencja

`cold-comparison001.json` porównał naprawiony drain ze starym przypiętym cold
POST828. Dla tych samych nativeT 2030..20000 zgodnych jest **1798/1798 native76
bitowo**, w tym wszystkie 40 B fizyki. Diagnostyczne porównania ±10 pasują tylko do
15/16 powtarzających się początkowych stanów; nie przyjęto nowego offsetu ani
tolerancji. Jest to dowód zgodności dwóch natywnych śladów na dłuższym odcinku,
a nie dowód opóźnienia Windows lub globalnej transformacji zegara klienta.

Ostatni potwierdzony wynik jest dostępny przy native 20000. Nie utożsamiono tego
ticka z fizycznym momentem przekroczenia geometrii mety. Pole
`physicalFinishNativeT` pozostało null; obliczony Time 17715 jest osobnym odczytem
oryginalnego wyniku.

## Powtórzenie

Polecenia wykonuje się w Ubuntu/WSL. Należy podać świeże nazwy katalogów,
ponieważ dowody otwierają się wyłącznie jako nowe pliki. Przygotowanie osobnego
runtime kopiuje 12 jawnie zweryfikowanych zasobów i nie uruchamia silnika:

```sh
export PYTHONPATH=/mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
export PYTHONDONTWRITEBYTECODE=1
PY=<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python
REPO=$PYTHONPATH

$PY "$REPO/scripts/headless_provision_immediate.py" \
  --v6-config <USER_HOME>/tmrl-headless-parity-20261021-001/config.json \
  --shim <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/build/forkoracle/native-immediate-v7-001/libforkshim.so \
  --destination <USER_HOME>/tmrl-headless-parity-20261021-005/production-profile001

$PY "$REPO/scripts/headless_finish_drain.py" \
  <USER_HOME>/tmrl-headless-parity-20261021-001/config.json \
  <USER_HOME>/tmrl-headless-parity-20261021-002/native-input-v7/build-receipt.json \
  <USER_HOME>/tmrl-headless-parity-20261021-005/drain-source-repair001 \
  /mnt/f/AITrackmania/imports/headless-405-20261021/downloads/TrackmaniaRL-RE-zrodla-i-kontekst-20261021.zip

$PY "$REPO/scripts/headless_finish_reference.py" \
  <USER_HOME>/tmrl-headless-parity-20261021-005/production-profile001/config.json \
  <USER_HOME>/tmrl-headless-parity-20261021-005/drain-source-repair001/jimmy-source-export.json \
  <USER_HOME>/tmrl-headless-parity-20261021-005/public-source-reference001
```

Powyższe nazwy odpowiadają zachowanym wykonanym próbom; ponowne użycie ich ma
zostać odrzucone. Runner zapisuje własne źródło i konfigurację. Testy sprawdzają
niepełne JSON/TMRESULT, obcego właściciela, błędne checkpointy/czas/clock,
nieautoryzowane audyty oraz dokładne mapowanie źródła na action78. Regresje
obejmują także istniejący adapter i guard końcowego readera. Końcowy wynik to
**36 PASS w 3,81 s**, bez uruchamiania silnika lub operacji Windows. Dokładne
argumenty i SHA źródeł znajdują się w `finish-tests-final002/receipt.json`.

Testy można odtworzyć w tym samym WSL, podając świeży katalog tymczasowy:

```sh
mkdir <USER_HOME>/tmrl-headless-parity-20261021-005/finish-tests-repeat001
cd "$REPO"
$PY -m pytest -q \
  --basetemp=<USER_HOME>/tmrl-headless-parity-20261021-005/finish-tests-repeat001/tmp \
  -o cache_dir=<USER_HOME>/tmrl-headless-parity-20261021-005/finish-tests-repeat001/cache \
  tests/unit/trackmania/test_headless_finish_drain.py \
  tests/unit/trackmania/test_headless_finish_reference.py \
  tests/unit/trackmania/test_immediate_final_reader_boundary.py \
  tests/unit/trackmania/test_immediate_environment.py
```

Indeks `F:/AITrackmania/reports/headless-training-parity-20261021-r5/finish-native-index.json`
wiąże najważniejsze raporty i ich SHA z dowodami ext4. Nie skopiowano prywatnych
binarek silnika, shima ani ghosta do katalogu raportów.
