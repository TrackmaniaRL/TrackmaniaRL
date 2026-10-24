> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Trening headless V7 na tym komputerze

**Status: gotowy do ograniczonych treningów native-only.** Nie jest to
kwalifikacja wielodniowej kampanii ani porównania szybkości uczenia z Windows.

Ten tryb służy do rzeczywistego treningu **native-only** bez uruchamiania
Trackmanii Windows. Używa aktualnego SDK 1.2.14, neutralnego resetu V7,
rzeczywistych wejść i stanów POST oraz osobnego, źródłowo potwierdzonego czasu
symulacji. Nie wymaga dowodu zegara Windows ani ukończonego ACK Windows.
Warunki niezależnego porównania treningów Windows/headless pozostają odrębne.

Obsługiwany profil to QR-DQN, MLP256, 32 kwantyle, 78 akcji, common6
(rzeczywista pozycja XYZ i prędkość XYZ), CPU float32. Nagroda korzysta z tej
samej przypiętej pełnej polilinii, czasu, odchylenia i oryginalnie potwierdzonej
mety. Domyślnie akcja trwa pięć natywnych interwałów. Dyskonto i kredyt uczenia
zależą od faktycznie wykonanych interwałów, także przy krótszym ostatnim kroku.

Potrzebne są nowe wagi albo checkpoint tego dokładnego profilu. Nie pasują
checkpointy native44, common8, wcześniejsze smoke ani kampania GNN/SD-SAC.
Nasiono steruje modelem, eksploracją i samplerem; nie losuje mapy ani fizycznego
resetu silnika. Sam krótki test aktualizacji wag nie mierzy jakości polityki.

## Gotowa instalacja lokalna

- Źródła: `<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania`.
- Python WSL: `<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python`.
- Oddzielny runtime: `<USER_HOME>/tmrl-headless-training-20261022-007/deployment/`.
- Dowód jednostki: `<USER_HOME>/tmrl-headless-training-20261022-007/clock-proof001/final001/clock.json`.
- SHA256 dowodu: `5fffae5f54ad88b20e757b5ef7e5da5c2c1360d4331b8d782e809d1bef5e9a33`.

Runtime, venv i wyniki pozostają na Linux/ext4. Launcher R8 odczytuje źródła z
osobnego, przypiętego katalogu
`<USER_HOME>/tmrl-headless-performance-20261022-008/optimized-source`.
Zmiany w worktree Windows wymagają osobnego odświeżenia tej kopii; sam worktree
można też jawnie wskazać w `PYTHONPATH`. Nie używaj starszej kopii `src001`. Zamrożone
środowiska, automaty, wyniki i budżet kampanii 7 168 000 pozostają bez zmian.

Launcher z optymalizacjami R8 znajduje się lokalnie w
`F:/AITrackmania/reports/headless-performance-20261022-r8/Uruchom-Szybki-Trening-Headless.ps1`.
Tworzy nowy katalog wyników i wypisuje jego nazwę. Domyślnie uruchamia ograniczoną
próbę 4096 nowych przejść z limitem całej komendy 240 sekund. Korzysta z osobnej
kopii źródeł na ext4 i ponownie używa sesji tylko przy wystarczającym zapasie
czasu i miejsca. Nie uruchamia kampanii ani Windows. Parametr `-FreshSessions`
wybiera osobny sterownik dla każdego epizodu; pozostałe ustawienia uczenia są
takie same. Wyniki pomiarów: [headless-performance.md](headless-performance.md).

```powershell
& 'F:/AITrackmania/reports/headless-performance-20261022-r8/Uruchom-Szybki-Trening-Headless.ps1'
```

Równoważny entrypoint w WSL:

```bash
export SRC=<USER_HOME>/tmrl-headless-performance-20261022-008/optimized-source
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1
<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python \
  -m scripts.headless_native_train \
  --config <USER_HOME>/tmrl-headless-training-20261022-007/deployment/config.json \
  --reference <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/experiments/headless-native16-training-001/reference.json \
  --clock-proof <USER_HOME>/tmrl-headless-training-20261022-007/clock-proof001/final001/clock.json \
  --clock-proof-sha256 5fffae5f54ad88b20e757b5ef7e5da5c2c1360d4331b8d782e809d1bef5e9a33 \
  --output <USER_HOME>/tmrl-headless-performance-20261022-008/runs/NOWA-NAZWA \
  --transitions 4096 --episode-native-units 20000 --native-iterations 5 \
  --replay-capacity 10000 --batch-size 32 --max-wall-seconds 240 --reuse-sessions
```

Każde uruchomienie wymaga nowego output. Liczba `--transitions` oznacza nowe
przejścia w danym uruchomieniu; checkpoint przechowuje dodatkowo licznik
skumulowany. `--episode-native-units 20000` daje 2000 natywnych interwałów,
nominalnie 20 sekund symulacji. Źródłowy zakres kończy się na native42190;
zwykła instalacja ma limit epizodu39950. Zegar ten nie deklaruje początku
wyścigu Windows ani czasu zegara hosta. Szczegóły:
[native-clock-training.md](native-clock-training.md).

## Zapis, wznowienie i zatrzymanie

`latest.pt` zapisuje model, target, optymalizator, RNG, replay, sampler, kredyt
czasu i następny identyfikator epizodu. Zapis jest atomowy, synchronizowany na
dysk, domyślnie co256 przejść oraz po zamknięciu epizodu i normalnym zatrzymaniu.
Błąd kroku lub uczenia zachowuje ostatni kompletny checkpoint. Wznowienie nie
odtwarza żywego snapshotu gry: zaczyna nowy rzeczywisty reset, zachowując cały
zapisany stan uczenia. W tej chwili wspierany jest jeden aktor i CPU.

R8 zapisuje dodatkowo dokładną sumę zapisanych czasów przejść jako
licznik/mianownik. Zapobiega to dryfowi częstotliwości aktualizacji podczas
dłuższego treningu. Starszy checkpoint jest nadal obsługiwany: jego wcześniejsza
przybliżona suma jest zachowana, a kolejne czasy są dodawane dokładnie.

Podaj poprzedni `latest.pt` wraz z jego SHA256 i nowym output. Zachowaj
`seed`, liczbę interwałów akcji, długość epizodu, pojemność replaya i batch size;
program odrzuca zmianę kontraktu lub runtime. W launcherze służą do tego
`-Resume` oraz `-ResumeSha256`; ścieżka checkpointu jest ścieżką WSL.
W Python CLI są to `--resume` oraz `--resume-sha256`.

Utworzenie pliku `STOP` w katalogu tego uruchomienia zatrzymuje kolekcję przed
następnym krokiem i zapisuje stan. Można wskazać własny plik przez `--stop-file`.
Nie usuwa się STOP-ów innych zadań. Przerwanie hosta nie jest metą ani truncation:
ostatni rzeczywisty nieterminalny endpoint pozostaje taki sam w replayu.

Każdy epizod ma świeży proces fizycznego rozgrywania i rzeczywisty neutralny
reset. Opcja `--reuse-sessions` zachowuje zweryfikowany sterownik i jego snapshot
pomiędzy epizodami, gdy konserwatywny zapas pozwala ukończyć kolejny epizod oraz
publikację dowodów. Przed nowym resetem program przechodzi do świeżej sesji,
jeśli brakuje zapasu miejsca albo czasu. Każda sesja ma własny deadline do240s;
nie odnawia się limitu starej sesji. Bez tej opcji każdy epizod dostaje osobny
sterownik. Katalog roboczy pozostaje pod zatwierdzonym
`runtime/build/server-probe`, ponieważ wymaga tego przypięty sterownik.
`report.json` zachowuje ścieżki własnych katalogów roboczych i prywatnych
pozostałych katalogów `/tmp`; są wliczane w budżet, bez kasowania dowodów.

Domyślny łączny limit plików uruchomienia to512MiB; sesja ma własny limit64MiB,
a nowy checkpoint maksymalnie128MiB. Program rezerwuje miejsce na aktywną sesję,
atomowy zapis i ograniczony zewnętrzny journal. Brak miejsca na kolejny epizod
kończy uruchomienie jako jawne zatrzymanie z checkpointem. To kontrolowany limit
aplikacyjny, nie kwota jądra. Dłuższe doświadczenie można kontynuować w nowym
katalogu przez resume, zachowując poprzednie dowody.
Można jawnie zwiększyć budżet przez `--max-run-bytes`; launcher przyjmuje
`-MaxRunMiB`. Dłuższy limit czasu ustawia się przez `--max-wall-seconds` lub
`-MaxWallSeconds`. Limit240s dotyczy każdej pojedynczej sesji silnika, a nie
sumy kolejnych świeżych sesji; program nie uruchamia takiej długiej próby sam.

Raport podaje przejścia, faktyczne interwały, aktualizacje, zmianę wag, kredyt
czasu, rzeczywiste zakończenia/truncations, koszty resetu, cleanup i pełny czas
hosta. `learning_comparison_run=false` oraz `paired_training_qualified=false`
pozostają jawne. Żadna próbka z resetu, awarii ani nieukończonego kroku nie jest
pozorowanym doświadczeniem treningowym.

## Sprawdzone uruchomienia r7

Końcowa regresja: **810 testów PASS w70,39s**, obejmująca core, distributed,
Trackmanię i integrację environment runtime. Ruff check/format przeszedł dla14
zmienionych plików Python; strict mypy dla8 modułów źródłowych, z platformą Linux.
Testy obejmują zawinięcie replaya, tożsamość akcji i wag przy wznowieniu na tej
samej granicy epizodu, STOP, błędny checkpoint, awarię aktora, wczesny EOF,
zerwane połączenie, FIFO oraz limity zapisu. Testy syntetyczne nie zastępują
poniższych rzeczywistych prób silnika.

| Próba | Nowe przejścia | Nowe aktualizacje | Epizody/sesje | Pełny czas procesu |
| --- | ---: | ---: | ---: | ---: |
| `train002` | 2048 | 512 | 6 | 104,20s |
| `resume001` z `train002/latest.pt` | 512 | 128 | 2 | 36,10s |
| `stop001`, STOP po zaobserwowaniu64 wierszy | 65 | 16 | 1, przerwany | 13,07s |
| `resume-stop001` z zapisu STOP | 64 | 16 | 1 | 17,73s |

Główny łańcuch zakończył się na2560 przejściach i640 aktualizacjach, zachowując
replay1000 po jego zawinięciu oraz zerowy całkowity kredyt pozostałych aktualizacji.
Wznowienie zaczęło od dokładnie zapisanych wag, a kolejne aktualizacje je zmieniły.
W obu głównych próbach wykonano pełne epizody o20000 jednostek oraz jawny
krótszy epizod kończący budżet. Wszystkie8 zakończyły się rzeczywistym limitem
środowiska; nie uzyskano mety nieuczoną/krótko uczoną polityką.

STOP nie dopisał truncation do ostatniego przejścia. Zapis zachował także
resztę0,25 kredytu; osobny łańcuch STOP/resume doszedł do129 przejść i32
aktualizacji. Wszystkie10 sesji tych czterech udanych prób mają potwierdzone
zamknięcie. Pierwotne `train001` odrzucone przed pierwszym krokiem pozostaje
jawnym nieudanym wynikiem; osobny dowód zamknięcia obejmuje także jego procesy.

Czasy obejmują cały proces Pythona: importy, walidację, reset, uczenie, zapisy
i cleanup, według `/usr/bin/time -v`. Nie obejmują czasu wcześniejszego kopiowania
runtime ani opóźnienia uruchomienia WSL przez Windows. Podczas pierwszej próby
trwały też regresje syntetyczne; to test poprawności, nie izolowany benchmark.
`maxRSS` raportowany przez `time -v` wynosił365052–391892KiB, swap0. Nie jest to
pomiar sumy szczytowej pamięci całego drzewa procesów ani prognoza jakości modelu.

Raporty tekstowe i instrukcja startu:
`F:/AITrackmania/reports/headless-training-20261022-r7/`.
Surowe dane i checkpointy:
`<USER_HOME>/tmrl-headless-training-20261022-007/`.
`summary-final.json` wiąże końcowy commit, testy, dowód czasu, wyniki treningów,
kontrolę zamknięcia i niezmienione zamrożone źródła.
