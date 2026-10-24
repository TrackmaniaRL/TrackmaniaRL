> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Równoległe środowiska native headless

Ta strona zachowuje opis i wyniki fazy R9. Nowsze tryby CPU batching,
nakładania uczenia na pracę silnika i opcjonalnego ucznia CUDA opisuje
[szybkość symulacji i uczenia R10](headless-throughput.md), wraz z rozróżnieniem
tempa pojedynczego aktora i łącznej przepustowości.

Tryb wielu aktorów korzysta z tego samego common6, modelu QR-DQN i zegara
native-only co [sprawdzony trening jednego aktora](headless-native-training.md).
Jeden proces posiada model, replay, sampler i dokładny budżet aktualizacji.
Każdy aktor ma własny proces Python, środowisko, prywatne gniazda, katalogi,
procesy jazdy oraz supervisor. Aktorzy nie uruchamiają osobnych uczniów.

## Kolejność danych i uczenia

Runda wybiera akcje kolejno według identyfikatorów aktorów, korzystając
z wag dostępnych na jej początku. Kroki silnika biegną równocześnie.
Po zakończeniu rundy wszystkie odpowiedzi przechodzą walidację przed
dopisaniem któregokolwiek przejścia. Replay i aktualizacje są obsługiwane
w stałej kolejności aktorów. Nowe wagi obowiązują od następnej rundy.
Wydana wersja polityki jest jawna; jest to ograniczone opóźnienie zachowania
w uczeniu off-policy, a nie równoważna trajektoria między różnymi liczbami aktorów.

Budżet uczenia sumuje wyłącznie czas faktycznie zaakceptowanych przejść
wszystkich aktorów. Krótki interwał kończący epizod przyznaje odpowiednio
mniejszy kredyt. Liczba aktualizacji nie jest mnożona ponownie przez liczbę
aktorów. Reset, nieukończony krok i nieudana runda nie przyznają kredytu.
Każdy epizod zachowuje własne obserwacje, endpointy, kontrolę i journal.

Błąd jednego aktora odrzuca rundę jako doświadczenie treningowe. Poprawne
odpowiedzi pozostałych są dowodami, a nie dopisywanymi po awarii próbkami.
Trening kończy się jawnie błędem, zachowując poprzedni poprawny checkpoint
i zamykając własne procesy. Awaria ani STOP nie dopisują fikcyjnej mety lub
truncation do ostatniego rzeczywistego przejścia.

## Izolacja i granice

Procesy aktorów powstają przez `spawn`, zanim otworzą sesje silnika.
Każde środowisko jest obsługiwane sekwencyjnie przez swojego właściciela.
Komunikacja wiąże aktora, żądanie i odpowiedź; publikacja checkpointu odbywa
się na granicy rundy, gdy nie ma kroków w toku.

Najpierw przydzielane są osobne ścieżki. Start natywny następuje dopiero po
kontroli wspólnego budżetu. Przy rollover najpierw zamykana jest stara sesja,
następnie naliczane zachowane pliki, a dopiero potem dopuszczany nowy start.
Rezerwa obejmuje sumę limitów aktywnych sesji, atomowy checkpoint i journale.
Każda sesja zachowuje własny limit 64 MiB i deadline do 240 s; ponowne użycie
nie odnawia jej czasu. Bezpieczne zamknięcie oraz zapis raportu mogą potrwać
po upływie deadline.

Pool używa tożsamości własnych procesów przy awaryjnym zamykaniu. Nie wysyła
sygnałów do wszystkich procesów gry, wspólnego użytkownika ani cudzej grupy.
Supervisor każdej sesji zachowuje niezależną odpowiedzialność za jej potomków.

## Checkpointy

Wielu aktorów ma osobny wersjonowany format checkpointu. Zapis wiąże kontrakt,
liczbę aktorów, scheduler, odświeżanie polityki i liczniki epizodów oraz zachowuje
wspólne wagi, optimizer, replay, sampler, RNG i dokładny kredyt czasu.
Wznowienie rozpoczyna świeże fizyczne epizody; nie odtwarza nieistniejącego
snapshotu jazdy. Checkpoint innego formatu lub liczby aktorów jest odrzucany.
Dotychczasowy launcher R8 pozostaje dostępny dla zapisów jednego aktora.

## Uruchamianie

Gotowy launcher Windows:
`F:/AITrackmania/reports/headless-multiactor-20261022-r9/Uruchom-Trening-Wielu-Aktorow.ps1`.
Gra Windows nie jest potrzebna. Backend pracuje w Ubuntu WSL, ze źródeł,
wyników i checkpointów zapisanych na ext4.

```powershell
& 'F:\AITrackmania\reports\headless-multiactor-20261022-r9\Uruchom-Trening-Wielu-Aktorow.ps1'
```

Domyślnie: 8 aktorów, 4096 łącznych przejść, replay 10000, batch 32, N5,
epizody do 20000 jednostek native, 240 s pełnego treningu i 2 GiB danych.
`-Actors` wybiera od 1 do 8; `-Transitions` oznacza łączną pracę wszystkich
aktorów. Wypisana ścieżka zawiera `report.json`, `latest.pt` i journale.
Plik `STOP` w tym katalogu zatrzymuje trening na granicy rundy.

Wznowienie wymaga `-Resume` z absolutną ścieżką ext4 do checkpointu,
`-ResumeSha256` z jego hashem oraz zgodnej liczby aktorów i parametrów
harmonogramu. Powstaje nowy katalog wyników. `-FreshSessions` wyłącza ponowne
używanie sesji; `-ShowCommandOnly` pokazuje argumenty bez uruchamiania silnika.

Launcher sprawdza manifest źródeł, zapisuje obok wyników kontekst uruchomienia
z przestrzeniami PID i mount oraz pilnuje dodatkowego limitu awaryjnego.
Awaryjny sygnał jest skierowany wyłącznie do własnego procesu treningowego.

Źródła R10 dodają do `scripts.headless_native_train_multi` opcje
`--batch-actions`, `--overlap-learning` i `--learner-device cpu|cuda`.
Domyślnie oba tryby są wyłączone, a uczeń pracuje na CPU. Launcher R9 korzysta
ze swoich przypiętych źródeł; nowe opcje wymagają uruchomienia wersji R10.
Overlap obsługuje batch32 i wiąże w checkpoincie także `--checkpoint-every`.
Przy wznowieniu muszą zgadzać się tryby oraz profil urządzenia. Pełny przykład,
granice opóźnienia polityki i warunki CUDA znajdują się w
[instrukcji R10](headless-throughput.md#parametry-uruchamiania).

## Pomiary

Sprzęt: i9-13900K, Ubuntu WSL, 24 dostępne vCPU, PyTorch 2.11.0 CPU.
Model, replay 10000, batch 32, dwa wątki ucznia i częstotliwość aktualizacji
pozostały stałe. Każdy dłuższy przebieg zebrał **8192 rzeczywiste przejścia,
40960 interwałów native i 2048 aktualizacji**. Czas obejmuje cały proces:
start, resety, zbieranie danych, uczenie, checkpointy i zamknięcie. Próby biegły
kolejno, bez równoległych testów obciążających CPU.

| Konfiguracja | Powtórzenia | Mediana czasu | Łączne przejścia/s |
| --- | ---: | ---: | ---: |
| R8, jeden aktor bez poola | 3 | 91,92 s | 89,12 |
| R9, jeden aktor przez pool | 1 | 97,54 s | 83,99 |
| R9, dwa aktory | 1 | 74,60 s | 109,81 |
| R9, cztery aktory | 3 | 60,88 s | 134,57 |
| R9, osiem aktorów | 3 | **54,67 s** | **149,83** |

Osiem aktorów skróciło pełny przebieg o **40,5%**, czyli dało około **1,68×**
przepustowości sprawdzonego R8 przy tym samym łącznym budżecie uczenia.
To najszybsza przetestowana konfiguracja tego hosta. W krótkiej próbie 2048
przejść cztery aktory potrzebowały 20,14 s, a osiem 20,67 s: koszt startu ma
większy udział. Dla pojedynczego aktora dotychczasowy bezpośredni R8 osiąga
lepszy czas niż dodatkowy pool.

Każda z trzech dłuższych prób R8: 92,3201 / 91,9187 / 91,3249 s.
Cztery aktory: 60,6680 / 60,9221 / 60,8769 s.
Osiem aktorów: 54,6735 / 54,3870 / 54,8219 s.
Powtórzenia tej samej konfiguracji porównano również po wszystkich fizycznych
interwałach, wejściach, nagrodach i pełnym stanie ucznia, RNG, samplera i budżetu.

Bezpośredni R8 zachowuje wcześniej udowodnione **2,52×** przyspieszenie względem
R7 przy identycznej trajektorii i modelu; szczegóły w
[raporcie jednego aktora](headless-performance.md). Zmiana liczby aktorów
wprowadza inne trajektorie, więc obu mnożników nie należy przedstawiać jako
jednego pomiaru tego samego treningu. Jakość jazdy wymaga osobnego dłuższego
uczenia i ewaluacji. Zegar pozostaje native-only, bez kwalifikacji zegara gry
Windows ani kampanii GNN/SD-SAC.

Wyniki należy rozróżniać na start aktorów, kroki środowiska bez uczenia oraz
cały trening ze wspólnym uczniem. Porównanie pełnego treningu utrzymuje model,
batch, replay, budżet przejść, zegar i stawkę aktualizacji. Zmiana liczby aktorów
zmienia zbierane trajektorie i nie dowodzi takiej samej jakości modelu w tym
samym czasie. Powtórzenia tej samej konfiguracji mają stały scheduler.

Rzeczywiste próby startu dla 1, 2, 4 i 8 aktorów przeszły. Każdy aktor wykonał
dwa osobne epizody, łącznie 48 decyzji N5. Sprawdzenie objęło równoczesne życie
procesów po resecie, odrębne ścieżki i gniazda, wszystkie stany native76 oraz
zastosowane wejścia względem wzorca jednego aktora. Próby zabicia własnego
workera podczas startu i po gotowości czterech aktorów także potwierdziły
izolację i zamknięcie procesów. Nie uruchamiały ucznia i nie przyznawały kredytu.

Pierwszy pełny trening jednego aktora przez pool dał dokładnie te same wejścia,
nagrody, stany fizyki, wagi, optimizer, RNG, sampler i budżet co niezmieniony R8.
Po optymalizacji sterownika testy dla 1, 3 i 4 aktorów potwierdziły identyczność
rekordów i pełnego stanu uczenia względem pierwszej wersji. Pomijanie ponownego
kopiowania niezmienionej polityki oraz grupowanie globalnego rozliczenia plików
zachowują kontrolę każdego resetu i maksymalnie 64 nowe wiersze między
rozliczeniami. Zamknięcie aktora wymusza świeże rozliczenie przed kolejną pracą.

Końcowy zestaw obejmujący core, distributed, Trackmania i integrację runtime
dał **990 zaliczonych testów**. Ruff, formatowanie i strict mypy nowych modułów
również przeszły. Testy uruchomiono na odrębnej kopii ext4; kod używany w
pomiarach zachował te same hashe.

Rzeczywisty STOP dwóch aktorów przy 256 przejściach, na wspólnej granicy
epizodów, oraz wznowienie kolejnych 256 dały identyczne dane fizyczne i pełny
stan ucznia, RNG, samplera i budżetu co nieprzerwane 512 przejść. STOP zachował
63 wykonane aktualizacje i należny kredyt; wznowienie rozliczyło go przed
kolejnymi akcjami i zakończyło 128 aktualizacjami. Zwykłe wznowienie w środku
epizodu nadal zaczyna świeżą jazdę i nie obiecuje identycznej trajektorii.

Zabicie wyłącznie własnego workera po zapisie 256 przejść nie dopisało
nieudanej rundy i zachowało bajty poprzedniego checkpointu. Zapis dwóch aktorów
odrzucono przy żądaniu trzech, zanim powstał pool. Osobny przebieg 66 przejść
objął 44 kroki N5 i 22 rzeczywiste końcówki N3: 286 interwałów, 2,86 s kredytu,
14 aktualizacji. Nie zastępowano końcówek pełnymi krokami ani fikcyjną metą.

Próba nierównego przydziału 257 przejść ośmiu aktorom także przeszła.
Siedmiu aktorów zakończyło po 32 przejścia, ósmy po 33. Po zamknięciu
pozostałych wykonano świeże rozliczenie plików przed ostatnim krokiem.

Końcowy audyt rozdziela aktualne tożsamości procesów od historycznych
potwierdzeń zamknięcia, ponieważ WSL zmienia przestrzenie PID i mount.
Profil diagnostyczny `profile-r9-a1-001` nie ma pierwotnego zapisu tych
przestrzeni; zachowano jego potwierdzenia supervisora bez porównywania starych
numerów PID do aktualnych. Ten profil nie stanowi podstawy wyników szybkości.
Pomiary i właściwe próby poprawności posiadają zapis kontekstu uruchomienia.
Audyt sprawdza również kod, przypięte assety i źródła dowodu zegara.

Wyniki, checkpointy i źródła tej fazy pozostają na ext4:
`<USER_HOME>/tmrl-headless-multiactor-20261022-009/`.
Małe raporty i sposób uruchamiania:
`F:/AITrackmania/reports/headless-multiactor-20261022-r9/`.
Klient Windows i zamrożona kampania GNN/SD-SAC nie są częścią tych uruchomień.
