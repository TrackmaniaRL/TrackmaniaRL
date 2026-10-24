> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R10: szybkość symulacji i uczenia headless

R10 rozszerza [trening wielu aktorów R9](headless-multiactor.md) o wspólne
obliczanie akcji, tańsze przygotowanie danych oraz nakładanie uczenia na pracę
silnika. Model nadal ma profil common6, QR-DQN, MLP256, fixed32 i 78 akcji.
Nagroda, dyskonto, Adam, batch32 oraz stawka aktualizacji za zaakceptowany czas
symulacji pozostają takie same. Domyślny uczeń używa CPU float32 i dwóch wątków.

Końcowy pomiar R10 daje **4,12× czasu rzeczywistego dla jednego aktora**
i **2,79× na aktora przy dwóch aktorach**, razem z uczeniem, checkpointami
i zamknięciem. Największą zmierzoną łączną przepustowość treningu dało osiem
aktorów CPU: **159,50 przejścia/s**, czyli 7,98× czasu symulacji łącznie.

## Co oznacza symulacja 2–3 razy szybsza od czasu rzeczywistego

Podstawą jest czas odczytany z faktycznych, zaakceptowanych interwałów silnika.
W przypiętym kontrakcie native jeden interwał ma nominalnie 0,01 s. Szybkość
symulacji liczymy jako sumę tego czasu podzieloną przez pełny czas procesu
treningowego. Ten mianownik obejmuje importy, start aktorów, resety, uczenie,
checkpointy, publikację dowodów i zamknięcie. Profilowanie diagnostyczne oraz
przygotowanie instalacji nie są częścią porównania szybkości.

Trzeba rozdzielić dwie miary:

- **Szybkość jednego aktora:** jego zaakceptowany czas symulacji na sekundę
  pełnego przebiegu. Wynik 3× oznacza trzy sekundy jego jazdy na sekundę hosta.
- **Łączna przepustowość:** suma zaakceptowanego czasu wszystkich aktorów na
  sekundę hosta. Osiem aktorów może zbierać łącznie 8× czasu rzeczywistego,
  mimo że każdy pojedynczo pracuje z szybkością około 1×.

Przy równym podziale pracy średnia dla aktora to wynik łączny podzielony przez
liczbę aktorów. Przy nierównym podziale lub STOP należy użyć rzeczywistych
czasów poszczególnych aktorów. Liczba decyzji nie zastępuje pomiaru czasu,
ponieważ końcowy krok może zawierać mniej niż pięć interwałów.

Są to miary przypiętego zegara **native-only**. Nie dowodzą zgodności zegara,
trajektorii ani treningu z klientem Windows. Istniejąca kwalifikacja parowania
z Windowsem nadal jest odrębna. Wynik szybkości zbierania doświadczeń także
nie dowodzi krótszego czasu do tej samej jakości jazdy.

## Przygotowanie danych i obliczanie akcji na CPU

Składanie batcha kopiuje rzeczywiste obserwacje i dostępne NEXT do własnych
tablic. Sprawdzenie skończoności odbywa się na tych tablicach oraz tablicy
nagród, zamiast osobno na każdym małym tensorze. Każdy wiersz nadal przechodzi
kontrolę kontraktu, zegara, timestampów, czasu, akcji, flag, źródła, hashy
obserwacji oraz wspólnej nagrody. Hash dotyczy dokładnie skopiowanych danych,
które otrzyma uczeń. Końcowy świeży digest odrzuca zmianę specyfikacji podczas
składania batcha. Nie ma cache walidacji zachowanego między batchami.

`--batch-actions` zbiera obserwacje aktywnych aktorów w jeden przebieg modelu.
Dotyczy wyłącznie przypiętego stateless IdentityTemporalCore, neutralnego risk
i polityki bez selektora akcji. Obserwacje i model polityki pozostają na CPU
w float32. Losowania epsilon i losowej akcji zachowują kolejność ID aktorów
oraz dotychczasowe przeplatanie wywołań RNG.

Macierze o innym rozmiarze batcha mogą zaokrąglać float32 inaczej. Gdy dwa
najlepsze Q są blisko, adapter ponownie liczy dany wiersz z pojedynczym
kształtem batcha. Raport podaje liczbę takich powtórek. To praktyczna kontrola
remisów, **nie uniwersalne ograniczenie błędu ani gwarancja identycznych bitów
Q** dla dowolnych wag. Tryb ma własny profil numeryczny w harmonogramie.

## Ograniczone nakładanie uczenia na symulację

`--overlap-learning` najpierw wydaje akcje jednej rundy, następnie wykonuje
przygotowane aktualizacje, gdy aktorzy symulują ich skutki. Główny wątek nadal
jako jedyny zmienia model, replay, sampler i budżet. Każde środowisko ma
własnego sekwencyjnego właściciela, a komunikacja zachowuje ograniczenie ramki,
deadline oraz powiązanie żądania z konkretnym aktorem.

Próbki są losowane i kopiowane w tej samej granicy dopisania pojedynczego
wiersza replaya co w trybie synchronicznym. Późniejszy aktor z tej rundy nie
może pojawić się wcześniej w takim batchu. Odroczone jest wykonanie
aktualizacji, a nie moment dopuszczenia doświadczenia. Kolejka ma najwyżej
16 przygotowanych aktualizacji i nie zawiera danych z niezaakceptowanej rundy.
Kredyt jest nadal naliczany wyłącznie z czasu zaakceptowanych przejść;
reset, oczekujące wejście, błąd aktora i odrzucona runda przyznają zero kredytu.

Wagi polityki mogą być opóźnione najwyżej o dwie rundy tego schedulera.
Nie oznacza to limitu dwóch aktualizacji optymalizatora: rozliczenie kredytu
po rozgrzewce replaya może wykonać kilka aktualizacji naraz. Wydane i aktualne
wersje wag oraz obserwowane opóźnienie są zapisane w raporcie.

Checkpoint wymaga końca całej rundy i braku kroków w toku. Przygotowane
aktualizacje są wcześniej kończone, również po STOP, ponieważ ich losowanie
już przesunęło RNG samplera. Jest to ograniczona praca na wcześniej
zaakceptowanych danych. Harmonogram wiąże częstotliwość checkpointów:
w trybie overlap zmiana `--checkpoint-every` wymaga innego checkpointu.
Wymuszony koniec kolejki i odświeżenie polityki wpływają na kolejne akcje.
Wznowienie nadal zaczyna świeże fizyczne epizody.

Batching i overlap są jawnie opcjonalne. Samo szybsze składanie danych oraz
ograniczenie kosztów komunikacji nie zmieniają algorytmu. Włączenie overlap
zmienia harmonogram wersji polityki i może zmienić trajektorie. Zgodność
numeryczna starego schedulera nie jest dowodem identyczności nowego.

## Opcjonalny uczeń CUDA

`--learner-device cpu` jest domyślne. `--learner-device cuda` przenosi wyłącznie
ucznia i jego aktualizacje na GPU. Osobna kopia polityki zachowania pozostaje
na CPU float32, także przy `--batch-actions`; eksploracja używa głównego CPU RNG.
CUDA nie jest zaleceniem domyślnym bez pomiaru całego treningu: mały model i
batch32 mogą nie pokryć kosztu transferów, synchronizacji i startu GPU.

Profil CUDA wymaga float32 IEEE, `matmul_precision=highest`, wyłączonego TF32
dla matmul i cuDNN oraz deterministycznych algorytmów. Przed inicjalizacją CUDA
ustawiane jest `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Nieodpowiedni wcześniejszy
kontekst jest odrzucany przed startem aktorów. Zapis harmonogramu wiąże wersję
Torch, runtime CUDA, precyzję, workspace i CPU polityki zachowania. Raport
dodaje nazwę urządzenia, capability i faktyczne ustawienia wykonania.

Cache GPU powstaje w nowym, własnym `output/gpu-cache`, z podkatalogami
`cuda` i `triton`. Istniejący katalog lub symlink jest odrzucany. Budżet
przebiegu i checkpointu rezerwuje dodatkowe 64 MiB, a
`CUDA_CACHE_MAXSIZE=67108864` ustawia rozmiar cache sterownika. Wszystkie
zachowane pliki podlegają także globalnemu rozliczeniu danych przebiegu.

Checkpointy CPU i CUDA mają różne profile wykonania i nie są wymienne.
Zgodność wznowienia dotyczy sprawdzonego urządzenia i środowiska; nie jest
obietnicą identycznych wyników na dowolnej architekturze GPU. Środowisko CUDA
jest odrębne od przypiętego środowiska CPU i zamrożonych runtime'ów.

## Parametry uruchamiania

Zalecany launcher Windows po opublikowaniu wersji R10:

```powershell
& 'F:\AITrackmania\reports\headless-throughput-20261022-r10\Uruchom-Szybki-Trening-Headless.ps1'
```

Domyślnie wybiera osiem aktorów, CPU, batching i overlap. Jest to konfiguracja
z największą łączną przepustowością w końcowych pomiarach. Do szybkiej
pojedynczej symulacji służy `-Actors 1`; `-Actors 2` daje około 2,79× czasu
rzeczywistego każdemu aktorowi i większą sumę doświadczeń niż jeden aktor.
Przykład pojedynczego aktora:

```powershell
& 'F:\AITrackmania\reports\headless-throughput-20261022-r10\Uruchom-Szybki-Trening-Headless.ps1' -Actors 1
```

Launcher wymaga co najmniej **10 GiB wolnego na C:** i **4 GiB na ext4**
przed uruchomieniem procesu treningowego. Kontrolę wykonują launcher
PowerShell i jego wrapper w WSL; weryfikowany jest także przypięty manifest
źródeł. `-ShowCommandOnly` pokazuje argumenty bez treningu.
`-LearnerDevice cuda` wybiera osobny interpreter GPU, `-SerialActions`
wyłącza batching, a `-SynchronousLearning` wyłącza overlap.

Nowe tryby należą do `scripts.headless_native_train_multi` w źródłach R10.
Launcher R9 nadal wskazuje przypiętą wersję R9. Poniższe wywołanie samego
modułu jest przykładem zaawansowanym z nową ścieżką wyników. Nie przechodzi
przez wrapper kontroli wolnego miejsca; aby zastosować opisane progi 10/4 GiB,
należy użyć launchera powyżej.

```bash
cd <USER_HOME>/tmrl-headless-throughput-20261022-010/optimized-source
PYTHONDONTWRITEBYTECODE=1 \
<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python -B \
  -m scripts.headless_native_train_multi \
  --config <USER_HOME>/tmrl-headless-throughput-20261022-010/config.json \
  --reference <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/experiments/headless-native16-training-001/reference.json \
  --clock-proof <USER_HOME>/tmrl-headless-training-20261022-007/clock-proof001/final001/clock.json \
  --clock-proof-sha256 5fffae5f54ad88b20e757b5ef7e5da5c2c1360d4331b8d782e809d1bef5e9a33 \
  --output <USER_HOME>/tmrl-headless-throughput-20261022-010/runs/manual-cpu001 \
  --actors 8 --transitions 8192 --native-iterations 5 \
  --episode-native-units 20000 --replay-capacity 10000 --batch-size 32 \
  --checkpoint-every 256 --max-wall-seconds 240 --max-run-bytes 2147483648 \
  --reuse-sessions --batch-actions --overlap-learning --learner-device cpu
```

Każde uruchomienie wymaga nowego katalogu wyników. `--transitions` oznacza
łączną liczbę przejść wszystkich aktorów. `--actors` dopuszcza 1–8; dobór
liczby aktorów zależy od tego, czy celem jest tempo pojedynczej symulacji,
czy łączna liczba doświadczeń dla ucznia. Oba wyniki należy mierzyć osobno.

Wariant CUDA wymaga interpretera z odrębnego, zweryfikowanego środowiska CUDA
oraz `--learner-device cuda`. Samo podanie flagi w środowisku CPU kończy się
jawną odmową. Wznowienie używa `--resume`, `--resume-sha256` i zgodnego
harmonogramu, w tym obu opcjonalnych trybów i urządzenia ucznia.

## Końcowe pomiary i dowody

Każdy przebieg końcowej serii wykonał **8192 rzeczywiste przejścia N5,
40960 interwałów i 2048 aktualizacji**, czyli 409,6 s zaakceptowanej symulacji
łącznie. Wszystkie dziesięć prób przeszło. Biegły kolejno, bez równoległych
testów obciążających host, a czas obejmował cały proces. Dla trzech powtórzeń
tabela podaje medianę; pozostałe konfiguracje mają po jednym pomiarze.

| Konfiguracja | Powtórzenia | Pełny czas / mediana | Przejścia/s | Łączna symulacja / czas hosta | Średnio na aktora |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kontrola R9, 8 aktorów CPU | 3 | 57,9981 s | 141,25 | 7,06× | 0,88× |
| R10, 1 aktor CPU | 1 | 99,3707 s | 82,44 | 4,12× | **4,12×** |
| R10, 2 aktorów CPU | 1 | 73,4724 s | 111,50 | 5,57× | **2,79×** |
| R10, 4 aktorów CPU | 1 | 56,2073 s | 145,75 | 7,29× | 1,82× |
| R10, 8 aktorów CPU | 3 | **51,3601 s** | **159,50** | **7,98×** | 1,00× |
| R10, 8 aktorów CUDA | 1 | 75,8975 s | 107,94 | 5,40× | 0,67× |

R10 CPU z ośmioma aktorami ma **1,129× przepustowości świeżej kontroli R9**:
około **12,92% więcej przejść/s** i **11,45% mniej czasu** przy tej samej
łącznej pracy. Surowe czasy kontroli R9: 64,146583570 / 57,960471754 /
57,998104842 s. R10 CPU8: 55,466245379 / 51,360119645 / 51,126007264 s.
CUDA na tym małym modelu była wolniejsza od CPU i pozostaje opcją do osobnych
eksperymentów, a nie domyślnym wyborem.

Cel co najmniej 2–3× czasu symulacji na sekundę hosta spełnia jeden aktor
R10, a dwa aktory przekraczają 2× każdy. Osiem aktorów służy największej
łącznej liczbie doświadczeń: wynik 7,98× nie oznacza takiego tempa każdego
samochodu. Średnia CPU8 przed zaokrągleniem wynosi 0,99688× na aktora.

Pierwsza, wcześniejsza seria rozpoznawcza R10 CPU8 dała 58,611813933 s dla
kontroli i 47,694916411 s dla trybu łączonego. Te pojedyncze próby pozostają
zachowanymi wynikami wstępnymi; końcowe wnioski powyżej opierają się na nowej
kontroli i trzech powtórzeniach wersji końcowej.

Osobne, wcześniejsze serie R9 dały medianę 54,67355 s dla ośmiu aktorów,
a bezpośredni R8 jednego aktora 91,91873 s za tę samą łączną pracę. R8 oznacza
około **4,46× nominalnego czasu rzeczywistego dla pojedynczego aktora**, razem
z uczeniem i checkpointami. Nie jest to zmierzony mnożnik względem działającej
gry Windows. Zysk R10 w końcowej serii liczymy ze świeżej mediany 57,9981 s;
historyczne mediany służą jako osobny kontekst. Mnożenie zysków z różnych
profili nie tworzy pomiaru tego samego treningu.

Walidacja obejmuje zgodność próbek i RNG, pełnego ucznia i Adama, rzeczywiste
interwały silnika, błędy i STOP, wznowienie na granicy fizycznych epizodów,
odrzucanie niezgodnego harmonogramu oraz zamknięcie własnych procesów.
Przy nowym schedulerze wymagane są także jego własne powtórzenia; samo
zachowanie liczby aktualizacji nie dowodzi identyczności danych lub jakości.

Na odrębnej instalacji CUDA zaliczono 20 testów modułu urządzenia ucznia,
w tym sześć przypadków z rzeczywistymi aktualizacjami na GPU. Obejmują one
cztery kombinacje batching/overlap, dokładne wznowienie pełnego stanu ucznia,
RNG, samplera i budżetu na granicy epizodów oraz odrzucenie checkpointu CUDA
przez profil CPU. Rzeczywiste próby CUDA objęły także STOP i wznowienie oraz
awarię workera i ucznia. To sprawdzenie działania i zamknięcia procesów;
wybór urządzenia dla wydajności wymaga osobnych pełnych pomiarów czasu.

Artefakty tej fazy pozostają na ext4:
`<USER_HOME>/tmrl-headless-throughput-20261022-010/`.
Małe raporty, plany porównań i końcowe podsumowanie są publikowane w
`F:/AITrackmania/reports/headless-throughput-20261022-r10/`.
Opis podstawy porównania z Windowsem znajduje się tam w
`BASELINE-AND-EVALUATION.md`; aktualne wnioski o szybkości muszą wskazywać
wersję źródeł i konkretne pomiary z tej fazy.
