> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Wydajność treningu headless R8

Profil pomiaru to CPU float32 QR-DQN, MLP256, fixed32, common6, 78 akcji,
batch32, replay1000, N5, epizody20000 jednostek native, checkpoint co256 przejść,
seed1729. Każda próba zaczyna od tych samych wag i RNG. Limit próby jest jawny;
nie uruchamia się kampanii ani klienta Windows.

## Co obejmuje pomiar

Podstawowa miara to czas całego nowego procesu Pythona: importy, weryfikacja
zasobów, resety, rzeczywiste kroki silnika, aktualizacje modelu, checkpointy,
publikacja dowodów i zamknięcie własnych procesów. Nie obejmuje przygotowania
instalacji ani uruchamiania WSL przez Windows. Źródła obu wersji, runtime, venv
i wyniki leżą na ext4. Próby porównawcze biegną kolejno, bez równoległych testów
CPU. Osobne profile `cProfile` służą do znalezienia kosztów; nie są pomiarem
przyspieszenia bez narzutu profilera. Czasy zagnieżdżonych wywołań nie sumują się.

Punkt odniesienia to commit `e2402f0d0be91c2d68b587af59dfd5d64920f151`.
Trzy świeże próby po2048 przejść i512 aktualizacji trwały64,159s,64,416s oraz
62,483s. Mediana wynosi64,159s, czyli31,921 przejścia/s. Wszystkie miały te same
akcje, nagrody, fizyczne stany każdego interwału i końcowe wagi.

Końcowe źródła R8, po poprawce ograniczonego odczytu cleanup, dały:

| Wersja | Pełne czasy trzech prób | Mediana | Przejścia/s |
| --- | --- | ---: | ---: |
| R7, osobna sesja na epizod | 64,159 / 64,416 / 62,483s | 64,159s | 31,921 |
| R8, bezpieczne ponowne użycie sesji | 26,882 / 25,375 / 25,500s | 25,500s | 80,314 |

To **2,516 raza szybciej i60,26% mniej czasu** dla tego samego profilu.
Każda próba wykonała2048 przejść,10240 rzeczywistych interwałów i512 aktualizacji.
Sześć epizodów zachowano; liczba sesji sterownika spadła z6 do2. Porównanie
wszystkich trzech prób obu wersji potwierdziło zgodność całej trajektorii
fizycznej, danych uczenia oraz końcowego modelu
`f43a68a04be5b8bd7b2895474e75acc35a80439f359879ffd5196d089c45a95e`.
Identyczne wagi dotyczą tej sprawdzonej sekwencji; dłuższy trening korzysta
z naprawionego dokładnego licznika czasu opisanego poniżej.

Wynik R7:2560 przejść/140,30s pochodził z mieszanej walidacji, z innym obciążeniem
maszyny. Nie jest używany jako mianownik zysku R8. Dawny wynik148 przejść/s
dotyczył innego kontraktu/modelu/backendu i także nie jest porównaniem.

## Zmiany bez zmiany uczenia

- Sesja silnika może obsłużyć kilka epizodów. Każdy ma osobny proces fizycznej
  jazdy, neutralny reset i journal. Przed nowym epizodem kontrolowany rollover
  zachowuje limit czasu sesji do240s oraz limit dowodów64MiB, z rezerwą na
  następny epizod, kopie i metadane zamknięcia.
- Licznik rozmiaru plików używa zdarzeń Linux inotify i świeżych metadanych
  zmienionych inode. Watch obejmuje też zapis przez zewnętrzny hardlink lub
  istniejący deskryptor. Każda nazwa hardlinku jest osobno naliczana. Przed
  publikacją końcową następuje pełne uzgodnienie; brak inotify lub limit watch
  przełącza do dokładnego pełnego skanu. Nie ma cache z opóźnieniem czasowym.
- Awaryjny pełny skan używa `os.scandir` i deskryptorów katalogów bez podążania
  za linkami, ograniczając powtórne odczyty metadanych. Kontrola przypiętych
  linków pozostaje świeża. Linux może zaakceptować dokładną dozwoloną ścieżkę
  rozwiązaną przez jądro z deskryptorem `O_PATH`; niezgodność, błędy procfs,
  usunięty cel lub niejednoznaczny wynik wracają do pierwotnego strict resolve.
  Dozwolone są wyłącznie dwa prywatne gniazda sterownika, `control.sock` i
  `branch.sock`; obce sockety, FIFO oraz aliasy katalogów są odrzucane.
- Losowanie z gęstego zakresu identyfikatorów usuwa tworzenie listy całego
  replaya. Zachowuje dokładnie ten sam porządek próbek i stan RNG.
- Oczekiwany kontrakt/clock są serializowane raz na batch zamiast dla każdego
  wiersza. Każdy wiersz nadal przechodzi wszystkie kontrole, w tym hash
  obserwacji, czas, nagrodę i terminal. Końcowy świeży digest odrzuca zmianę
  specyfikacji podczas składania batcha. Nie ma globalnego cache kontraktu.
- Wewnętrzna ścieżka targetów korzysta z już zweryfikowanego batcha; publiczna
  funkcja targetów zachowuje własną walidację. Arytmetyka, dyskonto, bootstrap,
  precyzja, częstotliwość aktualizacji i synchronizacja polityki pozostają takie
  same.
- Licznik kredytu dokładnie sumuje dziesiętną reprezentację czasu zapisanego
  w zaakceptowanych przejściach. Usuwa kumulowanie błędu float, które w starym
  N5 opóźniało aktualizację przy przejściu12488. Nie zmienia czasu wierszy,
  nagród ani stawki aktualizacji. Checkpoint zachowuje także dokładny licznik
  jako licznik/mianownik. Import starego zapisu zachowuje jego zapisaną
  przybliżoną sumę; nie odtwarza domniemanej historii z liczby przejść.
- Odczyt potwierdzenia zamknięcia ma limit1MiB, wymaga zwykłego pliku i nie
  blokuje się na FIFO. Uszkodzone potwierdzenie nie przyznaje poprawnego cleanup.

Limit rozmiaru jest kontrolą aplikacyjną, a nie kwotą jądra. Odczyt stanu plików
nie jest atomowy względem niezależnego procesu silnika. Licznik zdarzeń zachowuje
świeże sprawdzenie przy każdym wywołaniu i fail-closed po błędzie; przepełnienie
kolejki/zmiana drzewa wymusza pełne uzgodnienie. Liczba zdarzeń jednego drainu
oraz liczba przebudów jednego wywołania są ograniczone.

Deadline ogranicza aktywną pracę sesji. Bezpieczne zamknięcie procesów i zapis
raportu mogą potrwać po jego upływie; podany pełny czas obejmuje te czynności.

## Sprawdzenie zachowania

Porównanie obejmuje każdą zaakceptowaną akcję i nagrodę, rzeczywiste native76
we wszystkich interwałach, timestampy, liczbę interwałów, flagi, hash obserwacji
oraz końcowy model. PID-y, adresy procesów i ścieżki nowych sesji są różne;
nie są cechami modelu ani miarą zgodności fizyki.

Testy uczenia porównują także gradienty, online/target, Adam, RNG, sampler,
terminalne i ucięte przejścia oraz zachowanie replaya po zawinięciu. Testy
storage/lifecycle obejmują STOP, reset/krok z błędem, zapas przed rollover,
przerwanie aktora, utratę zdarzeń, błędne watch, częściowy zapis i zamknięcie
deskryptorów. Nie dopisuje się sztucznej mety ani truncation po awarii/STOP.

Końcowa wspólna regresja: **941 testów PASS w80,93s**, obejmująca core,
distributed, Trackmanię i integrację environment runtime. Ruff check/format
przeszedł dla20 zmienionych plików Python, strict mypy dla10 modułów źródłowych.
W tym zestawie są testy dokładnego budżetu do miliona przejść dla N1, N5
i mieszanego czasu, wznowienia oraz regresja przy przejściu12488.

Rzeczywista dłuższa próba `long5` wykonała16384 przejścia,81920 interwałów
i4096 aktualizacji w187,823s całego procesu, czyli87,231 przejścia/s. Zachowała
10000 wierszy replaya po zawinięciu;41 epizodów zakończyło się jawnym limitem
czasu, bez deklarowanej mety polityki. Czternaście sesji zamknięto poprawnie.
Dokładny czas zaakceptowanych przejść wynosi819,2, całkowity i ułamkowy kredyt
pozostały równy0. To osobna próba poprawności z replay10000, nie mianownik
porównania tabeli replay1000. Użyła jawnego limitu1GiB, bez zmiany domyślnych512MiB.
`time -v` podał maxRSS422672KiB i swap0; maxRSS nie jest sumą szczytów drzewa
procesów. Pełne przedziały wall każdej sesji nie są osobno zapisywane; sprawdzono
konfigurację deadline240s, zapisane czasy epizodów i zamknięcie.

Końcowe próby silnika obejmują:

- Awarię własnego aktora w drugim ponownie użytym epizodzie: trening odrzucony,
  zamknięcie potwierdzone, zachowany poprawny checkpoint400 przejść/100 aktualizacji.
  Wznowienie do800/200 dało dokładnie model przebiegu kontrolnego.
- STOP po zaobserwowaniu450 wierszy: zaakceptowano452, zachowano113 aktualizacji
  i ostatni nieterminalny stan. Wznowienie o128 przejść dało580/145.
- N3 z limitami1000 jednostek:257 przejść,757 faktycznych interwałów,
  37 aktualizacji oraz0,85 kredytu; końcowe skrócone kroki są rzeczywiste.
- Limit3MiB dowodów: odmowa po158 krokach, zero kredytu ucznia i wszystkie
  własne procesy zamknięte. Ograniczona publikacja cleanup pozostaje jawnie
  niepełna; ten test nie jest udanym treningiem.
- Przejazd źródłowy przez publiczny interfejs:1780 kroków N1, fizyczna meta
  17715ms,1 checkpoint,0 respawnów. `validationAcceptance=false` zachowano
  zgodnie z wynikiem walidatora. To test mety z referencyjnym sterowaniem,
  bez uczenia i bez deklaracji jakości wytrenowanej polityki.

`evidence_bytes` jest świeżym snapshotem przed publikacją raportu, obejmującym
także istniejące wtedy prywatne kopie `/tmp`. W `long5` wynosił492111953.
Późniejszy odczyt trwałych wyników dał405240518; brakujące prywatne duplikaty
86916800 dokładnie uzgadniają różnicę po doliczeniu45365 bajtów raportu.
Kopie opublikowanych POST/RX i journale pozostały zweryfikowane. Przyczyna ani
moment późniejszego zniknięcia prywatnych ścieżek nie są udowodnione.

Mikrotest procesora porównał1 i2 wątki na tych samych150 rzeczywistych batchach
w trzech powtórzeniach. Jeden wątek był wolniejszy i zmieniał gradienty, wagi
oraz Adam. Zostały2 wątki; odrzucony wariant nie jest częścią wdrożenia.

Pierwsza próba R8 została jawnie zachowana jako nieudana: przed pierwszym
przejściem odrzuciła należące do sterownika `branch.sock`. Dodano precyzyjną
obsługę tego gniazda i regresje bez zezwalania na obce pliki specjalne.
Jej procesy zostały zamknięte; nie zalicza się jej jako udanego treningu.

## Uruchomienie i zakres

Instrukcja oraz wznowienie: [headless-native-training.md](headless-native-training.md).
Lokalny launcher:
`F:/AITrackmania/reports/headless-performance-20261022-r8/Uruchom-Szybki-Trening-Headless.ps1`.
Surowe wyniki i checkpointy:
`<USER_HOME>/tmrl-headless-performance-20261022-008/`.
Małe raporty tekstowe:
`F:/AITrackmania/reports/headless-performance-20261022-r8/`.

To optymalizacja poprawnego, ograniczonego treningu native-only. Nie jest
porównaniem z Windows ani pomiarem czasu do uzyskania tej samej jakości modelu
w kampanii. Krótki test dokładności wyniku nie kwalifikuje wielodniowego
obciążenia. Nadal obowiązują common6/QR-DQN i osobne checkpointy tego profilu;
zamrożona kampania GNN/SD-SAC nie została zmieniona ani wznowiona.
