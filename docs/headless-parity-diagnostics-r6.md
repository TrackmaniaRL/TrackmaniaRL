> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R6: przygotowanie protokołu i analiza offline, 22.10.2026

Status pozostaje taki jak po R5: ograniczone eksperymenty headless są możliwe,
ale niezależny trening porównawczy z Windows nie jest zakwalifikowany.
R6 nie uruchamiał gry, hooków, natywnego silnika ani treningu i nie zmienia
protokołu V7, algorytmu, zamrożonej kampanii ani jej budżetu.

R5 w commicie `056e068349355ed337991bac5c9f743be63a1051` potwierdził metę
publicznym adapterem i 425/425 dokładnych przejść z rzeczywiście zmierzonych
wejść konsumenta Windows. Nadal trzeba powiązać **wydaną komendę** z konsumpcją
i następnym stanem. Powtórzenie zaobserwowanych wejść nie dostarcza tego dowodu.
Raport R5 i zamknięcie jego własnych modułów pozostają osobnymi dowodami;
gra nie jest potrzebna do poniższych prac offline.

## Gotowe źródła

- [Kontrakt command ACK](command-ack-r6-contract.md) oraz
  [czysty walidator](../scripts/command_ack_r6_protocol.py) rozdzielają komendy,
  publikację kolejki, odczyt konsumenta i rzeczywiste kolejne granice stanu.
  Walidator odrzuca luki, duplikaty, złą tożsamość resetu/pojazdu, spóźnione
  publikacje, rozbieżne bity SGB i brak końcowego stanu. Testy są syntetyczne;
  wynik ma stałe `physical_execution_qualified=false` oraz
  `paired_training_qualified=false`. Moduł nie ma transportu ani ścieżki do
  produkcyjnej bramki kwalifikacji.
- [Analiza PRE i warunków pojazdu](input_command008_preconsumer_static.md) oraz
  [reproduktor](../scripts/input_command008_static.py) sprawdzają przypięty plik
  wykonywalny i ograniczone wycinki instrukcji. Odrzucono dawny kandydat
  `0x7eb2c0`: klasa `0x0a044000`, rozmiar `0xc8`, to efekt headtrack. Rzeczywisty
  pojazd ma klasę `0x0a020000`. Zbieżne numery pól nie uzasadniały writera.
- [Analiza współbieżności](headless-command-ack-r6-threading.md) wskazuje wspólny
  mutex hosta dla badanych ścieżek Main i callbacka. Nie daje to dowodu
  bezpiecznego usunięcia natywnego hooka: wywołanie może wejść do dispatchera
  przed oczekiwaniem na ten mutex. Odczyt przywróconych bajtów nie zastępuje
  dowodu zakończenia wszystkich takich wywołań.
- [Inwentaryzacja pauzy](r6-pause-api-inventory.md) obejmuje 2585 klas i 16321
  deklaracji członków. Znalezione zegary, odtwarzanie i współczynniki wizualizacji
  nie deklarują zatrzymania świata z wykonaniem jednego interwału fizyki.
  Nie zmieniano tych pól ani ustawień.

## Pozostałe warunki wykonawcy

1. Wskazać rzeczywisty punkt zapisu przed wszystkimi istotnymi konsumentami
   całego kroku. `0x841553` poprzedza odczyty w funkcji `0x841500`, ale jej rodzic
   odczytuje wejścia wcześniej. `0x851f96` jest wcześniejszym kandydatem, przed
   clampem; nie jest jeszcze zakwalifikowanym writerem. Trzeba zachować odczyty
   pięciu pól warunków: `176c`, `1370`, `1600`, uint16 `1768`, float bits `1374`.
2. Dowieść bezpiecznego czasu życia natywnych callbacków, także podczas błędu
   i zdejmowania hooków. Sama serializacja AngelScript tego nie rozstrzyga.
3. Zmierzyć każdy związek command ID → literalne SGB konsumenta → następny
   physical10, bez luk i dopasowywania czasu w celu usunięcia błędów. Odczyt
   zwrotny po zapisie oraz komunikat `queued` nie są physics ACK.
4. Dla aktora w Pythonie jawnie rozliczyć czas odbioru i obliczenia kolejnej
   decyzji. Proponowany profil stałego neutralnego prefiksu D dodaje te same
   D+N interwałów na obu backendach i wymaga odrębnego kontraktu. Nie można
   podmienić nim czasu i checkpointów istniejącego V7. Natychmiastowe D=0
   wymaga dodatkowego dowodu pauzy fizyki albo wykonania aktora w procesie.

Te warunki są brakującymi dowodami technicznymi. Nie wdrożono nowego modułu
zapisującego wejścia. Przygotowanie syntetycznego protokołu nie zamyka ich
i nie odblokowuje kampanii.

## Lokalne raporty

Kanoniczny katalog R6:
`F:/AITrackmania/reports/headless-training-parity-20261022-r6/`.
Prywatne pliki wykonywalne i wycinki pozostają poza repozytorium, na ext4:
`<USER_HOME>/tmrl-headless-parity-20261022-006/`.
Nie opublikowano binarek, zasobów gry, snapshotów ani prywatnych śladów.

`summary-final.json` wiąże końcowe źródła i lokalny commit z raportami kontroli
offline. Wynik 708 testów całego poprzedniego etapu jest dowodem R5; R6 dodaje
odrębne testy nowych modułów. Nie przedstawiamy testów syntetycznych jako
nowego pomiaru fizyki.

Końcowa kontrola R6: **34/34 testy PASS w 3,02 s** (29 protokołu, 5 analizy
statycznej), Ruff check/format dla czterech plików Python oraz strict mypy
dla dwóch nowych modułów. Niezależny przegląd protokołu nie znalazł konkretnych
luk w sprawdzaniu kolejności, duplikatów, czasu publikacji, tożsamości komend
i zachowania flag kwalifikacji. XML testów jest w `pytest-root001.xml`.
