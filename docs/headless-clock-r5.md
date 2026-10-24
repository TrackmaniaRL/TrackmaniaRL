# Kontrolowany zegar i świeże obserwacje — r5

Własność tej części: `scripts/headless_clock_r5.py` oraz jego testy. Narzędzie
wykonuje wyłącznie analizę plików. Nie uruchamia gry, native runtime, UI ani
instalacji wtyczek; nie zmienia resetu ani odtwarzacza taśm. Kontroler, fokus,
ładowanie004/006 i rzeczywiste capture pozostają po stronie prowadzącego próbę.

## Plan i warunki

Taśmy przekazane do istniejącego `headless_windows_timing.py`:

- `neutral_8s`:160 decyzji36, nominalnie8s bez gazu, skrętu i hamulca.
- `neutral_gas_coast_8s`:40 decyzji36,20 decyzji39,100 decyzji36;
  nominalnie2s neutralnie,1s gazu na wprost i5s wybiegu.

Identyfikatory są z istniejącego porządku action78:36 oznacza SGB=(0,0,0),
39 oznacza SGB=(0,1,0). Nominalne50ms decyzji nie zastępuje zmierzonego czasu.
Analizator wymaga rzeczywistego zakresu8000–10000ms, siatki10ms i obserwowanego
rozpięcia gazu nie większego niż1500ms. To ostatnie jest kontrolą zapisu API,
nie potwierdzeniem rzeczywistego czasu zużywania gazu przez fizykę. Rzeczywisty
runner zachowuje istniejące STOP, limity, mutex kontrolera i cleanup.

Wymagane przed uruchomieniem przez operatora: wolny kontroler, brak innego
aktora, właściwa przypięta mapa/reset, potwierdzony fokus i jawnie załadowane
świeże004/006. Ustawienia profilu004 muszą zostać odczytane i przywrócone.
Nie łączy się oddzielnych podejść w jedną pozornie kompletną taśmę.

Przykładowe wywołanie capture z katalogu worktree, gdy operator spełni warunki:

```powershell
python -m scripts.headless_windows_timing F:/AITrackmania/reports/headless-training-parity-20261021-r5/fresh006-clock001 F:/AITrackmania/reports/headless-training-parity-20261021-r5/clock-tapes001.json --no-artificial-lag --fresh-telemetry-port 19179
```

Ta komenda steruje grą. Nie została uruchomiona przez autora analizatora;
rzeczywistą próbę wykonał prowadzący. Każde kolejne podejście wymaga nowego
katalogu i nowego pliku wynikowego, bez nadpisywania niezaliczonych śladów.

## Ścisły audyt bez uzupełniania

```powershell
python -m scripts.headless_clock_r5 inspect-windows F:/AITrackmania/reports/headless-training-parity-20261021-r5/fresh006-clock001/windows-timing.json F:/AITrackmania/reports/headless-training-parity-20261021-r5/clock-tapes001.json F:/AITrackmania/reports/headless-training-parity-20261021-r5/clock-audit001.json
```

Audyt sprawdza dokładne132B i SHA każdej odpowiedzi006, nagłówek/nonce/sekwencję,
kolejność czasów producenta i hosta, mapę, ciągłość obserwacji przy decyzjach oraz
powiązanie poleceń z zachowanym stanem. Dla każdego raceT zachowuje wszystkie
odczyty i odrzuca różne physics6 lub różne bajty SGB. Nie zaokrągla skrętu,
nie wybiera wygodniejszego duplikatu i nie interpoluje brakujących ticków.
Identyczne duplikaty są jawnie policzone, bez tworzenia nowych doświadczeń.

`capture_passed` nie oznacza `complete_tick_evidence`; żadne z nich samo nie
oznacza kwalifikacji zegara. Kod wyjścia2 oznacza brak pełnego dowodu, a raport
nadal zawiera braki. Przypadek może być użytecznym śladem diagnostycznym pomimo
niezaliczenia pełnej taśmy. Pusta lista przypadków nie daje sukcesu.

Rzeczywisty `clock-audit001.json`:

| Przypadek | Race ms | Zachowane różne ticki | Braki | Niejednoznaczne SGB |
| --- | --- | --- | --- | --- |
| neutral_8s |30–8030|794|1170,1390,1400,1410,1420,3420,8010|brak|
| neutral_gas_coast_8s |30–8030|799|4000,4010|brak|

Obie pełne taśmy są niezaliczone. Drugi zapis wykazał990ms rozpięcia odczytanego
gazu. Nie wolno odtworzyć brakujących ticków przez podtrzymanie poprzedniego
wejścia, nawet gdy otaczające odczyty są takie same.

## Osobne złączenie zachowanych obserwacji

Niepełny zapis Windows może nadal potwierdzić empiryczne przypisanie konkretnych
zachowanych stanów do rzeczywistych liczników native. Służy temu oddzielna
komenda `content-join`, która nigdy nie wystawia pełnej kwalifikacji zegara,
ciągłego interwału, przebiegu wejść, ACK ani treningu.

Native wejściowy ma dotychczasowy format `rows` z literalnymi `native_t` oraz
`postHex` każdego POST. Każdy licznik musi wystąpić raz, w rzeczywistej kolejności
z krokiem10; narzędzie odrzuca również braki native. Dane `state` mogą być
obecne, ale porównanie dekoduje physics6 z zachowanych bajtów POST. Wariant
raportu odtwarzacza jest także obsługiwany; przechowuje początkowy odczyt
physics6 i późniejsze literalne POST z dowodami interwałów.

```powershell
python -m scripts.headless_clock_r5 content-join F:/AITrackmania/reports/headless-training-parity-20261021-r5/clock-audit001.json \\wsl.localhost\Ubuntu\home\kuba\tmrl-headless-parity-20261021-005\neutral-clock001\report.json F:/AITrackmania/reports/headless-training-parity-20261021-r5/clock-observation-contentjoin001.json --case neutral_8s
```

Wykonany offline wynik:806 rzeczywistych POST native2190–10240; wszystkie794
zachowane Windows physics6 mają dokładne dopasowanie float32.397 kotwic służy
do dopasowania,397 pozostałych do weryfikacji. Ustalona dla tych obserwacji
relacja to `Windows race ms = nativeT - 2190`, a skrajne unikalne kotwice
obejmują8000ms. Żaden zachowany stan Windows nie pozostał niedopasowany.
Wszystkie siedem braków Windows pozostaje wymienione w raporcie.

Pole `observed_samples_affine_relation_verified=true` dotyczy tylko wymienionych
zachowanych obserwacji. Pola `clock_alignment_qualified`,
`continuous_interval_qualified`, `input_waveform_qualified`,
`input_consumption_acknowledgement_qualified`, `paired_training_qualified`
oraz `extrapolation_qualified` pozostają false. Nie rozszerza to automatycznie
bramki czasowej wspólnego treningu ani istniejącego krótkiego manifestu dowodów.
Nie jest to dowód wszystkich nieobserwowanych punktów między kotwicami.

SHA256 rzeczywistego raportu złączenia:
`86c1af3b409992843079e233716988f4b6149835cf26667a362bc332c6a15b76`.
Raport wiąże oba pliki wejściowe ich pełnymi SHA256. Nie nadpisano pierwszego
audytu ani surowych plików z brakami.

Stała, powtarzalna fizyka nie wyznacza jednoznacznej relacji zegarów. Złączenie
używa wyłącznie unikalnych dokładnych sygnatur positionXYZ+velocityXYZ do fit
i heldout. Powtórzenia są jawnie wykluczone z kotwic. W rzeczywistym neutralnym
śladzie r5 wszystkie794 zachowane sygnatury były unikalne; słowo „neutralny”
nie jest założeniem o nieruchomości pojazdu.

## Czy publiczne API daje tick input ACK

Odczytano przypięte lokalne deklaracje `Openplanet.h`, bez odczytu danych konta:

- `CSmScriptPlayer.InputSteer/InputGasPedal/InputIsBraking` to odczyty stanu.
- `CInputPort` udostępnia m.in. agregowane liczniki zdarzeń i pollingu.
- `CInputScriptManager` ma `Now`, `Period`, `PendingEvents`.
- `CInputScriptEvent` opisuje `PadButtonPress`, pad, przycisk i auto-repeat;
  w tym publicznym typie nie ma timestampu konsumenta ani numeru ticka fizyki.

W tych sprawdzonych deklaracjach nie znaleziono jawnego potwierdzenia, że dany
SGB został zużyty przez konkretny tick symulacji. Przypisanie `manager.Now` do
odczytu `InputSteer` nie daje takiego dowodu. To ograniczony wynik przeglądu
tych typów, nie dowód niemożliwości innego interfejsu. Oddzielna analiza PE
szuka rzeczywistego konsumenta; nadal nie wolno nazywać odczytu API ACK.

## Walidacja narzędzia

Osiem syntetycznych testów Linux przeszło: pełna siatka, brak ticka, różnica SGB
o1ULP, błędny hash, częściowe złączenie bez pełnej kwalifikacji, powtarzalne
nieruchome stany, brak native POST oraz pojedynczy niedopasowany zachowany stan.
Ruff przeszedł. Linux venv nie zawiera mypy, więc tej kontroli tutaj nie wykonano.
Testy nie uruchamiają gry ani native silnika. Rzeczywisty content-join był wyłącznie
analizą zachowanych plików, bez kolejnego capture.
