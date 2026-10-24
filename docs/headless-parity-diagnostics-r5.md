> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Diagnostyka i poprawki r5 — 21.10.2026

Status: ograniczone eksperymenty headless. Niezależny trening porównawczy i
kampania nadal wymagają kolejki/ACK komend Windows oraz kwalifikowanych granic
decyzji. Przy odczytanych wejściach konsumenta obie próby fizyki są bitowo zgodne.
Zamrożone instalacje, kampania i oczekiwane hashe pozostają bez zmian.

## Taśma i rzeczywista meta

Standardowy publiczny adapter `NativeImmediateEnvironment` również zakończył
mapę: neutralny reset2200,1780 rzeczywistych kroków action78/N1 i naturalny EOF
readera. Produkcyjny kontrakt zwraca `terminated=true`, `truncated=false`,
czas17715/CP1/respawn0 i zachowane `validationAcceptance=false`. Nie używa op2,
nie zmienia pinów/resetu i nie wykonuje uczenia. Jest to oddzielny mocniejszy
dowód działania standardowej ścieżki terminalnej; szczegóły i komendy w
[headless-finish-r5.md](headless-finish-r5.md).

Porównanie z dostarczonym eksportem Jimmy17715 wykazało dokładnie pięć różnic
SGB w 1922 znormalizowanych rekordach. Indeksy151..155, odpowiadające etykietom
race0..40, mają w taśmie diagnostycznej steer−0.503937/gas1; eksport ma steer0/gas1.
Pozostałe1917 rekordów zgadza się bajtowo. Nie zmieniano pliku bazowej taśmy.

`scripts/headless_finish_drain.py` odtwarza wyłącznie te pięć dosłownych SGB
przez V7 op6, zachowuje pięć audytów i po ACK2240 wydaje jawny release/op2.
Autonomiczna reszta ma zero kredytu learnera i nie staje się przejściem polityki.
Pełny, zamknięty JSON oryginalnego walidatora oraz ostatni TMRESULT potwierdziły
czas17715, jeden checkpoint, zero respawnów i wynik0 przy native20000.
Zachowano1798 kolejnych POST2030..20000. Normalny reap i cleanup: pozostałe0.

Walidator zachowuje `IsValid=false` z deklaracją40000ms i komunikatem o czasie
walltime19s. To rzeczywisty obliczony finish, a nie rankingowa akceptacja.
Istniejący produkcyjny `ImmediateSession.classify_terminal` zwrócił na tych
oryginalnych danych `terminated=true`, `truncated=false`, wynik17715/CP1,
`validationAcceptance=false`. Endpoint jest ostatnim rzeczywistym autonomicznym
POST, odrębnym od releaseACK; `physicalFinishNativeT` pozostaje null.

Próba bez odtworzenia pięciu wejść zachowana jako `drain-original001`:4298 POST
2030..45000, pełny JSON i43 TMRESULT, lecz `ValidatedResult=null`. Nie nazwano
jej metą ani DNF. Obie próby omijają znany reject337 przez jawny release przed
ARM, a nie przez ukryte skrócenie decyzji. Produkcyjny guard końca taśmy zostaje.

## Dłuższy pomiar Windows

`fresh006-clock001` wykonał320 decyzji, bez uczenia. Dwie taśmy miały po8000ms
rzeczywistego zakresu: neutralna oraz2sneutral/1sgas/5scoast. Cały program z
resetami i cleanup trwał24.687s; kontroler zamknięty, profil004 przywrócony.
Braki to odpowiednio7 i2 ticki, bez niejednoznacznego SGB lub physics6.
Pełny test ciągłego przebiegu wejść pozostaje niezaliczony.

Ścisłe odtwarzanie kończy się przed pierwszą luką:114/114 stanów do race1160
oraz397/397 do race3990 zgadza się bajtowo. Nie interpolowano braków ani nie
łączono prób. Osobny eksperyment neutralny zachował806 rzeczywistych native POST
2190..10240. Wszystkie794 zachowane stany Windows mają jednoznaczne dopasowanie:
397 fit,397 heldout, `race_ms=nativeT−2190`, zakres obserwowanych kotwic8000ms.
Raport jawnie nie kwalifikuje ciągłego interwału, waveform, ACK lub treningu.
Szczegóły i dokładne komendy: [headless-clock-r5.md](headless-clock-r5.md).

## Hipoteza jednego ULP

Wartości pochodzą z rzeczywistej niejednoznacznej próbki r4 przy race1530.
Trzy jawnie kontrfaktyczne eksperymenty mają po200 interwałów i nie zmieniają
produkcji. Pojedyncza podmiana w1530 nie zmienia żadnego z200 physics6 i nadal
daje161/200 zgodności ze starym Windows. Podmiana podczas całego skrętu daje
102/200 zgodności, a od1530 daje159/200. Nie odtwarza to brakującego rzeczywistego
przebiegu Windows ani nie wyjaśnia automatycznie różnicy pionowej. Każdy cleanup0.

## Rzeczywisty konsument wejść Windows

Nowy jawnie ładowany `HeadlessInputConsumer007` został rzeczywiście skompilowany
w Openplanet i użyty w dwóch ograniczonych próbach. Odczytuje pola wejść w
funkcji fizyki po oryginalnym zapisie filtered steer; nie zmienia danych gry.
Weryfikuje dokładne64 bajty kodu, pełną8-bajtową relokowaną instrukcję, tożsamość
owner→vehicle i niezależne przywrócenie oryginalnego prefiksu. Szczegóły RVA,
ABI i źródeł w [README007](../trackmaniarl/project/openplanet/HeadlessInputConsumer007/README.md).
Analiza cache/API i dosłownych bitów:
[headless-input-consumer-r5.md](headless-input-consumer-r5.md).

Próby live002/live003 zachowały199 i228 rekordów. Kontroler zamknięty, profil004
przywrócony i64 oryginalne bajty przywrócone w obu. Pierwszy rawphysics6 każdej
próby ma dokładnie jeden content-match do rzeczywistej neutralnej taśmy native,
odpowiednio2250/2260. Nie dobierano kotwicy przez przesunięcie zegara API.

W dłuższym przebiegu rzeczywiste bieżące SGB rekordu[i] użyte do następnego
rawpose[i+1] daje227/227 bitowo zgodnych physics6, w tym Y/VY. Jawna hipoteza
poprzedniego SGB daje13/226, a same-row daje14/228. Wszystkie warianty zachowano;
nie wybrano nowego offsetu ani tolerancji, żeby ukryć odchylenia. Jest to dowód
zgodności tej rzeczywistej trajektorii przy właściwej fazie wejść. Nie stanowi
rekonstrukcji starszej niejednoznacznej taśmy ani globalnego dowodu każdej mapy.
Druga próba daje198/198 w fazieA,49/197 wB i50/199 wC. Łącznie fazaA ma425/425
zgodnych interwałów, ze zgodnością wszystkich10 float32 quaternion/pos/vel,
nie tylko physics6. Wszystkie6 cleanup są poprawne. Szczegółowa komenda i
wszystkie warianty: [input_consumer_native_replay.md](input_consumer_native_replay.md).

Pola vehicle-observed i participant-cache potrafią zachować poprzednie wejście
przy przejściu, podczas gdy raw/filtered fizyki już zawierają nowy skręt.
W dłuższym śladzie21 pakietów API ma steer0xbe2952a6; odpowiadają21 jednoznacznie
dopasowanym fizycznym stanom konsumenta z actual/filtered/vehicle/cache0xbe2952a5.
Różnica jednego ULP nie jest wejściem, które wolno automatycznie wstrzyknąć do
native jako dokładną rekonstrukcję. Dane nie dowodzą atomowego odczytu między
wątkami, a w krótszej próbie takiej różnicy nie ma.
Dlatego odczyt API nadal nie jest ACK wydanej komendy. R13D pozostaje dosłownym
argumentem etapu, a nie automatycznie uznanym zegarem wyścigu. Nie wdrożono
stałego hooka jako produkcyjnego kontrolera ani nie uruchomiono porównania uczenia.

Pełne czasy poleceń Windows, z importami/resetami/zapisem/cleanup, są w osobnych
`*-command.json`. Własne moduły004/006/007 rozładowano, źródła zarchiwizowano,
porty19177/19178/19179/19180 nie nasłuchują i wszystkie4 lease UI są nieaktywne.
`live001` poprawnie odmówił konkurującym procesom testowym przed dotknięciem gry;
powtórzono dopiero po ich naturalnym zakończeniu.

## Odtwarzalność i wyniki

Źródła są w oddzielnym worktree i gałęzi `codex/headless-training-parity`.
Raporty Windows: `F:/AITrackmania/reports/headless-training-parity-20261021-r5/`.
Nowe surowe wyniki native: `<USER_HOME>/tmrl-headless-parity-20261021-005/`.
Binarki, snapshoty i prywatne zasoby pozostają poza repozytorium.

Agregacja powtórzeń nie uznaje już pustego zestawu przypadków lub zerowej liczby
interwałów za zgodność. Regresje obejmują obie ścieżki CLI. Raporty diagnostyczne
zapisują też błąd cleanup niezależnie od normalnego zakończenia.

Końcowy zestaw core/distributed/trackmania oraz integracji środowiska:
708 testów PASS39.80s; pełne polecenie43.46s, maxRSS419144KiB i zero swap.
Poprzedni szeroki zestaw667 testów również PASS36.70s. Ruff check/format dla19
zmienionych plików Python przeszedł; wąskie regresje obejmują też błąd close,
zastrzeżone nazwy raportów i oba profile długości próby007.
Mypy12 modułów źródłowych PASS z lokalnymi strict checks; importy zależności
sprawdzono w trybie silent, pominięto brak zewnętrznych stubów psutil
(`import-untyped`) w środowisku hosta używanym wyłącznie do odczytu. Po zmianach
samych adnotacji w runnerach mety ponownie36 regresji PASS4.46s.
36 zapisanych tożsamości procesów w18 rzeczywistych supervisor receipts:
pozostałe0. Normalne reap własnych silników potwierdzają raporty poszczególnych
prób. Kontrola nie sygnalizuje ani nie zatrzymuje żadnego procesu.
Pierwsze polecenie użyło domyślnego pytest basetemp pod/mnt i15 testów poprawnie
odrzuciło taką konfigurację. Poprawione polecenie ustawia jawne basetemp i cache
na ext4; nie zmieniono wymagań runtime ani testów, by ukryć odmowę.
Ponowna kontrola466 zamrożonych plików źródłowych i właściwego runtime/assets.json:
zgodne. Wcześniejsze porównanie omyłkowo użyło innego manifestu zasobów; zachowano
wynik i poprawny raport `integrity-r5-002.json` z pełnymi ścieżkami.

Wyniki diagnostyczne nie są benchmarkiem czasu uczenia. R5 obejmuje nowe kopie
prywatnych runtime, dane testów i surowe raporty na ext4; sama szybkość ich
odtwarzania nie przewiduje czasu osiągnięcia jakości kampanii. Wspólny model6
pozostaje profilem MLP/QR-DQN od nowych wag; dotychczasowe checkpointy GNN i
nieobecne pola obserwacji nie stają się z nim zgodne. Nie uruchamiano nowego
treningu ani kampanii. Budżet7168000 przejść i chronione zasoby pozostają bez zmian.
