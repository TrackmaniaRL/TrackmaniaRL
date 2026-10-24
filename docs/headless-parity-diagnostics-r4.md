> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Świeżość resetu i rzeczywiste próby Windows — r4

Status: **ograniczone eksperymenty V7/common6; pełne porównanie treningów i kampania
nadal niezakwalifikowane**. Nie uruchamiano uczenia porównawczego ani kampanii.
Nie zmieniono produkcyjnego shima V7, jego hashy, progów zgodności ani nagrody.

## Naprawiony błąd resetu Windows

`OpenPlanetEnvironment.reset()` zwracał stan pobrany przez `_wait_for_active_run`,
po czym blokujące `confirm_ready` opóźniało powrót do polityki. W trzech pierwszych
decyzjach pomiarowych zachowany nagłówek świeżej telemetrii wykazał wiek stanu
20.1557–27.2277 ms przed wysłaniem komendy. Wywołanie kontrolera trwało około
0.16–0.26 ms. To rozdziela oczekiwanie na gotowość od samego zastosowania komendy.

Reset teraz pobiera ostatni stan **po** potwierdzeniu gotowości i opcjonalnym
settle. To ten stan zasila reward i początkowy zegar epizodu. Cofnięty zegar albo
zakończony wyścig powodują błąd; nie następuje powrót do wcześniejszej próbki.
Info jawnie zawiera czasy próbki potwierdzającej reset i zwróconej obserwacji.
Brak dodatkowego sleep, stałego przesunięcia zegara lub dopisywania doświadczeń.

Próba po poprawce: wiek stanu przed gazem/skrętem/tapem odpowiednio
**0.3035 / 0.2921 / 0.2746 ms**. Rzeczywiste początkowe zegary wynosiły20/30/50 ms,
a nie wymuszone10 ms. Zmienność musi pozostać widoczna przy zestawianiu backendów.
Ta poprawka nie stanowi potwierdzenia ticka zużywającego wejście.

`fresh006-timing-fixed002`: neutralny gęsty odczyt i pięć taśm zakończyły się
poprawnie,170 decyzji bez uczenia, kontroler zamknięty, profil wejścia przywrócony.
Pełne polecenie z importami, resetami i cleanup: **26.0572632 s**. To pomiar
diagnostyki, nie benchmark szybkości uczenia lub prognoza czasu kampanii.

## 005 i 006 działają w przypiętym kliencie

Obie wtyczki zostały rzeczywiście skompilowane i załadowane w Openplanet
1.29.14/git6fd4200f, gra `2026-02-02_17_51`, SHA256 pliku wykonywalnego
`3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`.
Zachowano filtrowany log własnych modułów. Openplanet zgłosił także nieblokujący
komunikat o już załadowanej pośredniej zależności MLHook dla006; moduł uruchomił
się i obsługiwał rzeczywiste żądania. Nie zmieniano istniejącego Connect/MLHook.

`physical005-live001`:195 wywołań,195 zaakceptowanych rekordów, zero obcych
właścicieli; jawny stop, potwierdzone odtworzenie całych64 oryginalnych bajtów,
zamknięcie kontrolera i przywrócenie profilu. Wnętrze polecenia6.844 s obejmuje
reset, dwie około sekundowe fazy jazdy i cleanup. Nie było modelu ani aktualizacji.

193/195 stanów raw position/velocity ma dokładny odpowiednik w oddzielnym
Connect33 przy jednym unikalnym czasie wyścigu. Dwóch brakujących obserwacji
nie uzupełniono. To złączenie po zawartości, nie atomowy POST odczyt w tym samym
wywołaniu. PRE-copy destination zawierał wartości domyślne we wszystkich195
rekordach; nie należy utożsamiać go z poprzednią klatką API. Nadal nie wykazano
atomowości między wątkami ani zegara fizyki callbacka.

006 zwraca rzeczywiste132 B powiązane z nonce i sekwencją żądania, wraz z czasami
producenta i hosta. Dwa pełne zestawy prób Windows przeszły. Samo usunięcie kolejki
nie usuwa opóźnienia wejść: po poprawce resetu pierwszy odczyt gazu w czterech
taśmach występuje10–30 ms czasu wyścigu po zwróconym stanie. To odczyt API,
a nie potwierdzenie zastosowania wejścia w fizyce.

## Próbki wejść nie zawsze określają jeden tick

W świeżej taśmie mild_left przy race1530 odczytano dwa SGB:
`[-0.16535434126853943,1,0]` oraz `[-0.16535432636737823,1,0]`. Skręt różni się
o jeden ULP. Nie zaokrąglono tych wartości do wspólnej komendy. Odtwarzanie
ścisłej taśmy słusznie odrzuciło ten przypadek. Jest to konkretny dowód, że zegar
race ms i odczytane wejście nie tworzą jeszcze gwarantowanej pary na jeden tick.

Wcześniej zakończone prefiksy tego odtwarzania: neutral51/51 i gaz173/173
posXYZ/velXYZ bitowo zgodne. Gaz przerwano na pierwszej brakującej próbce; wynik
nie obejmuje całej taśmy. Cały raport porównawczy pozostaje niezaliczony.

Ujawniono też błąd narzędzia odtwarzania: gdy pierwszy brakujący tick zostawiał
tylko stan początkowy, kod indeksował pustą listę przejść. Teraz raportuje jeden
faktyczny stan i brak przejść polityki, zachowując `exact_physics_parity=false`.
Testy sprawdzają brak komendy polityki i odrzucanie niejednoznacznych wejść.

## PSS i eksperyment x87

Nowy odczyt metadanych ujawnił `PSS_THREAD_FLAGS_TERMINATED=1` i brak kontekstu
zakończonego wątku. Flaga jest zdefiniowana w zachowanym nagłówku Microsoftu.
Czytnik zachowuje takie wpisy osobno, bez odczytu wskaźnika i bez wymyślania FP.
Każdy aktywny wątek nadal musi mieć pełny, wyrównany kontekst z żądanymi flagami;
nieznane flagi, brak aktywnych kontekstów lub niepełna enumeracja są błędami.

Rzeczywisty capture:139 wpisów,135 aktywnych kontekstów i4 zakończone wątki.
Wszystkie135 aktywnych kontekstów mają x87 CW `0x027f` (53 bity), MXCSR control
nearest/FTZ-off/DAZ-off, wszystkie wyjątki maskowane. Nie zidentyfikowano wątku
fizyki ani jego etapu. Oba kody zwolnienia zasobów0, czas całego workera0.406 s.

Oddzielny **V9** sprawdził tę hipotezę na Linux/ext4: w zapisanym kontekście
sygnału zmienia wyłącznie bity precyzji x87 na53, odczytuje je z powrotem i
potwierdza niezmienność pozostałych30 B FXSAVE. Ma nowy wire TMA9/TMR9,
osobny build/source/patch/receipt; V7 i V8 pozostały bez podmiany. Nie zapisuje
pozycji, prędkości, zegara ani innego stanu fizycznego. Jest wyłącznie wariantem
eksperymentalnym, niedopuszczonym do produkcyjnego provisioningu.

W200 interwałach zapisane x87 CW wynosiło `0x027f`, lecz wszystkie200 stanów
posXYZ/velXYZ pozostało identycznych z V7. Zgodność z Windows nadal161/200
dynamicznych próbek, czyli162/201 z poprzednim stanem początkowym. Sama ta zmiana
precyzji nie wyjaśnia obserwowanej różnicy pionowej. Nie wymuszano innych
ustawień FP i nie zmieniono oczekiwań porównania.

## Testy, pliki i zamknięcie

- Core/distributed/trackmania oraz integracja environment/telemetry:645 PASS,
  36.60 s pytest,39.96 s pełne polecenie, max RSS418928 KiB, zero swap.
- Po uporządkowaniu importów/typów ponownie17 krytycznych testów PASS; mypy
  dla8 zmienionych plików źródłowych PASS, Ruff dla zmienionych plików PASS.
- Rzeczywiste próby profilu004: EOF klienta, TTL, nieznana operacja i utracone
  ACK zwolnienia. Wszystkie4 potwierdziły przywrócenie baseline bez sterowania.
- Odrzucenia fokusu zachowano. Kontynuowano dopiero po aktywowaniu gry przez
  Jakuba; nie obchodzono zabezpieczenia ani nie zatrzymywano cudzych aktorów.
- 30 zapisanych tożsamości procesów Linux: pozostałe0. Własne lease UI zwolnione.
  Wszystkie cztery własne moduły003–006 rozładowano, instalowane źródła
  zarchiwizowano; porty19177/19178/19179 nie nasłuchują. Kontrolery zamknięte,
  profil004 i kod005 przywrócone. Źródła nadal są w repozytorium jako opt-in.

Raporty Windows i podsumowanie:
`F:/AITrackmania/reports/headless-training-parity-20261021-r4/`.
Nowe sesje native, build V9 i ślady na ext4:
`<USER_HOME>/tmrl-headless-parity-20261021-004/`.
Nieudane próby zachowano, nie nadpisywano ich wynikami kolejnych uruchomień.

Pozostają do kwalifikacji: konsument wejść Windows z potwierdzeniem ticka,
rozstrzygnięcie różnicy Y przy faktycznie równoważnych wejściach, poprawna meta
i osobny kontrakt końca taśmy, dłuższy dowód zegara. Dopiero po nich wolno
uruchomić niezależny trening porównawczy na identycznym modelu i budżecie.
