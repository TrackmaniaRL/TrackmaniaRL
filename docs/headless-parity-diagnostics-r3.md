> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Dalsza kwalifikacja Windows/headless — 21.10.2026, r3

Status nadal: **gotowy do ograniczonych eksperymentów V7/common6; porównanie
treningów z Windows i pełna kampania niezakwalifikowane**. Nie zmieniono progów
parytetu, nagrody, zamrożonych kampanii, produkcyjnego shima ani jego hashy.

## Co faktycznie uruchomiono

1. `HeadlessRuntimeProbe003` został skompilowany i uruchomiony w Openplanet
   1.29.14/git6fd4200f, gra `2026-02-02_17_51`. Świeży eksport potwierdza
   dostępność API i metadane 12 typów. Potwierdzono działanie niezależnego
   collectora003. Nie wykonuje hooków ani zapisów stanu gry.
2. `HeadlessInputProfile004` został skompilowany i użyty w ograniczonej próbie.
   Jego jawny lease zmienia wyłącznie `FakeInputLagAvg/Var`, z odczytem przed,
   po zastosowaniu i po przywróceniu. Zastano `68 / float32(0.15)`, podczas
   testu odczytano `0 / 0`; klient potwierdził przywrócenie obu wartości.
   `MaxSampleRate`, `PollingEnabled` i `IgnoreFocusForGamePads` zachowano.
3. W tym profilu wykonano neutralny gęsty pomiar i pięć taśm. Wszystkie sześć
   przypadków zakończono, kontroler zamknięto, przywrócenie profilu potwierdzono.
   Czas całego polecenia: **25.815 s**, wnętrza programu 21.797 s. Była to
   diagnostyka wejść bez uczenia.
4. Dwie nowe sesje native odtworzyły po 200 interwałów łagodnego skrętu.
   Wszystkie 200 stanów native76 w obu sesjach są bitowo identyczne ze sobą
   i z poprzednim przebiegiem V7. Cleanup obu sesji: zero własnych procesów.
5. Oddzielny eksperymentalny V8 odczytał zapisany kontekst FP przerwanego
   silnika Linux. Kolejne 200/200 native76 nadal odpowiada V7. Produkcyjny V7
   pozostał bez zmian. MXCSR `0x1fa5` i x87 CW `0x037f` nie zmieniały się przy
   pierwszej rozbieżności pionowej. Szczegóły: [dowody FP](headless_vertical_fp_evidence.md).

Dowody podsumowujące są w
`F:/AITrackmania/reports/headless-training-parity-20261021-r3/`.
Nowe surowe sesje, build eksperymentalny i analiza binarek pozostają na ext4:
`<USER_HOME>/tmrl-headless-parity-20261021-003/`.

Końcowy zestaw core/distributed/trackmania: **619 testów PASS**, 34.76 s pytest,
38.22 s całego polecenia, max RSS 414080 KiB, zero swap. Testy socketowe
005/006 używają niezależnych atrap producenta; nie potwierdzają kompilacji
wtyczek w grze. Powtórna kontrola 466 zamrożonych plików i manifestu zasobów
przeszła bez różnic. Dwa później dodane testy socketowe niepewnego arm ACK
i błędnego potwierdzenia rollbacku także przeszły (4.02 s). Osobna kontrola
17 zapisanych raportów i 14 tożsamości procesów r3: pozostałe własne procesy0.

## Wyłączenie FakeInputLag nie wyjaśniło opóźnienia

W nowym pomiarze pierwszy odczyt gazu nadal pojawił się 20–40 ms czasu
wyścigu po stanie użytym do wydania komendy. Początek taśmy gazu: race10→40,
skrętu: 10→50, hamowania: 10→50, brake tap: 20→40. Nie utożsamiamy tego
odczytu z potwierdzeniem ticka, który zużył wejście. Sama obecność ustawień
68/0.15 nie dowodzi, że sterują tą ścieżką gamepada.

Pakiety starego strumienia przychodziły przeważnie co około 6.6 ms, bez
regularnych paczek co 40 ms. W części resetów kilka pierwszych pakietów
odczytano jeden po drugim w mniej niż 1 ms. Może to wpływać na świeżość
stanu przed pierwszą komendą; nie jest dowodem, że całą zwłokę powoduje TCP.
Zegar `perf_counter_ns` jest metadanymi hosta, a nie zegarem fizyki.

## Zbadany koniec oryginalnej taśmy

`finish001` wykonał 3999 niezmienionych oryginalnych interwałów oraz jeden
jawny neutralny interwał resetu. Nie wystąpiła meta ani EOF. Sesja została
zamknięta poprawnie po osiągnięciu limitu diagnostycznego.

`finish002` dotarł do ostatniego oryginalnego rekordu. Po potwierdzonej
granicy native42200 silnik ominął kolejny reader i wykonał dodatkowy POST
native42210. Shim odrzucił tę ścieżkę jako `FKPOST reject337` i zakończył
proces kodem95. To **nie jest prawidłowa meta ani poprawnie zakończony epizod**.
Oryginalny validator nie zwrócił pełnego wyniku; zachowany TMRESULT nie
potwierdza ukończenia. Dodatkowy POST nie może zostać zaliczony do żądania.

Produkcja odrzuca ten niezgodny wynik. Dodatkowa kontrola przed op6 wymaga
istnienia następnego readera (`p+n < input_count`), aby nie wysyłać komendy,
której granicy obecny protokół nie obsługuje. To jawna granica protokołu,
bez udawania terminalnego przejścia lub cichego skracania żądania. Obsługa
autonomicznego końca walidacji wymaga osobnego kontraktu zwolnienia/drain.
Dowody nieudanych prób pozostawiono bez nadpisania.
[Szczegóły i próba na żywej sesji](headless_vertical_finish_evidence.md)
potwierdzają odrzucenie przed jakimkolwiek żądaniem, niezmienione wejścia/RX/POST
oraz poprawne zamknięcie procesu.

## Nowe narzędzia źródłowe czekające na próbę Windows

**Nie zostały skompilowane ani uruchomione w grze:**

- `HeadlessPhysicalProbe005` i `headless_windows_physical_probe.py`. Analiza
  przypiętego PE wykazała callback RVA `0x7d0cb0`, który kopiuje quaternion,
  pozycję i prędkość z ciała fizycznego do cache API. Odczyt następuje przed
  zapisami do cache. Kod ma ograniczony hook, weryfikację wersji i 64 bajtów,
  limit 256 rekordów/5 s oraz kontrolę przywrócenia kodu. Samo znalezienie
  funkcji i przegląd źródła nie dowodzą działającego hooka. Zapis nie zapewnia
  atomowości między wątkami, zegara symulacji, potwierdzenia wejścia ani
  zmierzonego zachowania kontroli FP. [Instrukcja i dowody](../trackmaniarl/project/openplanet/HeadlessPhysicalProbe005/README.md).
- `HeadlessTelemetry006` i `headless_telemetry006.py`. Każde żądanie ma nonce
  i numer sekwencji; odpowiedź zawiera nowe surowe 132 B oraz czas producenta.
  Brak rzeczywistych danych oznacza jawną niedostępność, nigdy zera lub stary
  stan. Usuwa to kolejkę obserwacji wysyłanych bez żądania; nie dowodzi
  świeżości getterów ani fazy fizyki. Opcja diagnostyczna
  `headless_windows_timing.py --fresh-telemetry-port 19179` zachowuje nagłówki,
  dokładne bajty i odpowiedzi niedostępności. [Instrukcja](../trackmaniarl/project/openplanet/HeadlessTelemetry006/README.md).

Nie zmieniano istniejącego Connect. Każda nowa wtyczka wymaga osobnego,
świadomego wgrania; pakiet zawiera wyłącznie źródła. 005 skopiowano w trakcie
przygotowań, lecz przed załadowaniem przeniesiono do prywatnego archiwum
`physical005-pending-not-loaded`. Nie uzbrojono żadnego hooka Windows.

Jedna rzeczywista próba odczytu FP przez Windows PSS została odrzucona przez
walidację kontekstu. Zwolnienie markera i snapshotu: oba kody0; nie uzyskano
zweryfikowanego stanu FP klienta. [Zakres i ograniczenia PSS](headless_vertical_windows_fp_context.md).

## Pozostałe warunki

Nowe próby Windows zostały wstrzymane, gdy kontroler przejął osobny pilot
SD-SAC. Nie zatrzymywano go, nie zmieniano jego profilu, STOP-ów ani automatów.
Własne helpery UI z aktywnymi lease zostały zakończone; ostatnia próba lease
odmówiła działania przed wysłaniem wejść. 003 jest collectorem jednorazowym,
004 pozostaje załadowanym, bezczynnym listenerem z przywróconym profilem.
Ich rozładowanie wymaga wolnego kontrolera/okna gry; nie restartowano silnika
skryptowego podczas cudzego pilota.

Do kwalifikacji porównania nadal potrzebne są: potwierdzony konsument wejść
w ticku Windows, zmierzona relacja raw body/cache, wyjaśnienie różnicy Y/VY,
rzeczywista meta oraz dowód zegara poza krótkim początkiem. Nie uruchomiono
porównawczego treningu, nie zmieniono SD-SAC i nie prognozuje się czasu do
tej samej jakości na podstawie samej przepustowości kolekcji.
