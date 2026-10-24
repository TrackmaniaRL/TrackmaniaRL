> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R6: jawny kontrakt Windows008 command → physics ACK

Status: projekt protokołu i warunków kwalifikacji, bez nowego uruchomienia gry.
R5, jego piny, action78, reset V7 i gate treningu pozostają zamrożone.
Windows008 ma osobną nazwę profilu, wersję wire oraz osobne źródła i dowody.

**Kolejka może dowieść, która wydana komenda sterowała każdym rzeczywistym
interwałem. Sama kolejka nie usuwa czasu potrzebnego zewnętrznemu aktorowi na
odebranie wyniku i wybranie kolejnej akcji.** Dla interaktywnego Pythona bez
pauzowania gry proponowany pierwszy wykonalny profil ma jawny, stały neutralny
prefiks D interwałów przed każdą akcją N1/N5. Wszystkie D+N interwałów należą do
tego samego nowego kontraktu przejścia na obu backendach. Nie jest to profil
natychmiastowego sterowania V7.

## Co wynika z istniejących źródeł

Lokalny `<USER_HOME>/OpenplanetNext/Openplanet.h`, SHA256
`16b50c345edebd9bd139ba596a1401b83f1a2d0932a6a50912bc43bc25ce0332`, deklaruje
wersję gry 2026-02-02 17:51:19 i Openplanet 1.29.14 / `6fd4200f`.

| Źródło | Ustalenie | Ograniczenie |
| --- | --- | --- |
| SDK, linie 23456–23458 | `CSmScriptPlayer.InputSteer`, `InputGasPedal`, `InputIsBraking` są `const`; ostatnie pole jest `bool`. | Getter nie jest setterem i nie wiąże wartości z command ID. |
| SDK, `CGameScriptVehicle`, linie 11298–11316 | `AccelInput` jest `const`; dostępne `SetEngineActivated` steruje aktywacją silnika. | Nie ma tu kwalifikowanego API enqueue literalnego SGB z czasem i ID. |
| SDK, `CInputPort`, linie 21739–21786 | Są parametry próbkowania, sztucznego opóźnienia i statystyki wejść. | Statystyka liczby wejść nie dowodzi konsumpcji konkretnej komendy. |
| [README007](../trackmaniarl/project/openplanet/HeadlessInputConsumer007/README.md) i `Main.as:72` | Callback po zapisie filtered steer odczytuje actual gas/brake/steer przy vehicle `+0x98/+0x9c/+0xa0`, rawpose przy `+0x12e0` i dosłowne R13D. | Whole-step PRE/POST i znaczenie R13D jako zegara są nadal niekwalifikowane. |
| [R5 replay](input_consumer_native_replay.md) | Actual SGB[i] → rawpose[i+1] daje 425/425 zgodnych physical10 w dwóch przejazdach. | To odtwarzanie zaobserwowanych wejść, bez issued-command ACK. |
| Zapis `physical-input-apply027.txt`, odrzucony kandydat | Funkcja `0x7eb2c0` ma klasę `0x0a044000`, rozmiar `0xc8`, zidentyfikowaną jako `CSceneFxHeadTrack`. | To efekt headtrack, nie vehicle `0x0a020000`. Zbieżne offsety `+0x98/+0x9c/+0xa0` nie dowodzą żadnego writera wejść pojazdu. |
| Zapis `physical-input-clamp028.txt` | RVA `0x83de87` może zmienić gas; `0x83deb3` i `0x83df00` mogą zerować brake. | Wczesny store i natychmiastowy readback nie wystarczają, jeżeli późniejsza ścieżka zmienia wejście. |
| R6 `probe002/wheel_first.txt` | Binding `0x8415af` przekazuje vehicle RBX jako RCX do `0x843190`; `0x84319d` kopiuje RCX do R8. `0x84327a` czyta `[R8+0x9c]` przez MOVSS, `0x843286` używa COMISS, `0x8432ac` MULSS. | To niezależny dowód scalar float32 brake. Dotychczasowe próby 007 z brake 0 nie kwalifikują jeszcze zachowania brake 1. |
| Wcześniejszy dispatcher `0x12e86c0` | Kolejka komend pisze osobną strukturę vehicle `+0x1750`. | Enum tej kolejki nie został utożsamiony z literalnym SGB; nie należy wykorzystać go jako gotowego ACK. |

Prywatne zapisy pozostają w
`<USER_HOME>/tmrl-headless-parity-20261021-005/windows-input-consumer/`:

| Plik | SHA256 |
| --- | --- |
| `physical-input-apply027.txt` | `d03fb8f310f351113c9257e579f216384a5936395afc59293a714bbae91cf1bd` |
| `physical-input-clamp028.txt` | `1f3253b8729a67b47977901538256f7a97a712fdc1b3eda23fdaa4544f27247c` |
| `physical-input-producers026.json` | `4d134a5492d880a7ab2cf02207b8357d03d04114cea042e07b2c6dbe16b24ab6` |

026 sam deklaruje `instruction_boundaries_qualified=false` i
`participant_base_identity_qualified=false`. Jego surowy inventory nie jest
kwalifikacją kolejnego miejsca zapisu. Badanie punktu PRE w funkcji `0x841500`
oraz kolejności writerów jest osobnym zadaniem R6. Ten dokument nie uznaje
kandydata PRE za dowiedziony punkt wykonania.

Nowa analiza R6 wskazuje wąski warunkowy profil normalny: `176c & 0x40f == 0`,
`1370 & 0x0f == 0`, selector `1600 == 0`, word `1768 == 0` i float `1374 == 0`.
Są to warunki ze źródła dla wybranej gałęzi, nie oczekiwane wartości wywnioskowane
z próbki i nie zakończona kwalifikacja live. Szerokość word1768 to uint16;
float1374 trzeba zachować jako literalne bity i podać jawną maskę zgodną z
dowodem źródłowym. Samo sprawdzenie trzech pierwszych pól jest niewystarczające.
Punkt przed clampem w wywołującym rodzicu `0x851f96` oraz węższy punkt w funkcji
`0x841500` nadal wymagają osobnych pełnych dowodów kolejności i ABI.

## Osobny profil i granica odpowiedzialności

Proponowane identyfikatory:

- wire: `tmrl.windows-command-physics/v8`;
- wykonawca wejść: `windows-owned-vehicle-literal-sgb/v8`;
- dowód taśmy: `immutable-preloaded-tape/v8`;
- interaktywny profil hosta: `fixed-neutral-prefix-D/current-command-N/v8`;
- przyszły aktor wewnątrz procesu: `same-boundary-inprocess-policy/v8`, osobno
  kwalifikowany i początkowo niedostępny.

008 nie zastępuje gamepada ani stockowego kontrolera pod dotychczasową nazwą.
Ewentualne zapisy dokładnych bitów SGB w funkcji fizyki są nową metodą sterowania,
wymagającą jawnie załadowanego osobnego modułu i własnych pinów. Nie wolno nazwać
obserwacji stałej wartości inputu dowodem wykonania komendy o takim samym SGB.

SGB ma kolejność `[steer, gas, brake]` i trzy dokładne uint32 słowa float32.
Gas/brake actual są float32 `0x00000000` albo `0x3f800000`; zapis integer `1`
do actual `+0x9c` byłby błędem. Boolean braking z vehicle/cache pozostaje osobnym
polem obserwacji. Action78 zachowuje dotychczasowe pięć SGB, w tym brake tap
tylko w pierwszym interwale. Żadnej konwersji przez decimal, gamepad lub
automatyczne dopasowanie najbliższego skrętu.

## Minimalny protokół gotowy do implementacji

Wszystkie wiadomości zawierają wire, nonce sesji i generację resetu. Arm wiąże
SHA gry, Openplanet, źródeł modułu, dokładne prefiksy hooków, mapę, playground,
lokalnego participant i klasę/pointer jego vehicle. ID rośnie w obrębie resetu;
tego samego ID nie wolno przyjąć z innym payloadem. Stare generation, late
command, nakładanie interwałów i zmiana ownera są twardymi błędami.

| Wiadomość | Znaczenie |
| --- | --- |
| `hello` | Moduł idle, wersje, możliwości i jawne limity. Nie świadczy o instalacji hooka. |
| `arm` | Oddzielna zgoda na nowe, dokładnie przypięte hooki i ograniczony eksperyment. |
| `boundary` | Własny numer callbacka, literalny R13D, rawpose10, owner/generation. Jednostki czasu pozostają osobnym dowodem. |
| `enqueue` | Immutable `command_id`, `source_boundary_hash`, `start_boundary`, `N` równe 1 lub 5, N dokładnych SGB i payload SHA. |
| `queued` | Payload zweryfikowany i opublikowany do kolejki. **Nie jest physics ACK.** |
| `interval` | Faktycznie wybrany command ID/offset, dowód źródła kolejki, SGB przed zapisem, żądane bity, readback i odczyt w kwalifikowanym consumerze oraz dwie rzeczywiste granice rawpose. |
| `physics_ack` | Dokładnie N kolejnych kompletnych `interval`, bez luk i obcych zapisów; ostatni next-boundary istnieje. |
| `fault` | Powód i ostatnia kompletna granica; samo uszkodzenie protokołu nie jest metą. |
| `stop` / `status` | Osobne dowody zwolnienia sterowania, przywrócenia hooków i zamknięcia sesji. |

Przykład nagłówka syntetycznej komendy protokołu, **nie wykonany pomiar**:

```json
{
  "protocol": "tmrl.windows-command-physics/v8",
  "op": "enqueue",
  "reset_generation": 1,
  "command_id": 4,
  "source_boundary_hash": "<SHA256 rzeczywistej granicy decyzji>",
  "start_boundary": 20,
  "n": 1,
  "sgb_words": [[0, 1065353216, 0]],
  "payload_sha256": "<SHA256 całego kanonicznego payloadu>"
}
```

Wire dołącza nonce; przykładowe placeholdery powyżej muszą być odrzucone przez
walidator. Nie tworzymy fikcyjnego ACK z tego przykładu.

### Kiedy powstaje physics ACK

1. Callback dla właściwego vehicle atomowo pobiera gotowy immutable slot kolejki
   i wiąże lokalny execution ordinal z command ID i payload SHA.
2. W kwalifikowanym punkcie przed konsumpcją wpisuje wyłącznie dozwolone trzy
   kanały, zachowuje surowe before/after i natychmiastowy readback. Kolejka jest
   metadanymi naszego wykonawcy; nie udajemy, że gra ma takie command ID.
3. Odczyt w rzeczywistym consumerze musi potwierdzić te same bity przy tej samej
   tożsamości/generacji/etapie. Sam readback miejsca zapisu jest za słaby wobec
   późniejszych writerów. Zmodyfikowany filtered steer jest obserwacją, nie
   czwartym kanałem do niezależnego wpisania.
4. Następna rzeczywista granica tego samego consumera kończy interwał i dostarcza
   rawpose10. Jednego licznika R13D `+10` nie uznajemy automatycznie za 10 ms;
   raw wartość, ordinal i osobno kwalifikowane odwzorowanie czasu zostają osobno.
5. Dopiero komplet N interwałów tworzy ACK. Ostatni control sample bez kolejnego
   rawpose pozostaje niekompletny. Duplicate/gap/owner change nie są usuwane z
   raportu, interpolowane ani naprawiane przez offset.

W callbacku nie wolno czekać na socket, odpalać learnera, alokować/serializować
JSON ani wykonywać blokującego I/O. Wszystkie sloty i bufory dowodów są wstępnie
zaalokowane. Wątek transportu publikuje ukończone sloty; callback odczytuje je
przez mechanizm o dowiedzionym release/acquire lub równoważnej semantyce
uruchomienia. Sam globalny bool z callbacka 007 nie dowodzi bezpieczeństwa
współbieżnej kolejki. Brak takiego dowodu blokuje live writer.

## Taśma z góry a interaktywny aktor

### Pierwsza rzeczywista granica i reset

008 musi zachować pierwszą własną fizyczną granicę z surowymi physical10.
Zmiennego opóźnienia pobudki callbacka, np. obserwowanego wcześniej przy
50/60 ms API, nie wolno wyciąć z opisu resetu lub zastąpić zmyśloną granicą.
Można wybrać dokładnie jeden neutralny native content anchor: najpierw wymagana
jest unikalność physical6, następnie pełna zgodność physical10. Niejednoznaczne
stany stacjonarne i zgodne physical6 z innym quaternion są odrzucane.

Native wykonuje świeży, zapisany neutralny seek do tej rzeczywistej kotwicy.
Seek/reset są administracyjne i mają zero kredytu learnera. Kontrakt jawnie
wiąże pierwszą granicę decyzji z tym stanem. Gdy Windows wykona dalsze ticki
przed startem taśmy, każdy z nich musi wystąpić w jawnym, skończonym neutralnym
bootstrapie; alternatywnie należy zaliczyć cały rzeczywisty czas do pierwszego
przejścia w tym samym profilu na obu backendach. Niedozwolony jest logiczny
skok ze starego API stanu do późniejszego PRE.

### Immutable preloaded tape

Cała taśma ID/SGB/N i jej SHA są przyjęte **przed** pierwszym wykonywanym
interwałem. Znane wcześniej komendy można wykonywać bez przerw, a następnie
potwierdzić przez raw consumer i kolejną rawpose. Duplikaty SGB nie kasują ID.
Ta próba kwalifikuje dokładność wykonawcy, ciągłość ticków i fazę. Nie dowodzi,
że akcja została wybrana po obejrzeniu poprzedniego wyniku przez policy.

W czystym modelu `Publication.queued_after_sequence=-1` oznacza commit taśmy
przed akwizycją pierwszej granicy; `source_boundary_sha256=None` jawnie wyklucza
twierdzenie o interaktywnej decyzji. Arm wiąże jej digest z później rzeczywiście
odebraną kotwicą. Interaktywny profil musi podać prawdziwy hash granicy decyzji
i publikację po tej granicy, a przed początkiem akcji.

Na końcu taśmy jest jawny, skończony neutralny ogon administracyjny, potrzebny
do odczytu ostatniej rawpose i zwolnienia sterowania. Jego ticki i source ID są
zapisane, mają zero kredytu learnera i nie wolno dopisać ich do N ostatniej
akcji. Brak oczekiwanego next-boundary daje incomplete, nie fabricated finish.

### Dlaczego samo enqueue nie tworzy natychmiastowego host-loop

Dla action N1 ACK stanu na granicy b+1 może być wysłany dopiero po wystąpieniu
tej granicy. Odbiór, inference i dostarczenie następnej komendy następują później.
Tymczasem gra może wykonać dalsze ticki. Zwiększenie kolejki nie dostarcza
przyszłej decyzji policy, której jeszcze nie wybrano.

Trzymanie poprzedniej akcji do przyjścia nowej zmienia N, choćby payload nadal
deklarował N1/N5. Ciche neutralizowanie i pomijanie tych ticków zmienia przejście
tak samo. Przyjęcie wcześniejszej obserwacji dla przyszłego ticka to profil
opóźnionego sterowania, wymagający osobnego kontraktu.

### Wykonalny host-loop bez pauzy: stały neutralny prefiks D

Wartość D zostaje ustalona przed pomiarem na podstawie jawnego budżetu
transportu/inference, a nie dobierana do zgodności trajektorii. Początkowy
eksperyment może wybrać np. D=10 callback interwałów; nie nazywa ich milisekundami
bez kwalifikacji zegara. Budżet i błędy deadline pozostają w raporcie.

| Granica/interwał | Zdarzenie |
| --- | --- |
| b | Zapis rawpose decyzji, wysłanie boundary; początek jawnego neutralnego prefiksu. |
| b..b+D−1 | Dokładnie D neutralnych interwałów z własnymi zapisami actual SGB i rawpose. |
| przed b+D | Aktor dostarcza komendę wybraną z rawpose b; executor potwierdza payload źródłowym hashem b. |
| b+D..b+D+N−1 | Dokładnie N interwałów akcji i jej wersjonowanej sekwencji SGB. |
| b+D+N | Kompletna następna rawpose, ACK przejścia b→b+D+N; od razu kolejny jawny neutralny prefiks. |

Prefiks jest z góry autoryzowaną częścią profilu epizodu. Nie pojawia się jako
samodzielnie wymyślona akcja policy. Spóźniona komenda powoduje abort; nie
przedłużamy prefiksu, nie uruchamiamy akcji później i nie przypisujemy zwłoki
poprzedniej akcji. Niepełne administracyjne ticki są zachowane z rzeczywistym
czasem; nie tworzą pełnego doświadczenia learnera.

Native musi wykonać identyczne D neutralnych + N akcji z jedną obserwacją decyzji
na b i jednym logicznym doświadczeniem. Można dzielić prefix na V7 podżądania
o rozmiarze najwyżej 5, bez nowych decyzji policy i bez dodatkowego kredytu.
Discount, time penalty i update budget mają używać **całego** D+N rzeczywistego
czasu. To wymaga nowego input schedule i nowego digestu common contract;
istniejących checkpointów nie wolno oznaczyć jako zgodnych przez podmianę flagi.
Ten dokument nie wprowadza zmian do algorytmów ani scopegate.

Alternatywa D=0 wymaga aktora działającego na tej samej granicy w procesie gry
lub niezależnie dowiedzionego mechanizmu zatrzymywania symulacji. Pierwsza droga
wymaga osobnej kwalifikacji wykonania modelu, wag, preprocessing, tie-break,
wersji learnera oraz najgorszego czasu callbacka. Nie jest zapewniana przez
Angelscript callback lub zewnętrzną kolejkę samą w sobie.

## Limity, koniec epizodu i cleanup

Pierwszy live eksperyment: najwyżej 5 s, 256 zakończonych interwałów,
64 komendy, 64 MiB dowodów, jeden vehicle i reset. Queue overflow/underflow,
zły nonce, utrata ownera/focus/mapy, callback reentry, brak kolejnej rawpose,
R13D gap/wrap i opóźniony payload kończą próbę jako fault. Podstawowe limity są
sprawdzane także w callbacku, nie tylko w pętli socketu.

Na granicy, na której brak kolejnego autoryzowanego wejścia, nie wolno
samowolnie utrzymywać poprzedniego SGB. Jawna procedura abort/release neutralizuje
wyłącznie własne wejścia, zachowuje pozostałe rzeczywiste ticki i kończy
eksperyment; jej neutralny cleanup nie jest przejściem RL. Nie wolno blokować
symulacji oczekiwaniem na komendę.

EOF socketu, pusty queue, cap lub usunięcie hooka nie oznaczają mety. Oryginalny
wynik mety/checkpointów musi mieć osobne źródło, tożsamość epizodu i związek z
rzeczywistą granicą. W native zachowujemy sprawdzony pełny validator + TMRESULT;
w Windows trzeba osobno zakwalifikować odczyt autentycznego finish/checkpoint,
nie wpisywać `EndTime` lub pól physics. Możliwy jest ragged N tylko po osobnym
dowodzie prawdziwej mety i faktycznej liczby zakończonych interwałów. Domyślnie
niekompletny końcowy command nie otrzymuje ACK N.

Cleanup próbuje niezależnie: release własnych sterowań, zdjęcie wszystkich
nowych hooków, readback oryginalnych 64 B, przywrócenie profilu 004, zamknięcie
socketu/aktora i zapis raportów, także gdy wcześniejszy etap rzuci wyjątek.
TTL/pętla Main nie są systemowym watchdogiem po awarii procesu lub VM; nie
deklarujemy takiej gwarancji. Nie zatrzymujemy obcych aktorów, procesów lub gry.

## Kolejność dowodów i minimum gate

Implementacja czystej oceny to
[`scripts/command_ack_r6_protocol.py`](../scripts/command_ack_r6_protocol.py),
a wektory są w
[`test_command_ack_r6_protocol.py`](../tests/unit/trackmania/test_command_ack_r6_protocol.py).
`verify_episode(Profile, publications, boundaries, consumptions)` sprawdza
oddzielne immutable komendy, fakty publikacji, granice i consumer records.
Żąda wszystkich pięciu jawnych masek normalnych gate (`+0x176c`, `+0x1370`,
selector `+0x1600`, word `+0x1768`, float `+0x1374`) oraz hasha ich osobnego dowodu źródłowego; nie wylicza
oczekiwanych bitów z pierwszej próbki. Maski w unit testach są syntetycznymi
fixture, nie potwierdzeniem tych nieukończonych badań R6.

`ProtocolAssessment` zachowuje `physical_execution_qualified=false` i
`paired_training_qualified=false` jako pola, których konstruktor nie pozwala
przestawić. Zgodność czystego ledgeru nie dowodzi autentyczności jego producenta.
Moduł nie ma I/O, klienta gry, hooka, `close()` ani ścieżki do scopegate.
Testy cleanup samego wykonawcy będą należeć do osobnego modułu Windows008;
ten model nie symuluje udanego zdjęcia hooka.

1. **Czyste testy protokołu:** identyczne ID z innym payloadem, nonce/generation,
   late start, N nie w {1,5}, błędne długości, brake bool zamiast float bits,
   gap/duplicate/reentry/owner loss, końcowy sample bez next-pose, błąd cleanup.
   Syntetyczne wektory mają zero kwalifikacji live i nie odblokowują gate.
2. **Source proof:** pełne ABI hooków, ostatni możliwy writer przed konsumpcją,
   dokładne skradzione instrukcje, wszystkie branche pomijające consumer,
   release/acquire i identity. Ograniczone disassembly nie zastępuje tego dowodu.
3. **Bounded immutable tape:** wszystkie 78 akcji N1/N5, w osobnych świeżych
   seriach mieszczących się w cap, np. po najwyżej 32 akcje, tap/hold brake,
   zmiany neutral/gas/steer i powtarzające się identyczne SGB z różnymi ID.
   Każdy interwał musi mieć actual input i następny pose; błędy zachowane.
4. **Physical/clock join:** ciągły ledger bez API sampling gaps, dokładne
   physical10 porównanie, predeclared faza oraz fit/heldout jednostek czasu w
   określonym zakresie. Nie dopasowywać offsetu dla zniknięcia błędów.
5. **Interactive deadline proof:** akcja rzeczywiście wybrana z wcześniejszego
   boundary hash, deadline D spełniony, zero ukrytych ticków, identyczny prefiks
   na native i Windows. Zaliczenie immutable tape nie zastępuje tego kroku.
6. **Paired gate:** wspólny profil obserwacji, akcji, resetu, czasu, reward,
   terminal i duration; przypięte wagi/seed/geometry; pełne cleanup i źródła
   dowodów. Brak któregokolwiek dowodu pozostawia `paired_training_qualified=false`.

ACK dowodzi wykonania polecenia, a nie zgodności trajektorii wszystkich map.
Zgodność jednej trajektorii nie dowodzi interaktywnego ACK. Oba dowody oraz
kontrakt opóźnienia są konieczne przed uczciwym porównaniem uczenia.

## Weryfikacja samego protokołu

22 października 2026 czasu lokalnego: końcowa wersja z pięcioma gate fields ma
**29 testów PASS w 4,47 s**. Ruff check/format i mypy strict dla modułu PASS.
Pierwszy receipt zachowuje także 26 PASS przed rozszerzeniem gate; nie został
podmieniony. Testy używają wyłącznie
syntetycznych fixture. Nie uruchomiono native, Windows, hooka ani treningu.
Receipt pytest jest w
`<USER_HOME>/tmrl-headless-parity-20261021-006/command-ack-protocol002/receipt.json`;
kopie tekstowych wyników są pod
`F:/AITrackmania/reports/headless-training-parity-20261021-r6/command-protocol002/`.

Odtworzenie w WSL, ze świeżym katalogiem ext4:

```sh
export PYTHONPATH=/mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
export PYTHONDONTWRITEBYTECODE=1
PY=<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python
OUT=<USER_HOME>/tmrl-headless-parity-20261021-006/command-ack-protocol-repeat001
mkdir "$OUT"
cd "$PYTHONPATH"
$PY -m pytest -q --basetemp="$OUT/tmp" -o cache_dir="$OUT/cache" \
  tests/unit/trackmania/test_command_ack_r6_protocol.py
```

Mypy wykonano hostowym istniejącym interpreterem bez instalacji pakietów:

```powershell
& 'H:/Studia/inzynierskie/inzynierkav2/AITrackmania/.venv/Scripts/python.exe' `
  -m mypy --follow-imports=silent `
  --cache-dir F:/AITrackmania/reports/headless-training-parity-20261021-r6/command-protocol001/mypy-cache `
  -m scripts.command_ack_r6_protocol
```
