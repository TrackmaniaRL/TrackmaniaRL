> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Headless 1.2.14 — naprawa i kwalifikacja, 22.10.2026

**Status: gotowy do ograniczonych treningów native-only. Nadal niezakwalifikowany do
kampanii ani niezależnego porównania szybkości uczenia z klientem Windows.**

Aktualizacja r7 dodaje osobny rzeczywisty trening **native-only**:
`scripts.headless_native_train`, pełne epizody V7, świeżą sesję na epizod,
atomowe checkpointy i wznowienie learnera/replaya/RNG. Natywna jednostka czasu
ma dowód źródłowy silnika, niezależny od zakresu zegara Windows. Ten profil
może trenować QR-DQN/common6/MLP256 od nowych wag; zachowuje osobny digest
kontraktu i nie odblokowuje porównania z Windows. Instrukcja i ograniczenia:
[headless-native-training.md](headless-native-training.md).

Aktualizacja r6 jest wyłącznie offline. Powstał osobny model walidacji przyszłego
protokołu command → consumer → next physical state. Nie steruje grą i zachowuje
obie flagi kwalifikacji fizycznej/treningowej jako `false`. Analiza odrzuciła
pozorny writer `0x7eb2c0`: należy do efektu `CSceneFxHeadTrack`, a nie pojazdu.
Ustalono wcześniejszy kandydat PRE i dodatkowe warunki trybu pojazdu, lecz
nie zakończono dowodu całego kroku i bezpiecznego usuwania hooków. Nie znaleziono
też publicznej deklaracji pauzy świata z pojedynczym krokiem fizyki. R6 nie
uruchamiał gry, silnika ani treningu; szczegóły w
[diagnostyce r6](headless-parity-diagnostics-r6.md).

Aktualizacja r5: wykryto pięć wejść różniących taśmę diagnostyczną od
dostarczonego ghosta. Standardowy publiczny adapter V7 osiągnął metę17715ms
po1780 rzeczywistych akcjach N1, ze zwykłym neutralnym resetem i naturalnym EOF.
Pełny wynik oryginalnego walidatora i TMRESULT daje `terminated=true`, zachowując
`IsValid=false`. Oddzielny eksperyment release/drain potwierdza ten wynik,
ale jego autonomiczna część nie daje przejść ani kredytu learnera. Dłuższe dopasowanie
obserwacji potwierdziło 794/794 rzeczywistych stanów przez 8000 ms, z siedmioma
brakującymi próbkami Windows jawnymi w raporcie. To nie kwalifikacja ciągłych
wejść ani zgoda na porównanie treningów. Odczyt rzeczywistego konsumenta Windows
dał425/425 bitowo zgodnych przejść w dwóch próbach, także Y/VY i quaternion, przy jawnym replayu aktualnych
wejść do następnego stanu. Nadal nie jest to ACK komendy niezależnego aktora.
Zobacz [dowód mety r5](headless-finish-r5.md),
[dowód zegara r5](headless-clock-r5.md) i
[diagnostykę r5](headless-parity-diagnostics-r5.md).

Nowy profil V7 usuwa opóźnienie bieżącej akcji i resetuje do fizycznego stanu
odpowiadającego Windows race10. Działają wspólne cechy, nagroda, replay,
learner i checkpoint. Starsze odtworzenie wejść odczytanych z API Windows dało
805/844 zgodnych próbek. R5 rozdzielił cache od konsumenta fizyki:425/425
rzeczywistych interwałów jest zgodnych przy właściwych wejściach, a w21
pakietach API wykazano odchylenie skrętu o1ULP względem dopasowanego konsumenta.
Kontroler Windows nadal nie ma kwalifikowanego ACK komendy w ticku.
Nie wykonano pozorowanego porównania treningów.

Gałąź `codex/headless-training-parity` bazuje na SDK 1.2.14,
`f4c01857dad1d91955842430edb18cd89f12fa15`. Poprzedni etap V6 zapisano w
`df815f2df313c036f873abdef111865f8d809add`; opis jego pomiarów w tamtym commicie
jest historyczny. Poniższy dokument zastępuje wcześniejsze wnioski o braku
możliwości zmiany bieżącego rekordu i nieznanym przesunięciu zegara.

## Naprawione błędy

1. **Bieżące wejście.** Zakaz zapisu `k<=p` pochodził z protokołu V6.
   Breakpoint readera występuje przed załadowaniem bieżącego rekordu. Osobny
   protokół `immediate-current/v7`, op6, zapisuje `p..p+n−1` przed tym odczytem.
   Wersja ramek, audity, manifest i pin nowego shima są odrębne. Weryfikowane są
   rzeczywiste POST, zakres, epoch, tożsamość procesu i wyłączna zmiana kanałów
   SGB. N1/N5 nie potrzebują flush ani ukrytej decyzji. V6 zachowuje stare piny
   i ograniczenia; nie odziedziczył kwalifikacji V7.
2. **Reset.** `neutral-race-start/v7` uruchamia rodzica przy native2190,
   wykonuje jawny, audytowany neutralny interwał administracyjny i zwraca
   native2200. Nie przyznaje za niego kredytu ani przejścia polityki. Sześć pól
   pozycji/prędkości jest bitowo zgodnych ze stanem Windows race10. Stary gazowy
   reset2230 pozostaje wyłącznie profilem historycznym.
3. **Pochodzenie czasu.** Dopasowanie dokładnych float32 pozycji i prędkości
   daje `Windows race_ms = nativeT − 2190` w zmierzonym zakresie tej mapy i resetu.
   Poprzedni zapis: 11 zgodnych próbek, 6 dopasowania i 5 walidacji, 500 ms.
   Świeży gęsty zapis: 65 próbek, 33/32, 640 ms. Relacja fazowa readera jest
   osobnym dowodem: zmierzone wejście Windows w T wpływa na POST w T+10.
   Adapter treningowy odrzuca wyjście poza zakres wybranego dowodu zegara.
   Nie jest to kalibracja wszystkich map, zegara hosta ani pozostałych pól.
4. **Terminalny NEXT w replayu.** Jawne `None` końcowej obserwacji mylono
   z brakiem nadpisania. Materializacja rozróżnia obecność klucza i jego wartość.
   Terminal zachowuje `None`, ma zerowy bootstrap i działa po checkpoint/resume
   oraz zawinięciu FIFO. Brak NEXT przy nieterminalnym kroku i truncation
   powoduje błąd. Nie powstają tensory zerowe udające pomiar.
5. **Twarda awaria aktora.** Niezależny Linux subreaper uruchamia driver
   i śledzi prywatną dzierżawę aktora. EOF tej dzierżawy, także przy SIGKILL
   aktora w fabryce, uruchamia ograniczone sprzątanie potomków. Sygnały używają
   pidfd i pełnej tożsamości procesu. Test obejmuje potomka z osobną sesją
   i pozostawienie obcego procesu przy życiu. Nadzorca nie skanuje `/proc`
   w pętli wykonywania fizyki.
6. **Narastanie dowodów.** V7 ma jawny `max_trace_bytes`, domyślnie 64 MiB.
   Liczy trace, work i własny katalog RX `/tmp`, sprawdza budżet przed odczytem,
   zapisem i publikacją. Przekroczenie zamyka sesję bez publikacji przejścia.
   To sprawdzany limit aplikacyjny: producent natywny może nieznacznie przekroczyć
   próg pomiędzy kontrolami. Nie jest to kwota jądra ani limit pamięci drzewa.

Szczegóły adresów, źródeł, builda, fazy i niezgodnych próbek:
[headless-immediate-v7-engine-evidence.md](headless-immediate-v7-engine-evidence.md).
Źródłowy patch z manifestem 179 plików i receiptem builda jest w
`trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/`.
Nie dołączono binarek gry, Packs, snapshotów, ghostów ani wag.

Reverse Engineer Anything 6.7.0 z Ghidrą 12.1.4 zainstalowano osobno w
`<USER_HOME>/tmrl-rea-tools-20261021-001`; nie zmieniono globalnej konfiguracji
Codex/MCP. Udana analiza dotyczy jawnie zmapowanego wycinka readera 1024 B,
nie całego programu. Timeout wcześniejszej analizy całego ELF został zachowany.
Źródło narzędzia: [morluto/rea](https://github.com/morluto/rea).

## Wspólny kontrakt modelu i danych

`tmrl.common-kinematics6/v1` zawiera wyłącznie rzeczywiście dostępne pola:

| Cechy | Connect33 | Native76 | Dzielnik |
| --- | --- | --- | --- |
| pozycja XYZ | 4–6 | physicalPosition | 1000 |
| prędkość XYZ | 7–9 | physicalVelocity | 100 |

Nie zawiera czasu, pending, filteredSteer, obrotów, orientacji ani stanu kół.
Brakujące wymagane pole powoduje błąd. V7 raw zachowuje literalne appliedSGB
oraz bieżący **niewydany** rekord readera osobno; pendingSGB=None. Nie tworzy
z niego starej obserwacji native44.

Opcjonalny `tmrl.common-kinematics8/v1` dodaje RPM i surowy bieg, lecz RPM
różni się już na początku Windows/native. Common8 ma testy infrastruktury,
nie kwalifikację fizyczną. Do bieżących ograniczonych prób wybieraj common6.

Wspólny model to istniejący QR-DQN, MLP256, 32 kwantyle, 78 akcji, CPU float32,
IdentityTemporalCore, n-step1, Adam lr0.0001, epsilon0.1. Profil modelu, wymiar,
cechy, skale, nagroda, zegar, reset i harmonogram wiążą checkpoint. **Wymagane są
nowe wagi**; checkpointy native44, common8 i kampanii GNN są niezgodne.

Nagroda używa tej samej pełnej, przypiętej polilinii float64: najbliższa ciągła
projekcja na wszystkie odcinki, deterministyczny wybór przy remisie, bez historii
lub przeszukiwania okna zależnego od backendu. Powtórzone punkty pozostają
w źródle, a zerowe odcinki nie uczestniczą w projekcji. Jednostki to współrzędne
silnika, bez domyślnego utożsamiania z metrami. Wzór:
`0.1*delta_progress − 0.2*elapsed_seconds − 0.001*min(deviation,50) + 20*finish`.
To osobny jawny profil, nie zmiana nagrody zamrożonej kampanii.

Dyskonto to `0.995**(actual_duration_seconds/0.05)`, przy terminalu zero.
Kredyt aktualizacji: `0.25*actual_duration_seconds/0.05`, liczony z każdego
przyjętego przejścia, także po zawinięciu replaya; końcowa reszta jest zapisana.
Truncation zachowuje prawdziwy endpoint i bootstrap. Limit epizodu jest
wykonywany przez środowisko, także gdy pozostaje mniej niż N5. Harness nie
przerabia własnej przerwanej pętli w zakończenie epizodu.

`WindowsCommonAdapter` przechowuje oddzielnie wydane ID i odczytany przebieg
wejść z rzeczywistym race_ms. Nie uznaje samych wartości API za potwierdzenie
zastosowania wejścia fizycznego. Niezależny trening porównawczy wymaga kompletnej
bramki `tmrl.common-paired-qualification/v1`: piny dowodów resetu, wejść, czasu,
obserwacji i nagrody, oba kontrakty, identyczny hash początkowego modelu oraz
zakres mapy/reference/seed/budżetu. Obecne dowody nie spełniają tej bramki.
Tryb `--infrastructure-smoke` jawnie testuje wyłącznie native, bez jej pozorowania.

Kampania Windows wymaga GNN Simba V6 i BoundaryGraphFeaturePipelineV5:
`physics[60]`, `track[3,88]`, `context[CONTEXT_V3_DIM]`, `recovery[8]`.
Headless nie dostarcza pomiarów poślizgu i materiałów każdego koła, adherence,
flying duration oraz odpowiadającego kontekstu/recovery. Nie zakwalifikowano
kwaternionu jako zamiennika Windows Dir/Up. Connect33 z kolei nie ma native
pending i filteredSteer. Publiczne pliki API Openplanet istnieją lokalnie,
lecz mają datę 19.10; nie zebrano świeżego działającego dumpu refleksji ani
nie przedstawiono źródłowego collectora jako skompilowanego narzędzia.

## Wyniki porównania

Świeży pomiar Windows trwał 21.437 s wewnątrz programu, obejmując preflight,
resety, gęsty neutral i pięć taśm, zapis oraz zamknięcie kontrolera. Nie obejmuje
importów. Użyto jednego kontrolera i mutexu
`Global\TrackmaniaRL.ComparisonController`, sprawdzenia mapy i standardowego
fokusu. Nie wystąpiła odmowa fokusu. Jedna wcześniejsza próba z brakiem gracza
została zachowana; odzyskanie trybu walidacji wykonano standardowym guarded reset.

Po równoważnym resecie i dopasowaniu czasów **te same nominalne ID** dały zgodny
neutral, lecz cztery jazdy miały zgodną tylko początkową próbkę. Odległości końca:
gaz 0.56873, skręt 1.41257, hamulec 0.66400, tap 2.33258 jednostek silnika.
Nie zostały uznane za błąd fizyki.

Odczyt Windows wykazał zmienne opóźnienie pierwszego wejścia oraz efektywny
skręt −0.165354326 zamiast nominalnego −0.166666672. Po odtworzeniu **faktycznie
odczytanych** SGB co 10 ms, z potwierdzeniem ich bajtów w native POST:

| Taśma | Dokładne próbki posXYZ/velXYZ | Ograniczenie |
| --- | --- | --- |
| neutral | 51/51 | race10…510 |
| gaz | 190/190 | przerwano na brakującym pakiecie race1910 |
| łagodny skręt | 162/201 | drobne różnice wyłącznie Y/VY od race1630 |
| pełny hamulec | 201/201 | race10…2010 |
| brake tap | 201/201 | race20…2020 |

Końcowe pozycje są identyczne we wszystkich pięciu odtworzonych prefiksach.
Maksymalna różnica Y przy skręcie to 9.536743164e−7, VY 4.585832357e−6. Nie zmieniono
progów ani oczekiwanych hashy: pełny wynik dokładnego parytetu pozostaje false.
Brakującego pakietu nie interpolowano. Odtworzenie zapisu Windows nie jest
niezależnym treningiem ani pomiarem przyspieszenia uczenia headless.

## Testy, awarie i zasoby

[Wyniki r4](headless-parity-diagnostics-r4.md) zawierają dalszą poprawkę resetu
Windows (świeża obserwacja po confirm_ready), rzeczywistą kompilację/próbę005/006,
cztery live testy rollbacku, naprawę odczytu PSS i jawny eksperyment x87.
Pełny wynik kampanii nadal nie jest zakwalifikowany; starsze wyniki poniżej
zachowano jako historię dowodów.

Dalsze wyniki r3 i dokładne granice nowych narzędzi opisuje
[kontynuacja diagnostyki](headless-parity-diagnostics-r3.md). Świeży collector003
został uruchomiony; wyłączenie FakeInputLag004 nie usunęło obserwowanej zwłoki.
Odtworzenia native i odczyt FP są powtarzalne. Na etapie r3 przyczyna różnicy
pionowej była nieustalona, a próbniki005/006 nie miały jeszcze potwierdzonej
kompilacji. R4 potwierdził ich działanie, r5 dołożył rzeczywistego konsumenta007
i425 zgodnych interwałów; starsze wyniki poniżej pozostają historyczne.
Granica ostatniego readera jest jawnie
sprawdzana przed komendą; EOF po `FKPOST reject337` nie jest metą.

Aktualne raporty: `F:/AITrackmania/reports/headless-training-parity-20261021-r2/`.
Surowe dowody i nowe runtime na ext4:
`<USER_HOME>/tmrl-headless-parity-20261021-002/`. Poprzednie wyniki zachowano.

- Zbiorcze regresje core/distributed/V6/V7/lifecycle/clock/common: 375 passed
  w 33.93 s; całe polecenie 37.17 s, max RSS 417872 KiB, zero swap. Osobne końcowe
  testy nowych adapterów i aktualny łączny wynik są w `summary-final.json`.
- Produkcyjny adapter: wszystkie 78 akcji osobno przy N1 i N5 — 156 przejść,
  kontrola rzeczywistych POST/receipt/audit. Powtórzenie resetu, obcięcie N5 do
  pozostałego interwału, uszkodzone ramki/receipt/POST odrzucone.
- Test aktora SIGKILL po prawdziwym resecie i kroku V7: pozostałe własne procesy 0,
  błędy cleanup 0. Testy obejmują przerwanie fabryki i odłączonego potomka.
- Rzeczywisty budżet 500000 B wyczerpany po 12 zwróconych przejściach; odrzucony krok
  nie dopisał wiersza journal. Zmierzony zapis 500031 B potwierdza charakter
  aplikacyjny limitu. Sesja i potomkowie zamknięci, dane diagnostyczne zachowane.
- 64 syntetyczne przejścia, 8 terminalnych i 8 truncation, 9 aktualizacji:
  checkpoint learnera, optymalizatora, replaya i samplera działa; nie są to
  przejścia z gry ani dowód porównania treningów.
- Rzeczywisty common6 infrastructure smoke: 20 przejść, 2 truncations,
  5 aktualizacji, batch8, 1.0 s czasu doświadczenia i zero całych aktualizacji
  pozostałego kredytu (jawna reszta float64: 8.88e−16).
  Checkpoint modelu/optimizera/replaya/samplera/budżetu odtworzony poprawnie.
  Pełne polecenie z importami, resetami, zapisem i cleanup: 19.04 s,
  max RSS 375392 KiB, zero swap. Są to koszty krótkiego testu infrastruktury,
  wykonywanego obok innych kontroli, a nie kontrolowany benchmark kampanii.
  Osobny collect-only: 20 przejść, 0 modeli i aktualizacji; pełny czas wewnątrz
  programu 13.775 s. Nie przeprowadzono porównawczego treningu Windows/native.
- Ponowna kontrola zamrożonej instalacji: 466 plików źródeł i manifest zasobów
  zgodne; nie modyfikowano bazowej instalacji236/405 ani chronionych kampanii.

W historycznym zestawie r2 meta pozostawała niepotwierdzona. R5 wykazał ją przez
publiczny adapter po1780 krokach N1, ze standardowym neutralnym resetem.
Oddzielny source-repair release/drain uzyskał tę samą metę z zerowym kredytem
autonomicznych interwałów. Ścieżka EOF wymaga
oryginalnego validatora i TMRESULT, odrzuca sam błąd socketu jako sukces; częściowe
EOF i brak NEXT mają testy, ale nie zastępują próby rzeczywistej mety. Nie
zakwalifikowano wielogodzinnego treningu, CUDA, PPO ani SAC/GNN. Zachowano limit
sesji 240 s. SIGKILL samego nadzorcy wymagałby zewnętrznego cgroup/service; test
zabicia aktora go nie obejmuje. Pamięć replaya jest ograniczona pojemnością,
raw dowody rosną z liczbą kroków i wymagają jawnego budżetu.

## Historyczne uruchomienie diagnostyki common6

Pełne epizody, trwały trening i resume uruchamia się według
[instrukcji native-only r7](headless-native-training.md). Poniższe polecenia
zachowują wcześniejszy, ograniczony zakres pomiaru zegara Windows.

Polecenia poniżej są dla WSL Ubuntu. Używaj **nowych nazw katalogów**. Źródła
mogą być odczytywane z worktree Windows; venv, runtime, work i raw wyniki muszą
pozostawać na ext4. Nie używaj starego `src001` jako aktualnych źródeł.

```bash
export SRC=/mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1
export PY=<USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python
# Zweryfikowana prywatna biblioteka z oddzielnego builda, nie plik z oryginalnej instalacji.
"$PY" "$SRC/scripts/headless_provision_immediate.py" \
  --v6-config <USER_HOME>/tmrl-headless-parity-20261021-001/config.json \
  --shim <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/build/forkoracle/native-immediate-v7-001/libforkshim.so \
  --destination <USER_HOME>/tmrl-v7-NOWY \
  --duration-limit 500 --max-trace-bytes 67108864
```

Przepis odbudowania shima z lokalnego archiwum źródłowego jest w
`native_input_v7/README.md`. CLI provisioning sprawdza dokładny pin builda,
kopiuje tylko 12 przypiętych plików i tworzy osobne manifesty/config. Samo
skopiowanie runtime nie stanowi kwalifikacji innego builda.

Przykład bez uczenia, ograniczony do faktycznie dowiedzionego zakresu zegara:

```bash
"$PY" -m scripts.headless_common_live \
  --config <USER_HOME>/tmrl-v7-NOWY/config.json \
  --reference <USER_HOME>/tmrl-headless-parity-20261021-001/runtime/experiments/headless-native16-training-001/reference.json \
  --clock-proof <USER_HOME>/tmrl-headless-parity-20261021-002/native-input-v7/clock-match002.json \
  --clock-proof-sha256 13369bd4b8843585d19058989f2a5ae35b02fe6ce4b39be012678ef08ab53955 \
  --output <USER_HOME>/tmrl-v7-collect-NOWY \
  --collect-only --tape neutral --episodes 2 --decisions 10 --native-iterations 5
```

Do własnego testu infrastruktury learnera zastąp `--collect-only --tape neutral`
przez `--infrastructure-smoke`; podaj inny output. Wynik pozostaje native-only,
bez kwalifikacji porównawczej. Bez obu tych flag program wymaga pełnej bramki
`--gate FILE --gate-sha256 HASH` i odrzuci obecny niepełny parytet.
STOP sprawdzany jest przed resetem i każdym krokiem; domyślnie utwórz `STOP`
w katalogu output. Nie usuwaj STOP-ów innych zadań. Wznowienie checkpointu
odtwarza learner/replay/RNG; nie odtwarza żywego snapshotu silnika.

Powtórzenie gęstego Windows probe jest opcjonalne i nie może działać równolegle
z innym aktorem Windows. Skrypt `headless_windows_timing.py` używa lokalnych
helperów preflight/lease i konfiguracji wskazanych w jego stałych. Przy odmowie
fokusu zachowaj błąd i poproś Jakuba o aktywowanie gry. Nie obchodź tej odmowy.
Nie zmieniano SD-SAC, PR54, automatów ani profilu kampanii 7 168 000 przejść.

## Co pozostaje do kwalifikacji kampanii

Potrzebna jest kolejka komend Windows z potwierdzeniem ich identyfikatora i
rzeczywiście zastosowanego ticka. R5 potwierdził odczyt wejść w funkcji fizyki,
lecz007 ma limit256 rekordów/5s i nie implementuje kolejki ani ACK komend.
Obecny Connect
udostępnia odczyt w pętli `yield()`, a vgamepad działa według czasu hosta.
Skrócenie sleep lub stałe przesunięcie nie rozwiązuje zmiennego opóźnienia.
R5 potwierdził rzeczywistą metę przez publiczny adapter, dokładną fizykę przy
odczytanych wejściach i8000ms relacji między zachowanymi próbkami zegara.
Pozostaje kwalifikacja ciągłego przebiegu wejść niezależnego aktora Windows
i właściwych granic decyzji, a następnie krótki trening obu backendów na tym
samym modelu i budżecie doświadczenia. Nie rozszerzono produkcyjnego gate'a
common6 na podstawie częściowego śladu lub próby diagnostycznej.

Można już prowadzić ograniczone eksperymenty V7/common6 od nowych wag i
sprawdzać integrację backendu. **Czas uzyskania jakości kampanii pozostaje
nieznany.** Przepustowość kolekcji ani zmniejszenie budżetu z 8 192 000 do
7 168 000 (12.5%) nie dowodzą skrócenia czasu do tej samej jakości. Do takiej
prognozy potrzebne są porównywalne krzywe jakości względem doświadczenia i
pełnego czasu, obejmującego resety, uczenie, zapis oraz cleanup.
