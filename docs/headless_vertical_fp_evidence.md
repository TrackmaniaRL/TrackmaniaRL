> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Pionowa różnica mild_left: powtarzalność i zapis FP

Stan 21.10.2026: nie zmieniono progów, fizyki ani przypiętego V7. Ustalono,
że różnica Y pomiędzy zapisaną jazdą Windows i odtworzeniem jej zmierzonych
wejść nie wynika z zaobserwowanej losowości resetu native ani ze zmiany trybu
zaokrąglania native przy pierwszej rozbieżności. Przyczyna różnicy między
klientem Windows a serwerem pozostaje do rozstrzygnięcia przez jednoczesny
odczyt surowej fizyki klienta oraz jego cache API.

## Dwa świeże powtórzenia tych samych wejść

`scripts/headless_vertical_probe.py` wykonuje wyłącznie zatwierdzoną taśmę
200 interwałów mild_left z `windows-observed001/report.json`. W każdym z dwóch
nowych procesów nadrzędnych zaczyna przy native2190, jawnie wykonuje neutralny
interwał do native2200/race10, a potem odtwarza ten sam zmierzony SGB bez ID
akcji. Receipt, zapis kanałów/shadow i rzeczywiste POST są sprawdzane.

Oba przebiegi mają **200/200 native76 bitowo identycznych** z pierwotnym V7
i ze sobą. Obejmuje to cały quaternion/pozycję/prędkość, RPM, bieg, mirrory,
wejścia i filteredSteer. Nie jest to dowód identyczności wszystkich ukrytych
stanów klienta Windows i serwera. Wynik wyklucza zaobserwowaną niestabilność
tych pól pomiędzy trzema świeżymi uruchomieniami native tej samej taśmy.

Dowód: `<USER_HOME>/tmrl-headless-parity-20261021-003/vertical-replay001/report.json`.
Oba cleanup: `remainingOwned=[]`, returncode0.

## Pierwsza rozbieżność nie jest zmianą wejścia ani kontaktu w Connect33

Pierwsza różnica pozostaje przy race1630: VY różni się o
−3,725290298461914e−9, a pozycja jest wtedy identyczna. Od tego punktu w badanej
taśmie różnią się wyłącznie Y pozycji/prędkości. X/Z tych pól pozostają bitowo
zgodne. Maksymalna różnica Y pozycji to 9,5367431640625e−7 w dwóch próbkach,
VY 4,585832357406616e−6. Wynik dokładnego porównania pozostaje 162/201,
bez epsilon i bez usuwania Y.

Przed pierwszą różnicą i w jej otoczeniu SGB jest stały:
`[-0.16535432636737823, 1, 0]`, a native filteredSteer ma te same bajty.
Connect33 nie zgłasza zmiany materiałów kół (16), slip (0), skidding (0)
ani adherence (1). Same te pola nie obejmują wszystkich impulsów kontaktu,
ukrytej orientacji/rotacji klienta ani wszystkich stanów zawieszenia.

## Oddzielny V8 odczytuje zapisany kontekst FP

Eksperymentalny V8 ma nowy nagłówek TMA8/TMR8/version8 i bibliotekę SHA-256
`1e75a902928f7a02ac8df768dc9d18af4945eaeb3fcd6ad8f83dd8e14cc83cb9`,
596536 B. Nie zastępuje V7 i nie zmienia żadnego produkcyjnego pinu.
W osobnym rekordzie `TMFPU001/v1/256B` kopiuje 32 B z **zapisanego kontekstu
przerwanego silnika**, a nie bieżącego środowiska sygnałowego handlera.

Mały program C skompilowany względem nagłówków używanego glibc niezależnie
potwierdza `offsetof`:

| Pole | Offset |
| --- | --- |
| `ucontext_t.uc_mcontext.fpregs` | 224 |
| FXSAVE x87 control word | 0 |
| FXSAVE x87 status word | 2 |
| FXSAVE MXCSR | 24 |
| FXSAVE MXCSR mask | 28 |

Wszystkie kopie mają PID/TID właściciela i literalną granicę native. Decoder
sprawdza zgodność osobnych pól z dosłownymi kopiami zapisanych bajtów.
Kod nie zapisuje kontekstu FP, rejestrów FP, stanu auta ani zegara. Dotychczasowe
zapisanie właściwych kanałów wejściowych przez op6 pozostaje osobną operacją.

Powtórzono tę samą taśmę raz. **200/200 native76 nadal jest bitowo identycznych
z V7**. Dodatkowy odczyt nie zmienił badanej trajektorii. Cleanup zakończył się
z returncode0 i `remainingOwned=[]`.

### Rzeczywiście zapisane bity

We wszystkich 200 dynamicznych próbkach i początkowej granicy native2190:

| Pole | Wynik |
| --- | --- |
| MXCSR | `0x1fa5` |
| SSE rounding bits 13–14 | 0: nearest-even |
| FTZ bit15 / DAZ bit6 | oba 0 |
| SSE exception masks bits7–12 | `0x3f`: wszystkie maskowane |
| SSE exception status bits0–5 | `0x25`: invalid, divide-by-zero, precision |
| x87 control word | `0x037f` |
| x87 rounding bits10–11 | 0: nearest-even |
| x87 precision bits8–9 | 3: 64-bit significand |
| x87 status | 100×`0x0220`, 100×`0x0020` |

Status wyjątków nie jest trybem zaokrąglania. Flagi `0x25` istnieją już przed
jazdą i nie zmieniają się przy race1630; nie przypisano ich do pierwszej
różnicy Y. Zmiana C1 w x87 status również nie jest zmianą control word.
Nie wymuszano hipotetycznych ustawień klienta Windows. Samo przypuszczenie,
że CRT Windows może używać innej precyzji x87, nie jest dowodem jego stanu.

Dowody: `<USER_HOME>/tmrl-headless-parity-20261021-003/vertical-fp-v8/`
zawiera manifest pełnej kopii 179 źródeł V7, patch dwóch plików, manifest
zmodyfikowanych źródeł, receipt builda, `fp-layout.c` i `fp-layout.json`.
Build release/locked, cargo/rustc1.99.1, zakończył się w 6,67 s.
`vertical-fp-capture001/report.json` zawiera surowe zapisy i porównania V7/V8.

Decoder `scripts/headless_vertical_fp_decode.py` ma 7 testów obejmujących
bitfields RC/PC/FTZ/DAZ, sprzeczne zapisane bajty, reserved MXCSR bits oraz
niepełne lub błędnie oznaczone rekordy. Wszystkie testy przeszły.

## Co trzeba jeszcze rozdzielić

W przypiętym ELF serwera getter CSmScriptPlayer nie odczytuje bezpośrednio
vehicle physical40. Tablica dispatchera `0x3c0e18` daje:

- Position/member65 → blok `0x1194dab` → cache participant+`0xe24`;
- Velocity/member72 → blok `0x1195256` → cache participant+`0xe30`;
- oba przechodzą guard dostępności `0x121cf00` i kopiują bajty bez skalowania.

Jest to dowód **ELF Linux**, bez przenoszenia adresów na PE Windows.
Dostarczone wcześniejsze przeglądy runtime potwierdzały zgodność tego cache
z raw body przy literalnym T−10 w 34/34 i 40/40 historycznych punktach.
Nie dowodzą jednak zachowania cache klienta Windows w obecnej taśmie.

Rozstrzygający odczyt klienta powinien połączyć w tym samym znanym etapie:
raw body position/velocity, cache/API position/velocity, czas oraz zapisane
rzeczywiste sterowanie; ewentualnie również MXCSR i x87 control word.
Jeżeli raw body klienta jest zgodne z serwerem, różnica leży w ścieżce
obserwacji. Jeżeli różni się już raw body, trzeba porównać wykonany wariant
fizyki, ukryty stan i FP na konkretnym pierwszym różniącym zapisie. Obecne
dane nie uzasadniają jeszcze nazywania żadnej z tych hipotez ustaloną przyczyną.
