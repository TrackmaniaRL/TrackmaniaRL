> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# V7: wejście bieżącego interwału i wyrównanie zegara, 21.10.2026

**Status: kwalifikacja ograniczonych eksperymentów. Porównanie szybkości uczenia
z Windows nadal nie jest zakwalifikowane.** Usunięto wymuszone przez protokół
V6 opóźnienie jednego rekordu. Potwierdzono wspólny zegar próbek fizycznych
na badanej mapie i neutralnym starcie. Kontroler Windows nadal stosuje wejścia
z opóźnieniem zależnym od przebiegu, a jego efektywny skręt różni się od
nominalnej wartości ID. Odtworzenie rzeczywiście zmierzonego przebiegu wejść
jest osobnym testem diagnostycznym, nie wspólnym kontraktem treningowym.

## Dowód miejsca zatrzymania

W dostarczonym źródle V6 `post_observe/reader.rs` breakpoint ma adres względny
ELF VA `0x119f163`. Zatrzymuje oryginalną instrukcję `MOV ECX,ESI`; dopiero
potem wykonują się przesunięcie indeksu o 5 bitów oraz dwa `MOVUPS` odczytujące
32-bajtowy rekord. `native_ipc::at_reader` działa przed emulacją tej instrukcji.
Zakaz `k<=p` w V6 był więc ograniczeniem protokołu; dane `p` nie zostały jeszcze
załadowane przez te instrukcje. V7 wprowadza osobny op6 i wersję ramek, zamiast
osłabiać walidację istniejącego V6.

Reverse Engineering Anything z Ghidrą 12.1.4 niezależnie zdekompilował reader
`0x119f0f0`: indeks dzieli zapytanie czasowe przez 10, a odczyty wybranego
rekordu poprzedzają kopiowanie do wyniku. Dowód
`ev_341b3a6963cfc1110c10b0190ae50623ec1e807c0459ae6ece26375505936b8a`
dotyczy jawnie zmapowanego wycinka 1024 B:

| Pole | Wartość |
| --- | --- |
| Oryginalny ELF SHA-256 | `3b6107824feea402bd32c888e994dcd6df4a10afc8676a2647ac9560ccf0d779` |
| VA początku readera | `0x119f0f0` |
| Offset plikowy początku readera | `0x119e0f0` |
| SHA-256 wycinka | `6878c578692178d6bda55c9812b86eb4a7159dc36ab039faff9d22741e4f8c5d` |

Pliki: `<USER_HOME>/tmrl-headless-parity-20261021-002/rea-reader003/`:
`slice-provenance.json`, `rea-evidence.json`. Analiza tego wycinka nie obejmuje
całego grafu wywołań ani zewnętrznych celów. Wcześniejszy import całego ELF
z timeoutem zachowano osobno; nie stanowi wyniku udanej analizy.

Nowy shim ma 594648 B i SHA-256
`fb041afb982e7cc7a5922d49c3dd0ed49c2e6ae698dd0fd103415fbeddf98004`.
Źródłowy patch, pełny manifest 179 importowanych plików i receipt builda są
w `trackmaniarl/trackmania/native_headless/_runtime/native_input_v7/`.
Oryginalna biblioteka V6 nadal ma SHA-256
`a14837a1c64595a697ddea89637e4f12a904df7ddaa84d814b4fc2cd47e97781`
w zamrożonej instalacji i poprzednim środowisku developerskim.

## Rzeczywiste wejście, reset i faza

`probe003/report.json` sprawdza osobno wszystkie 78 ID przy N1 i N5:
156 decyzji, 468 faktycznych POST. Dla każdej decyzji zgodne są bieżący rekord,
range, shadow steer, receipt i wszystkie zastosowane kanały POST. Nie ma flush
ani nieuwzględnionego interwału w kroku. Próba zaczyna się przy literalnym
native2190; nie wykonuje pięciointerwałowego gazowego prelude.

Wzór z odczytanych argumentów i instrukcji:

```text
query = next_native - (start_getter - 10) - (archive_start_offset + 10)
p = query / 10
archive_input_label_ms = p * 10 + archive_start_offset
```

Caller `0x1219f89` przekazuje czas następnej granicy, wynik getter-a
`0x121cf50` i `context+0x20e8`. Zapisane readery dają start_getter2200,
archive_start_offset−1510. Przy granicy2190 reader ma p151, etykietę wejścia0
i następną granicę2200. Nie oznacza to, że próbka fizyki2200 ma czas wyścigu0.

Wyrównanie próbek przez sześć rzeczywistych pól pozycji/prędkości, dokładnie
bit po bicie float32, niezależnie daje:

```text
Windows race_ms = NativeSimClock - 2190
```

| Źródło Windows | Unikalne zgodne próbki | Dopasowanie / odłożona walidacja | Zakres race_ms |
| --- | --- | --- | --- |
| poprzedni neutralny probe | 11 | 6 / 5 | 10…510, zakres 500 |
| nowy gęsty pomiar neutralny | 65 | 33 / 32 | 10…650, zakres 640 |

Powtórzone pomiary tego samego czasu są łączone wyłącznie przy zgodnej fizyce;
różne stany przy jednym czasie powodują odmowę. Dopasowanie wymaga co najmniej
sześciu unikalnych stanów, trzech próbek odłożonych i zgodnego przyrostu czasu.
Nie opiera się na nieruchomym aucie, jednym trafieniu ani zadanym przesunięciu.

Każdy dopasowany reader potwierdza: etykieta następnego wejścia równa się
czasowi **bieżącej** próbki fizycznej Windows, a następny POST jest o 10 ms
późniejszy. Przykład: native2190 ↔ race0; p151/label0 steruje przejściem do
native2200 ↔ race10. Aby porównać z resetem Windows race10 lub race20,
diagnostyka jawnie wykonuje odpowiednio jeden lub dwa neutralne interwały
administracyjne od race0 i zapisuje je poza decyzjami polityki.

To kwalifikacja względnego zegara i fazy próbek tej mapy/resetu; nie kalibracja
czasu ściennego, wszystkich map ani dowód zgodności RPM, kół i pełnego stanu.
RPM nie należy do sześciu porównywanych pól i różni się przy wczesnym starcie.

## Te same nominalne ID nadal nie dają tych samych trajektorii Windows

Oba zapisane zestawy pięciu taśm odtworzono z jednakowego stanu początkowego,
w rzeczywistych czasach próbek. Każdy krok bierze własną zmierzoną różnicę
zegara. Interwały 60/70 ms są dzielone na transportowe porcje do N5; podział
nie powtarza tapu i nie tworzy nowej decyzji polityki.

| Taśma | Stary Windows: zgodne próbki / wszystkie | Nowy Windows: zgodne / wszystkie | Nowa odległość końcowych pozycji |
| --- | --- | --- | --- |
| neutral | 11/11 | 11/11 | 0 |
| straight_gas | 1/41 | 1/41 | 0,5687263445 |
| mild_left | 1/41 | 1/41 | 1,4125671892 |
| full_brake | 1/41 | 1/41 | 0,6640020129 |
| brake_tap | 1/41 | 1/41 | 2,3325812577 |

Jedyną zgodną próbką czterech nominalnych jazd jest stan początkowy; pierwsza
różnica występuje po pierwszej decyzji. Pełne końce nowej próby:

| Taśma / race_ms | Native XYZ | Windows XYZ |
| --- | --- | --- |
| neutral / 510 | 495,999481201; 10,003152847; 624 | identyczne |
| straight_gas / 2020 | 496; 10,008690834; 655,369934082 | 496; 10,007761955; 654,801208496 |
| mild_left / 2010 | 498,692687988; 10,010752678; 654,860290527 | 498,490417480; 10,010563850; 653,462280273 |
| full_brake / 2010 | 495,999633789; 10,011874199; 643,809875488 | 496; 10,012644768; 643,145874023 |
| brake_tap / 2020 | 496; 10,008289337; 652,356018066 | 496; 10,010085106; 650,023437500 |

Gęsty pomiar ujawnia opóźnienie pierwszego gazu: API gas pojawia się przy
race30, 60, 50 i 70 dla kolejnych jazd; efekt prędkości pojawia się przy
race40, 70, 60 i 80. Efektywny skręt Windows wynosi −0,16535432636737823,
podczas gdy nominalne ID33 daje −0,1666666716337204. Te pomiary nie opisują
jednego stałego opóźnienia, które można bez dowodu odjąć od wszystkich kroków.

## Diagnostyka odtworzenia rzeczywiście zmierzonych wejść

Osobna ścieżka `--observed` zapisuje dowolny skończony, zmierzony SGB przez
op6/N1. Nie wytwarza ID akcji. Każdy zapis wiąże SHA-256 oryginalnego pakietu
33×float32, indeks decyzji/pakietu, czas wyścigu i dokładne bajty SGB. POST
potwierdza zastosowanie tych samych bajtów; audit potwierdza wyłączną zmianę
kanałów bieżącego rekordu i shadow steer.

Zmierzony SGB z próbki raceT steruje interwałem T→T+10. Brak ticka przerywa
pełny prefiks; wartości pomiędzy pakietami nie są interpolowane. Wyniki:

| Taśma | Dokładne sześć pól / porównane próbki | Zakres | Koniec: odległość pozycji |
| --- | --- | --- | --- |
| neutral | 51/51 | race10…510 | 0 |
| straight_gas | 190/190 | race10…1900 | 0 |
| mild_left | 162/201 | race10…2010 | 0 |
| full_brake | 201/201 | race10…2010 | 0 |
| brake_tap | 201/201 | race20…2020 | 0 |

W straight_gas brakuje pakietu race1910; dalsza część pozostała nieodtworzona.
W mild_left pierwsza różnica przy race1630 dotyczy VY o −3,725290298e−9.
Maksymalna różnica Y pozycji wynosi 9,536743164e−7 w dwóch próbkach, a VY
4,585832357e−6. X/Z pozycji i prędkości są dokładnie zgodne we wszystkich
próbkach tej taśmy. Zmienionych progów nie użyto: wynik pozostaje 162/201,
a pełny globalny parytet jest false. Łącznie zgodne jest 805/844 próbek.

Wynik izoluje duże wcześniejsze różnice do harmonogramu kontrolera Windows
i jego efektywnego skrętu. Nie zamienia odtworzenia zmierzonych wejść w
sterowanie dostępnymi wcześniej standardowymi ID ani wynik porównania uczenia.

## Artefakty, odtworzenie i cleanup

Prywatny katalog dowodów:
`<USER_HOME>/tmrl-headless-parity-20261021-002/native-input-v7/`:

- `probe003/report.json`: wszystkie ID N1/N5, receipts, audity, POST;
- `neutral-clock001/report.json`: 66 kolejnych neutralnych stanów native;
- `clock-match002.json`, `clock-match-fresh001.json`: faza i odłożona walidacja;
- `windows-replay001/report.json`, `windows-replay002/report.json`: nominalne ID;
- `windows-observed001/report.json`: wyłącznie zmierzone przebiegi wejść;
- `build-receipt.json`, `original-source-manifest.json`, `v6-to-v7.patch`.

Źródła Windows: poprzedni
`F:/AITrackmania/reports/headless-training-parity-20261021/windows-probe.json`
oraz nowy
`F:/AITrackmania/reports/headless-training-parity-20261021-r2/windows-timing002/windows-timing.json`.
Hash każdego wejściowego raportu jest częścią wyniku. Próby nie nadpisują
wyników, nie zapisują wag i nie uruchamiają uczenia.

Każda udana sesja kończy się odrębnym raportem supervisora: returncode0,
remainingOwned[]. Raw RX jest utrwalany przed dekodowaniem. Nie pozostały własne
procesy tych prób. Nie jest to gwarancja wielogodzinnej kampanii ani rezultat
testu synchronizacji klienta Windows.

`headless_clock_alignment.py` ma 11 testów obejmujących niejednoznaczność,
regresję zegara, pojedyncze trafienie, odłożoną walidację i fazę readera.
Zmierzony adapter wejść jest wyłącznie narzędziem diagnostycznym. Przed
porównaniem jakości/czasu uczenia potrzebny jest wspólny kontrakt aplikacji
wejścia, czasu decyzji i używanych obserwacji; obecny plugin Connect zapisuje
telemetrię w pętli z `yield()`, bez atomowego sterowania krokiem symulacji.
