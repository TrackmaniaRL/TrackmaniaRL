# Kampania na czterech komputerach — przygotowana, nieuruchomiona

Aktualne nowe starty: **weeks-20261022-v5**, biblioteka **1.2.16**, **7 168 000 kroków na seed** (12,5% mniej), 30 prób końcowych. [Instrukcja i podział zespołu](release_20261022_v5/START_HERE.md). Plan obejmuje 15 nowych konfiguracji; żadnej nie uruchomiono. Start wymaga wydanego tagu v1.2.16 w historii checkoutu. Paczka zawiera gaz podczas odliczania i poprawione zamknięcie po STOP. Rozpoczęte próby zachowują swoje źródła, konfiguracje i budżety; starsze plany i wyniki pozostają oddzielne.

## Materiały historyczne

AKTUALNE: używaj **weeks-20261014-v3**, wspólnego launchera Python dla Windows/Linux X11. [Instrukcja i podział dla zespołu](release_20261014_v3/START_HERE.md). V3 zastępuje paczkę v2 przy nowych startach; stare plany/pins pozostają historyczne i wymagają własnego commita. Nie aktualizuj aktywnych treningów. Linux/Proton wymaga lokalnego check-game; Wayland/WSL/headless nie są obsługiwane. Timing raportowany bez progu, SD-SAC wyłączone.

AKTUALNE14 października: człowiek jawnie zniósł blokadę za pojedyncze opóźnienia. Nowa kampania **weeks-20261014-v2** obejmuje QR/IQN/SAC/TQC oraz PPO persistent recovery. SD-SAC wyłączone. Timing nadal raportowany, próg100ms nie blokuje v2; błędy/niepełne pomiary nadal blokują. Przygotowano15świeżych runów, żadnego nie uruchomiono. [Instrukcja dla kolegów i polecenia startowe](release_20261014/START_HERE.md). Używaj wyłącznie skryptów z release_20261014; wszystkie wcześniejsze BLOCKED/ACTIVE/stare polecenia poniżej są historyczne.

Decyzja z 8 października 2026: SD-SAC pozostaje **experimental** i wypada ze
standardowych pilotów, pełnych treningów oraz przydziału komputerów. Jego scheduler
jest wstrzymany. Aktualnym planem jest `campaign.json`; stare kolejki nie są planem
nowej kampanii. Raport naprawczy: [SD_SAC_FABLE_5_1_REPORT.md](SD_SAC_FABLE_5_1_REPORT.md).

## Przydział i budżet

| Komputer / osoba | Kolejność algorytmów | Seedy dla każdego | Nominalna jazda łącznie |
| --- | --- | --- | --- |
| Jakub (`jakub`) | QR, następnie IQN | 17, 29, 43 | 28,44 doby |
| Borys (`borys`) | ciągły SAC | 17, 29, 43 | 14,22 doby |
| Kamil (`kamil`) | TQC | 17, 29, 43 | 14,22 doby |
| Kuba P. (`kuba-p`) | PPO | 17, 29, 43 | 14,22 doby |

Przyjęty do przygotowania budżet to **8 192 000 kroków na seed**, czyli 113,78 h
samej interakcji przy 20 decyzjach/s. To założenie planistyczne na kilka tygodni,
nie wynik pomiaru wydajności tych komputerów. Całość: 15 niezależnych treningów,
122 880 000 kroków i 450 końcowych przejazdów. Uczenie, restarty epizodów,
checkpointy i ocena wydłużą czas. Na komputerze Jakuba algorytmy działają kolejno;
na różnych komputerach można trenować równolegle. Nie współdzielą replayu ani wag.

Profil `configs/weeks/` ma nowe identyfikatory `tmrl-test-weeks-20261008-v1-...`.
Historyczny `configs/full/` ma nadal 2 048 000 kroków. Wyników tych budżetów nie
łączymy w jednym rankingu. Wszystkie algorytmy nowej kampanii dostają taki sam
budżet, bez kończenia po dobrym wyniku. Dla PPO horyzont harmonogramu learning rate
wynika z nowego budżetu; to celowa różnica względem starego protokołu.

Zachowano mapę, nagrodę, limit epizodu 150 s, V5/GNN, akcje, gamma, replay i
ustawienia poszczególnych algorytmów. Nie przenosimy zmian diagnostycznych SD-SAC.
QR/IQN (78 akcji) raportujemy oddzielnie od SAC/TQC/PPO (3 sterowania ciągłe).

## Co faktycznie jest sprawdzone

Przygotowanie konfiguracji i testy oprogramowania nie potwierdzają gotowości
czterech zdalnych komputerów. Zachowane lokalne pomiary: IQN 29/30, QR 29/30,
SAC 30/30, TQC 29/30; szczegóły i ścieżki dowodów są w [READINESS.md](READINESS.md).
PPO ma starszy baseline 8/10, lecz późniejsza poprawiona próba dała tylko 3/10.
Przed jego wielodniowym startem potrzebny jest świeży ograniczony pilot na
docelowej konfiguracji i jawna decyzja na podstawie jazdy oraz timingu.
Wyniku 8/10 nie przedstawiamy jako kwalifikacji późniejszego PPO.

Każdy operator przed startem sprawdza lokalną kalibrację, połączenie, prawidłową
mapę, timing <=100 ms, zapis/odczyt checkpointu i metryki W&B. Nie kopiujemy
kalibracji sterowania z innego komputera. Nie uruchomiono tu żadnego treningu,
preflightu gry ani kalibracji.

## Instalacja i zamrożenie źródeł

Na każdym komputerze użyj osobnego, czystego checkoutu tego samego opublikowanego
commita brancha `codex/tmrl-algorithm-runs`. Nie instaluj zamiast niego biblioteki
z PyPI, nie aktualizuj kodu ani zależności w trakcie kampanii.

```powershell
git clone --branch codex/tmrl-algorithm-runs https://github.com/TrackmaniaRL/TrackmaniaRL.git
cd TrackmaniaRL
git rev-parse HEAD
uv sync --frozen --group dev
```

Zapisz hash commita; musi być identyczny u wszystkich. Wymagane: Windows,
Python 3.12, NVIDIA/CUDA, Trackmania, Openplanet i wirtualny gamepad według
[instrukcji gry](../../readme/trackmania.md). Dołączona mapa:
`my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx`, geometria w `assets/` obok.
Otwórz mapę w edytorze; restart odbywa się przez walidację edytora.
UID: `oqIJ5rQDRrNwLPTh9H2p_W4tLof`.

Każdy używa własnego prywatnego `.env` z `WANDB_API_KEY` i dostępem do projektu
`my-trackmania-agent`. Ustaw w swojej sesji `WANDB_ENTITY=dsc-pjatk-warsaw`, jeśli
to uzgodniony projekt zespołu. Kluczy ani `.env` nie dołączaj do raportów/Gita.
Nie przenoś z audytów zmiennej `CUDA_VISIBLE_DEVICES=-1`.

Bez aktywnego treningu sprawdź połączenie i zmierz **własną** krzywą sterowania:

```powershell
.venv/Scripts/python.exe -m trackmaniarl track check
.venv/Scripts/python.exe -m trackmaniarl track check-steering --output artifacts/tmrl-test-comparison/steering-local.json
```

Te dwa polecenia dotyczą gry; kalibracja porusza sterowaniem. Wykonuje je operator
na gotowym, odblokowanym pulpicie. Nie obchodzimy zabezpieczeń Windows, blokady
pulpitu ani polityki wykonywania skryptów. Gra musi być na pierwszym planie przez
całą jazdę. Uzgodnij z operatorem zasilanie/chłodzenie, sen systemu, aktualizacje
i miejsce na dysku; launcher nie zmienia ustawień systemu.

## Sprawdzenie planu bez jazdy

Poniższe polecenia nie tworzą kontrolera ani nie uruchamiają treningu:

```powershell
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.campaign verify
powershell -NoProfile -File experiments/tmrl_test_comparison/run_campaign.ps1 -Worker borys
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.campaign host-check --worker borys
```

Podstaw swoje `jakub`, `borys`, `kamil` lub `kuba-p`. `host-check` wymaga czystego
checkoutu, przypiętych źródeł/konfiguracji/mapy/zależności, CUDA, prywatnego klucza,
kalibracji i minimum 100 GiB wolnego miejsca. Zapisuje nowy raport hosta bez
sekretów. Nie zastępuje preflightu gry ani lokalnego pilota. Naruszenie przypięć
blokuje całą kampanię. Nie poprawiaj YAML-a w miejscu ani nie obchodź fingerprintu.

Na komputerze autora próby uruchomienia skryptu bez jazdy zablokowała polityka
PowerShell. Nie zmieniano jej ani nie używano obejścia. `host-check` wykrywa
`Restricted` i wymagania podpisu przed przydzieleniem nowej próby. Operator musi
mieć zatwierdzoną konfigurację wykonywania/podpisywania skryptów; nie stosuj
`ExecutionPolicy Bypass`. Sam plan można obejrzeć bez skryptu PowerShell:

```powershell
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.campaign jobs --worker borys
```

Opcjonalna kontrola biblioteki bez gry wykonuje małe syntetyczne aktualizacje na
CPU i zapis/odczyt checkpointów w katalogu tymczasowym:

```powershell
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.check
```

## Start dopiero po lokalnej gotowości

**Tych poleceń nie wykonano podczas przygotowania.** Operator uruchamia tylko
swoje polecenie, w jednej sesji, po opisanych sprawdzeniach:

```powershell
# Jakub: QR 17/29/43, następnie IQN 17/29/43
powershell -NoProfile -File experiments/tmrl_test_comparison/run_campaign.ps1 -Worker jakub -Launch
# Borys: SAC 17/29/43
powershell -NoProfile -File experiments/tmrl_test_comparison/run_campaign.ps1 -Worker borys -Launch
# Kamil: TQC 17/29/43
powershell -NoProfile -File experiments/tmrl_test_comparison/run_campaign.ps1 -Worker kamil -Launch
# Kuba P.: PPO 17/29/43, po wyjaśnieniu lokalnego pilota
powershell -NoProfile -File experiments/tmrl_test_comparison/run_campaign.ps1 -Worker kuba-p -Launch
```

Przed każdym seedem launcher sprawdza UID, gotowość i restart mapy, blokadę jednego
kontrolera, brak aktywnego treningu/oceny oraz nieistnienie katalogu nowej próby.
Start od zera, bez przenoszenia wag/replayu pilota. Treningi jednego komputera
wykonują się kolejno. Po kompletnym checkpointcie następuje ocena 30 przejazdów;
PPO używa własnej końcowej oceny trenera, bez drugiego zestawu 30.
Niepełny checkpoint, błąd, STOP lub niepoprawny pomiar zatrzymuje kolejkę.
Sam niski wynik jakości przy kompletnym, poprawnym pomiarze jest wynikiem badania,
nie powodem usuwania próby ani dobierania lepszego seeda.

## Nadzór, dysk i zatrzymanie

Supervisor w `campaign.py` sprawdza STOP i potomków co 2 s, zapisuje PID wraz
z czasem utworzenia oraz status i log launchera. Limit pojedynczej próby to
336 godzin czasu ściennego, niezależnie od tempa uczenia. STOP, wolny dysk <10 GiB
lub limit czasu żądają zakończenia przez plik STOP algorytmu. Po 600 s na zapis
supervisor kończy tylko obserwowane własne procesy i sprawdza ich zamknięcie.
Nie przechodzi dalej po wymuszonym zamknięciu. Nie usuwa STOP i nie restartuje
próby. To nadzór lokalny w procesie kolejki, **nie niezależna usługa po wyłączeniu
terminala/systemu**. Pozostaw sesję uruchomioną; nie zamykaj jej przez Ctrl+C.

Zatrzymaj bezpiecznie swój komputer, tworząc jeden plik, np.:

```powershell
New-Item -ItemType File -Path artifacts/tmrl-test-comparison/weeks-20261008-v1/STOP-borys
```

Działają też `artifacts/STOP`, `artifacts/tmrl-test-comparison/STOP`,
`artifacts/tmrl-test-comparison/weeks-20261008-v1/STOP` i `artifacts/STOP-sac`
(analogicznie `qr`, `iqn`, `tqc`, `ppo`). Supervisor przekazuje STOP do aktywnego
algorytmu. Historyczne STOP na komputerze autora pozostają zachowane; nie są
automatycznie kasowane ani kopiowane do nowych checkoutów.

Zachowujemy ostatnie trzy checkpointy i końcowy checkpoint; off-policy zapisuje
co 5000 aktualizacji, PPO co 10 aktualizacji. Replay 1 mln może znacznie zwiększać
rozmiar snapshotów. 100 GiB to minimalny próg startu, nie gwarantowany cały budżet
dysku. Sprawdzaj dzienny przyrost logów/replayu i wykonuj kopie ukończonych prób
na drugi dysk, bez usuwania aktywnych zapisów. W&B nie zastępuje lokalnego backupu.

Codziennie i po przerwie sprawdź rosnące liczniki kroków/aktualizacji, świeżość
checkpointów i logów, błędy/niefinity, timing, W&B i wolny dysk. Stagnację wyjaśnij
przed kontynuacją. Nie ustanowiono nowego zdalnego schedulera; operatorzy nadzorują
swoje komputery. Scheduler napraw SD-SAC pozostaje wstrzymany.

## Przerwa i ręczna kontynuacja

Nie ma automatycznego resume/retry. Po awarii zachowaj cały folder, logi i STOP,
sprawdź zamknięcie wszystkich procesów i integralność ostatniego checkpointu.
Wznawianie wolno rozważyć wyłącznie na identycznym kodzie, konfiguracji,
fingerprincie i z zachowaniem replayu/liczników. Nie używaj `--reset-replay`.
Ręczne wznowienie jest osobnym działaniem operatora, np.:

```powershell
.venv/Scripts/python.exe -m trackmaniarl resume experiments/tmrl_test_comparison/configs/weeks/sac-s17.yaml 'SCIEZKA_DO_ZWERYFIKOWANEGO_CHECKPOINTU.pt' --stop-file artifacts/STOP-sac
```

To polecenie omija supervisor kampanii; wymaga jawnego nadzoru operatora,
preflightu i osobnej weryfikacji końcowego checkpointu oraz oceny. Nie wykonuj go
z aktywnym STOP. Usunięcie/archiwizacja własnego STOP po diagnozie jest świadomą
decyzją operatora, nigdy automatyczną naprawą. Nie wznawiaj starych SD-SAC.

Po zakończeniu i poprawnej ocenie wszystkich wcześniejszych prób można wybrać
następny przypisany run przez `-StartAt`, np. `tmrl-test-weeks-20261008-v1-sac-s29`.
Launcher sprawdza kompletność i pomiar każdego pomijanego poprzednika; nie pominie
nieudanego, niekompletnego zapisu. Katalog istniejącej próby nigdy nie jest nadpisywany.

## Wyniki do oddania

Dla każdego seeda zachowaj końcowy checkpoint i SHA256, manifest/resolved config,
snapshot źródeł, hash commita, fingerprint, liczniki, logi JSONL, status/identyfikatory
supervisora, raport preflightu i pełną ocenę. Dołącz link W&B, GPU/CPU/sterownik,
FPS/ustawienia gry, kalibrację i wszystkie błędy/skips/timing. Oceniaj końcowy
checkpoint, bez wyboru na tych samych 30 przejazdach. Raportuj ukończenia/30,
warunkowe czasy ukończonych przejazdów, postęp porażek, czas treningu i wszystkie
trzy seedy. Przy 0 met czas ukończenia jest niedostępny, nie zerowy.

Przed wspólnym porównaniem skontroluj kompletność budżetu, finite, fingerprint,
SHA, rozliczenie aktualizacji i zamknięcie procesów. Średnich z trzech seedów nie
zastępuje 30 przejazdów jednej sieci. Wyników starych budżetów/wersji nie dopisuj
do tej kampanii. Zmiana budżetu lub kodu oznacza nowy plan i nowe identyfikatory.
