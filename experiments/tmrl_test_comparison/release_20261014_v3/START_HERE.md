# Start treningów — Windows i Linux X11

Aktualna paczka: **weeks-20261014-v3**, zastępuje poprzednią paczkę v2.
Nowy launcher jest napisany w Pythonie; nie potrzebuje PowerShella ani skryptów
`.ps1`. Nie zmienia polityki Windows. Starych planów, pinów i wyników nie zmieniono.
Nie aktualizuj kodu już trenującego v2: dokończ na jego commicie i zgłoś to
koordynatorowi. Nie wznawiaj starego checkpointu na v3.

## Podział i budżet

| Osoba / identyfikator | Algorytmy, kolejno na swoim PC | Seedy |
|---|---|---|
| Jakub / `jakub` | QR, potem IQN | 17, 29, 43 każdego |
| Borys / `borys` | SAC ciągły | 17, 29, 43 |
| Kamil / `kamil` | TQC | 17, 29, 43 |
| Kuba P. / `kuba-p` | PPO persistent recovery, oryginalny updater | 17, 29, 43 |

SD-SAC pozostaje experimental i **nie wchodzi do kolejki**.
Każdy run: losowy start, **8 192 000 kroków + 30 ocen końcowych**, bez wag/replayu
z pilota. Wszystkie 15 konfiguracji mają równy budżet. Przy 20 decyzjach/s to
około 4,74 doby samej jazdy/seed; 14,22 doby dla 3 seedów, u Jakuba 28,44 doby.
Uczenie, resetowanie i zapisy wydłużą czas. Limit jednej próby: 336 h, ostatnie
600 s przeznaczone na zapis/zamknięcie. Brak automatycznego restartu/resume.

Decyzją człowieka opóźnienia ponad 100 ms **są raportowane, nie blokują same
w sobie kampanii**. Błędne lub brakujące pomiary, zły UID, niepełny checkpoint,
błędy kontrolera/telemetrii i STOP nadal blokują dalszy start. Nie ukrywaj skips.
PPO recovery ma finalne wyniki 10/10 i 9/10, mediany 51,835 i 51,980 s. Nadal
zdarzyło się utknięcie przy 79,75%; tygodnie treningu nie są gwarancją poprawy.
PPO różni się pipeline recovery od pozostałych algorytmów, co zaznaczamy w
porównaniu konfiguracji. KLguard nie jest używany. Model/GNN i nagroda bez zmian.

## 1. Wspólny kod i instalacja

Wymagane: NVIDIA/CUDA, Python 3.12, Git, uv, Trackmania2020, Openplanet i gamepad
wirtualny. Przynajmniej 100 GiB miejsca na start plus zapas/backup. Trenuj na
lokalnym dysku, poza synchronizowanym folderem chmurowym. Zapewnij zasilanie,
chłodzenie i brak snu/wylogowania. Kod w repo, nie wersja biblioteki z PyPI.

```text
git clone --branch codex/tmrl-algorithm-runs https://github.com/TrackmaniaRL/TrackmaniaRL.git
cd TrackmaniaRL
git checkout --detach HASH_Z_RELEASE_JSON
uv sync --frozen --group dev
git rev-parse HEAD
```

Zastąp HASH_Z_RELEASE_JSON dokładnym commitem z dołączonego `RELEASE.json`.
Wszyscy używają tego samego commita. ZIP zawiera instrukcje; do uruchomienia
potrzebujesz czystego checkoutu Git. Nie pulluj/nie aktualizuj zależności ani YAML-i
podczas treningu. Na Windows instalacja przypiętego vgamepad może wywołać
instalator sterownika ViGEmBus; operator świadomie przygotowuje własny system.

Sprawdź CUDA, identycznym poleceniem na obu systemach:

```text
uv run --frozen --group dev python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK CUDA')"
```

Dostępność musi być True. Nie ustawiaj `CUDA_VISIBLE_DEVICES=-1`.

## 2. Gra — Windows

Zainstaluj Trackmania i Openplanet oraz sterownik ViGEmBus.
W Plugin Manager zainstaluj podpisany, zarządzany TrackmaniaRL Connect
(`SAC_GetData`), wersja pakietu 2.4.0; włącz School Mode. Nie kopiuj luźnych `.as`.
Mapa jest w repo: `my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx`.
Wczytaj ją w edytorze i rozpocznij walidację z widocznym autem na starcie.
Nie używaj innej mapy/geometrii. UID: `oqIJ5rQDRrNwLPTh9H2p_W4tLof`.
Przy Race Validation / PRESS A BUTTON przywróć normalnie aktywny wyścig przez
Improve. Pulpit odblokowany, gra na pierwszym planie przez całą jazdę.

## 3. Gra — Linux (eksperymentalnie, wyłącznie X11)

Gra działa przez **Steam/Proton**, a Python i wirtualny gamepad natywnie na
Linuksie. WSL, headless/SSH i Wayland nie są obsługiwanymi hostami gry.
Wybierz sesję X11/Xorg przy logowaniu; samo XWayland w sesji Wayland nie wystarcza.
Nie uruchamiaj treningu jako root/sudo. Szczegóły instalacji Proton/Openplanet
są w [instrukcji integracji](../../../readme/trackmania.md).

Przykład przygotowania Ubuntu/Debian (inne dystrybucje: odpowiednie pakiety):

```bash
sudo apt install libevdev2 xdotool protontricks
sudo modprobe uinput
```

Administrator musi nadać Twojej lokalnej sesji normalny dostęp do uinput,
np. regułą udev `KERNEL=="uinput", TAG+="uaccess"` w
`/etc/udev/rules.d/50-uinput.rules`, potem odświeżyć reguły i zalogować sesję
ponownie. Nie zastępuj tego `chmod 666`, uruchamianiem trenera jako root ani
wyłączaniem zabezpieczeń. Host-check wymaga odczytu/zapisu `/dev/uinput`
(lub `/dev/input/uinput`) oraz `DISPLAY` i xdotool.

Zainstaluj Trackmania w Steam, wybierz Proton i zainstaluj Openplanet przez
Protontricks **w prefixie tej gry**. Opcja uruchomienia Steam:

```text
WINEDLLOVERRIDES="dinput8=n,b" %command%
```

Zainstaluj zarządzany plugin TrackmaniaRL Connect i School Mode jak na Windows.
Udostępnij dołączoną mapę w katalogu map tego prefixu, wczytaj ją w edytorze i
rozpocznij walidację. Biblioteka łączy się z pluginem po localhost; nie wystawiaj
portów na sieć. Sprawdź, że gra widzi wirtualny gamepad i force feedback;
konfiguracje używają rumble do kolizji. Brak rumble zgłoś, nie przedstawiaj tego
hosta jako równoważnego. Nie zmieniaj sterowania na klawiaturę.

Reset wykorzystuje xdotool: Delete, potem Return, wyłącznie przy aktywnym oknie
zatytułowanym Trackmania. Kod nie przełącza fokusu i nie wysyła klawiszy do okna
w tle. Jeśli reset nie działa w danym środowisku, zachowaj raport i zgłoś problem.
Nie dopisuj ręcznego resetu do pętli, nie modyfikuj YAML-a.

**Zweryfikowano kod i systemowe blokady, ale nie przeprowadzono treningu ani
preflightu w grze na Linux/Proton w tej sesji.** Każdy linuxowy host musi przejść
własną kalibrację i `check-game` przed wielodniowym startem. W wynikach zapisujemy
OS, Proton, GPU/sterownik i ustawienia gry; nie zakładamy pełnej równoważności OS.

## 4. W&B i lokalna kalibracja

Własny klucz W&B zapisz w prywatnym, ignorowanym `.env` w katalogu repo:
`WANDB_API_KEY=...`. Nie wysyłaj go na czat/Git ani z wynikami. Uzyskaj dostęp
do zespołu dsc-pjatk-warsaw, projektu my-trackmania-agent.

Windows PowerShell — ustaw w terminalu startowym:

```powershell
$env:WANDB_ENTITY = 'dsc-pjatk-warsaw'
```

Linux bash:

```bash
export WANDB_ENTITY=dsc-pjatk-warsaw
```

Następne polecenia są wspólne dla obu systemów:

```text
uv run --frozen --group dev python -m trackmaniarl track check
uv run --frozen --group dev python -m trackmaniarl track check-steering --output artifacts/tmrl-test-comparison/steering-local.json
```

Kalibracja rusza sterowaniem; wykonuj ją bez aktywnego treningu/oceny. Nie kopiuj
jej z innego PC. Przy zlewających się poziomach skrętu ustaw Analog Sensitivity 1,0
oraz minimalną Dead Zone i zmierz ponownie. Potem zachowaj plik i ustawienia przez
całą kampanię. YAML-e już wskazują ten lokalny plik, nie edytuj ich.

## 5. Sprawdzenie komputera i gry

Wszystko z głównego katalogu repo; zmień tylko borys na swój identyfikator.
`verify`, `jobs` i `host-check` nie tworzą kontrolera ani nie sterują grą:

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign verify
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign jobs --worker borys
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign host-check --worker borys
```

Wyślij koordynatorowi hash commita i raport host-check bez klucza. Następnie, przy
włączonej grze na pierwszym planie, sprawdź jeden reset/mapę/gotowość/telemetrię:

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign check-game --worker borys
```

`check-game` steruje resetem, ale nie uruchamia uczenia i nie zużywa run_id.
Jest pod blokadą jednego kontrolera i sprawdza STOP. Raport ma status ready,
correctUID i 33 pola. Po odmowie nie obchodź kontroli, zgłoś raport błędu.
Launcher powtarza tę kontrolę przed każdym seedem. Nie odpalaj dodatkowych
track-check/kalibracji/UI podczas treningu lub oceny.

## 6. Start kolejki — wspólne polecenia Windows/Linux

Każdy uruchamia tylko swoją linię w otwartym terminalu:

```text
# Jakub — QR, potem IQN
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign run --worker jakub --launch
# Borys — SAC
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign run --worker borys --launch
# Kamil — TQC
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign run --worker kamil --launch
# Kuba P. — PPO recovery
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261014_v3.campaign run --worker kuba-p --launch
```

Bez `--launch` polecenie run tylko wyświetla kolejkę. Nie odpalaj wszystkich
czterech linii na jednym PC. Na Windows działa wspólny named mutex, na Linux
blokada flock współdzielona między checkoutami tego użytkownika. Nie uruchamiaj
osobno actor/learner/benchmark. Po każdym seedzie 30 ocen raz, z ostatniego pełnego
checkpointu; PPO używa oceny końcowej swojego trenera. Następny seed od zera.
W tej sesji niczego z tej długiej kampanii nie uruchomiono.

## 7. Nadzór, STOP i wyniki

Sprawdzaj kilka razy dziennie: rosnące transitions/updates, świeże checkpointy/logi,
W&B, temperatury i dysk. Sam PID/GPU nie dowodzi postępu. Nie zamykaj terminala,
nie wylogowuj się, nie aktualizuj źródeł. To supervisor lokalny, nie niezależna
usługa po awarii systemu lub zabiciu procesu SIGKILL. Nie jest gwarancją działania
przez tygodnie bez opieki. Normalny SIGTERM/Ctrl+C żąda STOP i bounded save;
lepiej użyć pliku STOP. Windows zamknięcie okna może przerwać nadzór.

STOP swojego PC, np. Borysa:

```powershell
# Windows
New-Item -ItemType File -Force artifacts/tmrl-test-comparison/weeks-20261014-v3/STOP-borys
```

```bash
# Linux
touch artifacts/tmrl-test-comparison/weeks-20261014-v3/STOP-borys
```

Supervisor kontroluje STOP, dysk i własnych potomków co 1 s, daje maksymalnie 600 s
na zapis (w granicach limitu 336 h), potem zamyka tylko zapisane PID+czas utworzenia.
STOP nie jest kasowany. Błąd, brak pełnego checkpointu albo oceny nie rozpoczyna
następnego seeda; nie ma automatycznego wznowienia. Zachowaj logi/folder/STOP,
nie powtarzaj użytego run_id, zgłoś koordynatorowi. Próg startu dysku 100 GiB,
awaryjne żądanie STOP poniżej 10 GiB. Nie czekaj na wyczerpanie miejsca.

Treningi: `artifacts/tmrl-test-comparison/tmrl-test-weeks-20261014-v3-ALGO-sSEED/`.
Status, preflight i logi etapów: `artifacts/tmrl-test-comparison/weeks-20261014-v3/RUN_ID/`.
Host-check: `weeks-20261014-v3/hosts/`. Oceny off-policy w osobnych katalogach
`RUN_ID-benchmark-*`, PPO w `RUN_ID/evaluation.json`.

Oddaj wszystkie 3 seedy (Jakub 6): foldery treningu/ocen, ostatnie checkpointy+SHA256,
konfigurację/manifest/fingerprint, kalibrację, raporty hosta/preflightu/nadzoru,
hash commita i linki W&B. Podaj GPU/CPU/sterownik/OS/Proton/FPS, czas treningu,
ukończenia/30, czasy ukończonych, postępy porażek i timing/skips. Nie oddawaj .env,
kluczy i .venv. W&B nie zastępuje backupu; kopiuj ukończone zapisy na drugi dysk.
Nie wybieraj najlepszego checkpointu na tych samych 30 przejazdach i nie ukrywaj
nieudanych seedów. 0 met to brak czasu ukończenia, nie czas 0 s.

Źródło API resetu X11: [xdotool — dokumentacja projektu](https://github.com/jordansissel/xdotool/blob/master/xdotool.pod).
