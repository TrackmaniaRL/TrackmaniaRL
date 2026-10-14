# Instrukcja dla zespołu — treningi od 14 października 2026

Ta instrukcja dotyczy wyłącznie nowej kampanii **weeks-20261014-v2**.
Nie uruchamiaj starych skryptów z katalogu nadrzędnego ani konfiguracji `full/`.
Koordynator przygotował plany; nie uruchomił tych długich treningów.

## Kto co trenuje

| Osoba / identyfikator | Zadanie, kolejno na swoim komputerze | Seedy |
|---|---|---|
| Jakub / `jakub` | QR, potem IQN | 17,29,43 każdego |
| Borys / `borys` | SAC ciągły | 17,29,43 |
| Kamil / `kamil` | TQC | 17,29,43 |
| Kuba P. / `kuba-p` | PPO z persistent recovery, oryginalny updater | 17,29,43 |

**SD-SAC nie trenujemy**: experimental, wyłączony z kolejki.
Każdy seed zaczyna od zera; nie kopiuj wag ani replayu z pilotów.
Jeden komputer = jeden kontroler gry, algorytmy i seedy uruchamiają się kolejno.
Komputery nie łączą wag/replayu; wspólne są protokół i projekt W&B.

Budżet każdego treningu: **8 192 000 kroków**, potem **30 ocen** końcowego
checkpointu. To około 4,74 doby samej jazdy na seed przy 20 decyzjach/s;
3 seedy około 14,22 doby, u Jakuba 6 seedów około 28,44 doby. Uczenie, resetowanie,
zapisy i oceny wydłużają ten czas. Limit jednej próby:336 h, bez automatycznego
restartu ani wznowienia. Nie kończ wcześniej po dobrym wyniku i nie dobieraj seeda.

## Co zmienia ta wersja

Człowiek zdecydował14 października, że przekroczenie100 ms samo nie blokuje kampanii.
We wszystkich15 konfiguracjach `max_step_race_time_ms: null`; timing i skips nadal
są mierzone i muszą trafić do wyników. Brak pomiarów, błędy sterownika/telemetrii,
niepełna ocena, zła mapa lub checkpoint nadal zatrzymują kolejkę.
Historyczne strictFAIL pozostają zapisane; nie zmieniamy starych wyników.

PPO dostało wyłącznie `PersistentPpoRecoveryPipeline`, przetestowany na dwóch
świeżych seedach: final 10/10, mediana 51,835 s oraz9/10, mediana 51,980 s. Oryginalny updater,
model/GNN, nagroda i pozostałe hiperparametry pozostają. WariantKL odrzucony.
Jeden przejazd seed29 zatrzymał się przy 79,75%; nie gwarantujemy bezbłędnej jazdy
ani braku regresji po tygodniach. PPO ma odmienną cechę recovery od pozostałych
algorytmów — porównanie opisuj jako konfiguracje systemów, nie izolowany efekt
samego algorytmu. Stary plan v1/baseV5 nie jest nowym planem dla PPO.

## 1. Źródła i środowisko

Wymagane: Windows, NVIDIA/CUDA, Python 3.12, Git, uv, Trackmania2020,
Openplanet i działający wirtualny gamepad ViGEmBus/vgamepad.
Minimum 100 GiB wolnego miejsca na start; przy checkpointach/replayu zaplanuj
zapas lub drugi dysk. Nie używaj współdzielonego katalogu chmurowego do aktywnych
zapisów. Zasilanie/chłodzenie/sen i aktualizacje ustaw tak, by komputer mógł
pracować bez przerw. Gra potrzebuje odblokowanej interaktywnej sesji.

Pobierz branch **codex/tmrl-algorithm-runs**:

```powershell
git clone --branch codex/tmrl-algorithm-runs https://github.com/TrackmaniaRL/TrackmaniaRL.git
cd TrackmaniaRL
git rev-parse HEAD
uv sync --frozen --group dev
```

Koordynator przekazuje z paczką `RELEASE.json` z dokładnym commitem. Każdy musi
mieć ten sam commit. Jeśli branch zdążył się zmienić, w nowym checkoutcie wykonaj
`git checkout --detach HASH_Z_RELEASE_JSON` przed instalacją i ponownie sprawdź
`git rev-parse HEAD`. Po starcie nie pulluj, nie aktualizuj zależności ani YAML-i.
ZIP źródeł służy przekazaniu/archiwum; uruchamiaj z czystego checkoutu Git,
ponieważ host-check wymaga Gita i czystego stanu.

Instalacja przypiętego vgamepad może uruchomić instalator sterownika ViGEmBus.
Wykonaj ją jako świadome przygotowanie swojego komputera; sprawdź działanie
sterownika przed treningiem. Nie instaluj zamiast repo wersji TrackmaniaRL z PyPI.

Sprawdź CUDA:

```powershell
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK CUDA')"
```

Wynik dostępności ma być `True`. Nie ustawiaj `CUDA_VISIBLE_DEVICES=-1`.

## 2. Gra i mapa

W Openplanet Plugin Manager zainstaluj zarządzany, podpisany **TrackmaniaRL Connect**
(`SAC_GetData`), wersja protokołu projektu 2.4.0; włącz School Mode.
Nie kopiuj luźnych skryptów`.as`, nie obchodź podpisów. [Opis integracji](../../../readme/trackmania.md).

Mapa jest już w repo: `my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx`.
Wczytaj ją ręcznie w **edytorze**, rozpocznij walidację wyścigu, z autem na starcie.
OczekiwanyUID: `oqIJ5rQDRrNwLPTh9H2p_W4tLof`. Korzystamy z dołączonej geometrii;
nie nagrywaj innej mapy ani własnej geometrii. Gra ma być na pierwszym planie,
bez menu pauzy/końca przejazdu; podczas jazdy nie używaj komputera do innych zadań.
Jeśli pojawi się Race Validation / PRESS A BUTTON, przywróć normalnie aktywny
wyścig przez menu Improve. Nie uruchamiaj ponownie nieudanej próby bez diagnozy.

## 3. W&B oraz kalibracja tego komputera

Każdy ma własny klucz W&B i dostęp do zespołu`dsc-pjatk-warsaw`, projektu
`my-trackmania-agent`. Stwórz lokalny, ignorowany `.env` w głównym katalogu repo
z własnym `WANDB_API_KEY=...`. Klucza nie wysyłaj na czat/Git ani w paczce wyników.
W terminalu, z którego będziesz uruchamiać trening:

```powershell
$env:WANDB_ENTITY = 'dsc-pjatk-warsaw'
.venv/Scripts/python.exe -m trackmaniarl track check
.venv/Scripts/python.exe -m trackmaniarl track check-steering --output artifacts/tmrl-test-comparison/steering-local.json
```

Sprawdzanie sterowania rusza gamepadem — wykonuj bez aktywnego treningu.
Nie kopiuj kalibracji z innego PC. Jeśli poziomy skrętu się zlewają, ustaw w grze
Analog Sensitivity 1,0 i minimalną Dead Zone, powtórz kalibrację przed pierwszym
treningiem; potem zachowaj ustawienia i plik przez kampanię. Każdy YAML już
wskazuje na ten lokalny, ignorowany plik; nie edytuj YAML-a.

## 4. PowerShell i sprawdzenie bez treningu

Skrypty używają `powershell.exe` (Windows PowerShell 5.1). Sama instalacja PS7 nie
zmienia polityki tej powłoki. Sprawdź:

```powershell
powershell.exe -NoProfile -Command "Get-ExecutionPolicy -List"
```

`Restricted` blokuje start. Na własnym, niezarządzanym komputerze operator może
po przejrzeniu pobranego kodu ustawić standardową politykę dla swojego konta:

```powershell
powershell.exe -NoProfile -Command "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
```

Nie używaj `Bypass`/`Unrestricted` jako obejścia. Gdy politykę narzuca organizacja
lub wymagany jest podpis, zgłoś do administratora/koordynatora; nie omijaj jej.

W kolejnych poleceniach zmień tylko `borys` na własny identyfikator z tabeli:

```powershell
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.release_20261014.campaign verify
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.release_20261014.campaign jobs --worker borys
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.release_20261014.campaign host-check --worker borys
powershell.exe -NoProfile -File experiments/tmrl_test_comparison/release_20261014/run_campaign.ps1 -Worker borys
```

Bez `-Launch` ostatnie polecenie tylko pokazuje kolejkę. Host-check sprawdza źródła,
konfiguracje, CUDA, klucz, kalibrację, politykę skryptów i dysk; zapisuje raport
hosta. Nie zastępuje gotowości gry. Wyślij koordynatorowi ten raport oraz hash
commita przed pierwszym startem; nie trzeba udostępniać klucza. Kontrolę mapy,
33 pól telemetrii i gotowego gracza wykona też launcher przed każdym seedem.
Nie wykonuj dodatkowych track-check/kalibracji/UI podczas treningu lub oceny.

## 5. Start — każdy uruchamia tylko swoją linię

Z głównego katalogu repo, w terminalu pozostawionym otwartym:

```powershell
# Jakub: QR, potem IQN
powershell.exe -NoProfile -File experiments/tmrl_test_comparison/release_20261014/run_campaign.ps1 -Worker jakub -Launch
# Borys: SAC
powershell.exe -NoProfile -File experiments/tmrl_test_comparison/release_20261014/run_campaign.ps1 -Worker borys -Launch
# Kamil: TQC
powershell.exe -NoProfile -File experiments/tmrl_test_comparison/release_20261014/run_campaign.ps1 -Worker kamil -Launch
# Kuba P.: PPO recovery
powershell.exe -NoProfile -File experiments/tmrl_test_comparison/release_20261014/run_campaign.ps1 -Worker kuba-p -Launch
```

Nie uruchamiaj wszystkich czterech poleceń na jednym komputerze. Nie startuj
osobno learnera/actora/benchmarku. Kolejka zrobi 30 ocen raz po pełnym seedzie;
PPO wykorzystuje własną ocenę końcową. Następny seed zaczyna od zera.
To autoryzowane ręczne uruchomienie przez operatora; przygotowanie paczki niczego
nie uruchomiło. Pełne budżety na pozostałych PC nie zostały jeszcze zweryfikowane.

## 6. Kontrola i STOP

Sprawdzaj kilka razy dziennie oraz po przerwie: rosnące transitions/updates,
świeże logi/checkpointy, W&B, temperatury, wolny dysk. Sam PID lub wykorzystanie
GPU nie jest dowodem postępu. Link W&B skopiuj koordynatorowi po pierwszych metrykach.

Wyniki: `artifacts/tmrl-test-comparison/tmrl-test-weeks-20261014-v2-ALGO-sSEED/`.
Log/status nadzoru: `artifacts/tmrl-test-comparison/weeks-20261014-v2/RUN_ID/`.
Oceny off-policy mają osobne katalogi`RUN_ID-benchmark-*`; PPO`evaluation.json`
w katalogu treningu. Raporty hosta:`weeks-20261014-v2/hosts/`.

Zatrzymanie swojego komputera (podstaw własną osobę):

```powershell
New-Item -ItemType File -Force -Path artifacts/tmrl-test-comparison/weeks-20261014-v2/STOP-borys
```

Nie zamykaj terminala/Ctrl+C podczas zapisu. Supervisor sprawdza STOP/dysk co 2 s,
po żądaniu daje 600 s na zapis, zamyka wyłącznie własne procesy i nie idzie dalej.
To nadzór w działającym procesie, nie niezależna usługa po zamknięciu terminala,
wylogowaniu lub awarii systemu. Próg awaryjny dysku 10 GiB; nie czekaj na jego
osiągnięcie. Po błędzie zachowaj logi, STOP i cały folder. Nie usuwaj automatycznie
STOP, nie retry/resume, nie podmieniaj konfiguracji/fingerprintu. Zgłoś problem.

## 7. Co oddać

Dla wszystkich 3 seedów (u Jakuba 6): końcowe checkpointy, cały folder treningu,
oceny/pełne 30 prób, logi, konfigurację/manifest/fingerprint, SHA256, raport hosta,
status nadzoru, kalibrację, hash commita i linki W&B. Podaj GPU/CPU/sterownik,
ustawienia gry/FPS, czas treningu, ukończenia/30, czasy ukończonych, postępy
porażek i opóźnienia/skips. Nie wrzucaj `.env`, kluczy ani całej`.venv`.
W&B nie zastępuje lokalnej kopii; kopie rób na drugi dysk z ukończonych zapisów.
Nie wybieraj najlepszego checkpointu na tych samych 30 przejazdach i nie ukrywaj
nieudanych seedów. Wynik 0 met oznacza brak czasu ukończenia, nie czas 0 s.
