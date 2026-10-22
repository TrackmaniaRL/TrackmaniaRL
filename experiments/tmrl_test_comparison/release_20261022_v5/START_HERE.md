# Nowe treningi zespołu — wersja 1.2.16, kampania v5

Ta instrukcja jest aktualna dla **nowych startów**. Kampania
`weeks-20261022-v5` używa biblioteki **1.2.16** i wspólnego limitu
**7 168 000 kroków na seed**, o 12,5% mniej niż wcześniejsze 8 192 000.
Nie aktualizuj rozpoczętej próby: kończy się na swoim kodzie i konfiguracji.
Stare checkpointy, plany, STOP i wyniki zachowaj. V5 nie wznawia wcześniejszych prób.

V5 zawiera wydanie 1.2.16, w tym gaz podczas odliczania na starcie oraz poprawione
zamknięcie treningu: najpierw zatrzymanie zbierania danych, potem zaległe
aktualizacje i zapis modelu. Zabezpieczenia STOP i limity czasu nadal obowiązują.
SD-SAC pozostaje wariantem eksperymentalnym poza wspólną kampanią. V4 zachowano
bez przepisywania; do nowych startów służy wyłącznie plan V5.

## Podział

| Osoba / identyfikator | Algorytmy, kolejno | Seedy każdego algorytmu |
|---|---|---|
| Jakub / `jakub` | QR, IQN | 17, 29, 43 |
| Borys / `borys` | SAC | 17, 29, 43 |
| Kamil / `kamil` | TQC | 17, 29, 43 |
| Kuba P. / `kuba-p` | PPO z persistent recovery | 17, 29, 43 |

Wszystkie 15 konfiguracji mają taki sam budżet i **30 prób końcowej oceny**.
SD-SAC pozostaje wyłączone. Zapis modeli etapowych: 2,048 / 4,096 / 6,144 /
7,168 mln kroków; rzeczywiste liczniki są zapisane w nazwie i metadanych kopii.
Modele etapowe nie wywołują dodatkowych ocen i nie służą wybieraniu najlepszego
modelu po fakcie. Oceniany jest ostatni checkpoint.

Konfiguracje jawnie przyjmują poprawiony limit projekcji prędkości 278 m/s z
wydania biblioteki, zamiast historycznych 100 m/s z v3. Pozostałe współczynniki
nagrody, gamma, n-step, sterowanie i pipeline recovery PPO zachowujemy.

Przy 20 decyzjach/s sam czas gry to około **99,56 h / 4,15 doby na seed**.
Przy zmierzonym tempie QR około 18,84 kroku/s byłoby to około **105,7 h / 4,4 doby**;
nie jest to pomiar wydajności IQN, SAC, TQC ani PPO. Dwa niezależne seedy można
uruchamiać na dwóch osobnych komputerach. Jeden komputer/pulpit ma jeden kontroler.

Nowy budżet oznacza nową kohortę. Historycznych finałów przy 8,192 mln nie traktujemy
jako równobudżetowych wyników v5. Dotyczy to także obecnego QR17 na zamrożonej
1.2.12. Aby zebrać pełne porównanie v5 z trzema seedami, trzeba zebrać trzy nowe
próby każdej metody; przygotowanie planu niczego automatycznie nie uruchamia.

## 1. Kod i środowisko

Potrzebne: Python 3.12, Git, uv, NVIDIA/CUDA, Trackmania 2020, Openplanet i lokalny
gamepad wirtualny. Biblioteka z PyPI nie zawiera tej aplikacji eksperymentalnej;
używamy wspólnego kodu z gałęzi zespołu.

```text
git clone --branch codex/tmrl-algorithm-runs https://github.com/TrackmaniaRL/TrackmaniaRL.git
cd TrackmaniaRL
git checkout --detach COMMIT_PODANY_PRZEZ_KOORDYNATORA
uv sync --frozen --group dev
git rev-parse HEAD
```

Wszyscy otrzymują ten sam commit wraz z gotowym `campaign.json`. Nie uruchamiaj
`prepare`, nie przeliczaj pinów i nie zmieniaj YAML-i. Nie aktualizuj kodu ani
zależności podczas treningu. Instalacja sterownika na Windows może wymagać
interakcji operatora; przygotuj ją przed właściwą próbą.
Przed kontrolą gry i startem launcher wymaga tagu `v1.2.16` w historii checkoutu.
Samo przygotowanie i `verify`/`jobs` działają bez gry; nie obchodzą bramki wydania.

Sprawdź akcelerator:

```text
uv run --frozen --group dev python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK CUDA')"
```

Dostępność CUDA musi wynosić `True`. Nie ustawiaj `CUDA_VISIBLE_DEVICES=-1` na
hoście treningowym. Zapewnij co najmniej 100 GiB wolnego miejsca, dodatkowy zapas
na zachowane modele, lokalny dysk poza synchronizacją chmurową, zasilanie i brak
uśpienia. Kopie etapowe zawierają również replay i stan optymalizatora.

## 2. Gra

Podpisany plugin TrackmaniaRL Connect (`SAC_GetData`), pakiet 2.4.0, School Mode.
Mapa: `my-trackmania-agent/maps/trackmaniarl-test.Map.Gbx`.
UID: `oqIJ5rQDRrNwLPTh9H2p_W4tLof`. Otwórz ją w edytorze i rozpocznij walidację,
z samochodem na starcie. Gra pozostaje na pierwszym planie, pulpit odblokowany.

Windows używa przygotowanego vgamepad/ViGEmBus. Linux jest eksperymentalny:
Steam/Proton, natywny Python, sesja X11/Xorg, xdotool i uprawnienia lokalnego
użytkownika do uinput. Wayland, WSL oraz headless/SSH nie są hostami tej kampanii.
Nie uruchamiaj trenera jako root i nie obchodź kontroli fokusu.
Szczegóły: [integracja gry](../../../readme/trackmania.md).

## 3. Weryfikacja i przydział — bez sterowania grą

W poniższych poleceniach zastąp `borys` swoim identyfikatorem z tabeli.

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261022_v5.campaign verify
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261022_v5.campaign jobs --worker borys
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261022_v5.campaign host-check --worker borys
```

## 4. Kontrola gry i start

Kontrola gry wymaga przygotowanego, wolnego hosta i wykonuje preflight:

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261022_v5.campaign check-game --worker borys
```

Po zaliczeniu kontroli operator uruchamia swoją kolejkę:

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.release_20261022_v5.campaign run --worker borys --launch
```

Bez `--launch` polecenie tylko pokazuje przydział. Każdy seed zaczyna od losowej
inicjalizacji, bez wag/replayu pilota. Limit bezpieczeństwa pozostaje 336 h, z
ostatnimi 600 s na zapis i zamknięcie. Nie ma automatycznego restartu ani resume.

## 5. Błąd, STOP i wyniki

STOP, zmiana pinów, błąd kontrolera/telemetrii, zły UID, niepełny checkpoint lub
niekompletna ocena zatrzymują dalsze uruchomienia. Zachowaj komunikat i lokalne
artefakty; zgłoś koordynatorowi. Nie kasuj STOP i nie próbuj kolejnych seedów ręcznie
po awarii. Ctrl+C żąda kontrolowanego zamknięcia — daj czas na zapis.

Opóźnienia ponad 100 ms pozostają **report-only**, zgodnie z decyzją z 14 października.
Brakujące lub nieprawidłowe pomiary nadal są błędem jakości pomiaru. Nowe metryki
rozdzielają inferencję, preprocessing, diagnostykę i obsługę danych; nie zmieniają
nagrody, akcji, n-step, gamma ani warunków akceptacji.

Artefakty są lokalnie w `artifacts/tmrl-test-comparison/`, metryki w dotychczasowym
projekcie W&B. Nie dodawaj klucza do YAML-a ani repozytorium. Finał obejmuje ostatni
checkpoint, jego tożsamość, 30 prób, jakość jazdy i osobno timing. Wynik niespełniający
progu jakości nadal opisujemy uczciwie; nie wybieramy innego checkpointu po wyniku.

[Zakres weryfikacji paczki](VALIDATION.md).
