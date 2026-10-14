# Porównanie algorytmów na tmrl-test

AKTUALNE14 października: człowiek jawnie zniósł blokadę za pojedyncze opóźnienia. Nowa kampania **weeks-20261014-v2** obejmuje QR/IQN/SAC/TQC oraz PPO persistent recovery. SD-SAC wyłączone. Timing nadal raportowany, próg100ms nie blokuje v2; błędy/niepełne pomiary nadal blokują. Przygotowano15świeżych runów, żadnego nie uruchomiono. [Instrukcja dla kolegów i polecenia startowe](release_20261014/START_HERE.md). Używaj wyłącznie skryptów z release_20261014; wszystkie wcześniejsze BLOCKED/ACTIVE/stare polecenia poniżej są historyczne.

Przygotowane do późniejszego uruchomienia. Każdy wygenerowany eksperyment zapisuje
lokalne JSONL/checkpointy oraz wysyła metryki do projektu W&B `my-trackmania-agent`.
To eksperyment pokazujący wymienność algorytmów biblioteki, nie warunek publikacji
artykułu ani rozstrzygający ranking algorytmów RL.

## Zakres

Aktualna kampania na czterech komputerach: pięć algorytmów, trzy seedy,
8 192 000 kroków/seed, **przygotowana i nieuruchomiona**.
[Plan, instalacja i nadzór](MULTI_COMPUTER_TRAINING.md).
SD-SAC jest **experimental**, wyłączony ze standardowych kolejek i przydziałów;
scheduler napraw jest PAUSED. [Raport i prompt dla Fable 5.1](SD_SAC_FABLE_5_1_REPORT.md).

Generator standardowy zapisuje 32 samodzielne YAML-e dla ośmiu algorytmów;
lista w `manifest.json`. Cztery zachowane konfiguracje SD-SAC mają oddzielny
`experimental-manifest.json` i wymagają osobnego zakresu badawczego.
Historyczny protokół pozostaje dostępny:

| Etap | Seedy na algorytm | Kroki na trening |
| --- | --- | --- |
| pilot każdego algorytmu | 17 | 145 408 |
| full | 17, 29, 43 | 2 048 000 |
| nowa kampania `weeks` — 5 przydzielonych algorytmów | 17, 29, 43 | 8 192 000 |

Algorytmy dyskretne (78 akcji): Q, QR, IQN, FQF. SD-SAC pozostaje eksperymentalny.
Algorytmy ciągłe (gaz, hamulec, skręt): SAC, TQC, REDQ, PPO.
Wyniki tych dwóch grup pokazujemy osobno: zmienia się też przestrzeń sterowania.

Zakończona lokalna kampania obejmowała IQN, QR, discrete SAC, TQC, PPO i SAC.
Aktualna gotowość oraz ograniczenia są w `READINESS.md`. Podział pełnych
treningów i instrukcje dla kolegów są w `HANDOFF.md`.
Pilot służy wykryciu problemów, sprawdzeniu szybkości i tego, czy pojawia się
uczenie. Brak mety po 100 tys. kroków nie dowodzi, że algorytm jest słaby.
Po pilocie ustalamy konfiguracje i uruchamiamy pełne treningi od zera.
Nie przenosimy wag ani replayu pilota do pełnych prób.

Przy 20 decyzjach/s sam czas interakcji to około 2,02 h dla każdego pilota,
28,44 h/historyczny full i 113,78 h/nowy weeks. Pięć przydzielonych algorytmów
na trzech seedach daje 15 długich treningów, nominalnie 1706,67 godzin hostów.
To przeliczenie nominalne: aktualizacje, restarty, opóźnienia i ewaluacja
mogą wydłużyć rzeczywisty czas. Nie łączymy wyników różnych budżetów w jeden ranking.

## Co jest wspólne

- Mapa `trackmaniarl-test.Map.Gbx`, UID `oqIJ5rQDRrNwLPTh9H2p_W4tLof`.
- Geometria, pipeline V5 i enkoder GNN + Simba V6 z konfiguracji V107I/V108.
  Adapter `BatchedIncidentEncoder` tylko spłaszcza i odtwarza osie batch/sekwencja
  dla PPO; nie zmienia liczby parametrów ani obliczeń pojedynczej obserwacji.
- Nagroda i warunki zakończenia z archiwalnej konfiguracji V108, limit 150 s,
  decyzja co 50 ms, gamma 0,995, brak racing line.
- Losowa inicjalizacja, wszystkie parametry uczone, bez demonstracji,
  warm-startu, filtra referencyjnego i zamrażania enkodera.
- Off-policy: uniform replay 1 mln, batch 256, 1-step, warmup 10 tys.,
  0,25 aktualizacji/krok. Q/QR/IQN/FQF: LR 1e-4, hard target co 1000 aktualizacji.
  Q/QR/IQN/FQF w asynchronicznym `train` stosują epsilon 0,3 → 0,01 przez 500 tys. kroków.
  Aktorzy SAC/TQC używają własnych rozkładów; epsilon nie steruje ich eksploracją.
  Nie stosujemy historycznej heurystyki przytrzymywania eksploracyjnych akcji.
- IQN: 64 kwantyle treningowe/target, 32 ewaluacyjne; QR/FQF: 64.
- PPO: rollout 2048, 10 epok, minibatch 256; bez dodatkowej normalizacji
  obserwacji i nagród. Nagroda oraz wejścia są już zdefiniowane przez pipeline.
- Brak wcześniejszego kończenia treningu po dobrym wyniku; równe budżety kroków.

To wspólny enkoder, nie identyczny cały model. Aktorzy i krytycy mają własne
kopie enkodera; REDQ ma 10 krytyków, SAC/TQC mają 2. Budżet aktualizacji i
domyślne parametry implementacji nie są tuningiem optymalnym dla każdego
algorytmu. REDQ zachowuje domyślny interwał aktualizacji aktora 20; przy wspólnym
UTD 0,25 nie jest to standardowy eksperyment REDQ z wysokim UTD.
PPO ma inny schemat aktualizacji i wykonania, więc porównuj też czas ścienny.
Warstwy Simba normalizują efektywne wagi przy forwardzie; learner value-based
dodatkowo projektuje przechowywane wagi po aktualizacji, a actor-critic nie.
To kolejna różnica implementacyjna, którą należy zachować w opisie eksperymentu.
Nie obiecujemy odtworzenia 36,977 s: tamten wynik pochodził z wieloetapowego
treningu i dodatkowego filtra, nie z tego protokołu.

## Uruchamianie później

Polecenia PowerShell z katalogu głównego repozytorium. Wymagają obecnego
środowiska `.venv`, CUDA, działającego OpenPlanet i vgamepad. Otwórz właściwą
mapę w edytorze; konfiguracja używa `restart_input: editor_validation`.
Pliki mapy i geometrii są wskazane względnie w `my-trackmania-agent/`.
Nie trzeba ich kopiować. Uruchamiaj tylko jeden trening sterujący grą naraz.

```powershell
# Kontrola bez gry (małe syntetyczne aktualizacje na CPU, pliki tymczasowe):
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.check

# Nowa kampania: pokazanie przydziału bez jazdy (późniejszy start opisuje plan):
./experiments/tmrl_test_comparison/run_campaign.ps1 -Worker kamil

# Pierwszy pilot — uruchomi grę przez istniejące połączenie:
.venv/Scripts/python.exe -m trackmaniarl train experiments/tmrl_test_comparison/configs/pilot/iqn-s17.yaml

# Kolejne standardowe algorytmy: qr, tqc, q, fqf, sac, redq, ppo.
# Historyczny pełny etap: katalog full, seedy 17 / 29 / 43, np.:
.venv/Scripts/python.exe -m trackmaniarl train experiments/tmrl_test_comparison/configs/full/iqn-s29.yaml
```

Artefakty trafiają do `artifacts/tmrl-test-comparison/`; CLI nadaje próbom
identyfikatory. Do powtarzalności zachowaj manifest, resolved config, snapshot
źródeł, sprzęt, logi i czas trwania każdej próby. Nowa kampania wymaga czystego
checkoutu, wspólnego commita i zgodnych przypięć źródeł/konfiguracji/mapy.
`WandbTracker` odczytuje `WANDB_API_KEY` z prywatnego `.env` w katalogu głównym;
klucz nie trafia do YAML-i, manifestu ani W&B configu.

## Ocena

Launcher `run_assigned.ps1` wykonuje dla każdego seeda trening i ocenę 30 prób.
PPO wykorzystuje istniejącą końcową ocenę trenera, bez drugich 30 prób.
Nieudane progi jakości są raportowane jako nieudane; nie unieważniają kompletnego,
poprawnego pomiaru. STOP, niepełny checkpoint lub błąd wykonania zatrzymują kolejkę.
SD-SAC jest wyłączony zarówno ze standardowych pilotów, jak i pełnych treningów.

Przy ręcznym użyciu CLI po każdym zakończonym treningu oceniaj **końcowy checkpoint**, 30 przejazdów,
bez eksploracji i bez filtra referencyjnego. Podstaw rzeczywistą ścieżkę
zapisaną przez zakończony trening:

```powershell
.venv/Scripts/python.exe -m trackmaniarl benchmark experiments/tmrl_test_comparison/configs/full/iqn-s17.yaml 'SCIEZKA_DO_KONCOWEGO_CHECKPOINTU.pt' --trials 30
```

Progi historycznego benchmarku (37 s i 100% ukończeń) pozostają ambitnym
punktem odniesienia; ich niespełnienie nie unieważnia pomiaru porównawczego.
Nie wybieraj checkpointu na podstawie tych samych 30 przejazdów, które potem
publikujesz jako wynik. Ewentualny wybór najlepszego checkpointu wymaga
osobnej walidacji i nowych prób końcowych.

Raportuj dla każdego seeda: ukończenia/30, medianę i średnią czasu ukończonych
przejazdów (jawnie jako warunkowe), liczbę niepowodzeń, osiągnięty postęp,
czas uczenia, liczbę kroków i opóźnienia sterowania. Przy zerze ukończeń czas
to brak wyniku, a nie zero. Pokaż wszystkie trzy seedy, nie tylko najlepszy.
30 przejazdów jednej sieci nie zastępuje trzech niezależnych treningów.
Krzywe nagrody/postępu z treningu są diagnostyczne, nie są czasami ewaluacji.

Po zmianie protokołu zaktualizuj `generate.py`, następnie wygeneruj YAML-e
ponownie i sprawdź je. Generator nie uruchamia treningów:

```powershell
.venv/Scripts/python.exe -m experiments.tmrl_test_comparison.generate
```
