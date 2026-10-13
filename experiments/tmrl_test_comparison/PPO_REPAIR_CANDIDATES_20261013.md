# PPO: przygotowane poprawki stabilności i recovery

Status: **PREPARED_NOT_LAUNCHED**, 13 października2026. Żaden nowy trening,
rollout, benchmark ani optimizerstep nie został wykonany. Monitor pozostajePAUSED.
Zmiany są opt-in w `pilot_tools`; wspólne PPO/GNN/reward i zamrożony runtime
pozostają niezmienione. Oryginalne CP50/CP100 i wyniki są zachowane.

## 1. Kontrola końcowego KL i transakcyjne wycofanie epoki

Obecny updater wykonuje wszystkie minibatchsteps epoki, następnie sprawdza
średnią KL obliczoną z log-ratios sprzed poszczególnych kroków. Ta średnia
dotyczy kolejnych, różnych stanów modelu, a nie końcowej polityki na pełnym
rolloucie. W aktualizacjach51–100 earlystop działał50/50 razy, jednak końcowy
KL przekraczał target0,02 w43/50; średnia logowanaapproxKL0,01687, końcowa0,02540.

To uzasadnia wzmocnienie kontroli, nie dowodzi, że overshoot sam spowodował
7/10. PPO-Clip nie gwarantuje twardej granicy odległości polityk, co opisuje
[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
Kontrola KL przed dalszymi krokami występuje też w
[źródle Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html).

Przygotowano `pilot_tools/ppo_guarded.py`:

- `GuardedPrefixPPO` jest osobną, niewłączoną domyślnie klasą.
- Przed pierwszą epoką weryfikuje zgodność zachowanej polityki zachowania:
  zbyt duży lub niefinitywny początkowy KL odrzuca batch jako stale/invalid.
- Po każdej epoce mierzy empiryczny KL na CAŁYM rolloucie, bez próbkowania
  entropii i bez dodatkowego zużyciaRNG.
- Jeśli wynik przekracza TEN SAM target0,02 lub jest niefinitywny, przywraca
  wagi aktora i wartości, cały stanAdama, scalera oraz torchRNG sprzed tej epoki.
  Zostają wcześniejsze zaakceptowane epoki; dalsze epoki są zatrzymane.
- Loguje attempted/rejectedepochs, rejectedKL, optimization_skipped i finiteflag.
  Przy zeru zaakceptowanych kroków metryki średniejstraty0 mają jawny skippedflag.

Nie dodano nowej siatkiLR/entropy/clip/weights. Przywrócenie wyłącznie wag bez
momentówAdama byłoby niespójne, dlatego rollback obejmuje całą transakcję.
Koszt: kopia stanu i dodatkowe forwardy, możliwe wolniejsze uczenie wartości,
szczególnie przy częstym odrzucaniu pierwszej epoki. Granica dotyczy EMPIRYCZNEGO
KL na ostatnim rolloucie, nie całej przestrzeni stanów ani skumulowanego driftu
przez50 rolloutów. Nie gwarantuje monotonicznej poprawy jazdy.

## 2. Recovery nie powinno zapominać trwającego utknięcia

V5 ma prawidłową, przyczynową informację o prędkości i odległości od granic;
nie stwierdzono potrzeby nowej architektury ani poakcyjnych etykiet jako cech.
Natomiast V6 otwiera recoveryresidual dopiero po zgodzie dwóch sygnałów:
recovery_memory i retained_speed_deficit. Oba zanikają czasowo.

Odtworzenie rzeczywistych nieudanych śladów daje:

| Końcowa próba (indeks) | Aktywna recoverygate w ostatnich100 krokach: V5 | Kandydat |
|---|---:|---:|
| 2 | 12 | 100 |
| 4 | 9 | 100 |
| 5 | 13 | 100 |

Gałąź recovery znikała po około0,5–0,7s, mimo dalszego stania przy bandzie przez
około5s. Jest to konkretna słabość pamięci incydentu; samo jej usunięcie nie
uczy jeszcze poprawnej sekwencji cofnięcia i skrętu.

Przygotowano PPO-only `PersistentPpoRecoveryPipeline` w `pilot_tools/ppo_recovery.py`:

- Zachowuje istniejące8 cech i pełny modelGNN.
- Podtrzymuje wyłącznie wcześniej wykryty incydent, gdy prędkość<2m/s i
  minimalny geometrycznyclearance<0,15 szerokości połówki korytarza.
- Nie inicjuje recovery od samego stania, nie zna81% ani współrzędnych nawrotu.
- Utrzymuje wcześniejsze memory/deficit do wznowienia ruchu lub opuszczenia
  krawędzi; reset epizodu zawsze czyści pamięć.
- Prepared replay/batch nie aktualizuje pamięci drugi raz. Odczytywane są tylko
  bieżąca i wcześniejsza surowa telemetria, bez terminationreason/counterendpoint.

Kontrola wszystkich20 zapisanych przejazdów: wszystkie10 dobrych śladówCP50
mają ZERO zmian cech. Wszystkie ruchome klatki>=2m/s mają ZERO zmian recovery;
physics/context/track zachowują dokładną identyczność. W trzech utknięciach gate
pozostaje aktywna100/100. To kontrola semantyki/zakresu, nie dowód skutecznego
wyjazdu. Działanie na innych mapach/parkingu przy krawędzi pozostaje niezweryfikowane.

## 3. Oddzielna kontrola regresji deterministycznej

`pilot_tools/ppo_regression_gate.py` przygotowuje raport z pełnych ocen,
oddzielając poprawność pomiaru, ukończenia, czas jazdy i limitytimingu.
Wywołuje review przy mniejszej liczbie ukończeń lub medianie wolniejszej>10%.
10% to jawny PRÓG PRZEGLĄDU, nie dowód statystyczny lub nowa bramka kwalifikacji.
Na obecnych danych:7/10 oraz mediana wolniejsza21,69%, oba powodyreview.
Nadal obowiązuje strictgate8/10+timing<=100ms; nie osłabiono jej retrospektywnie.

Helper nie wybiera checkpointu, nie podmienia finału naCP50, nie uruchamiafull
ani sam nie zatrzymuje treningu. Integracja w przyszłym nadzorcy wymaga nowego
zamrożonego planu. Przed porównaniem nadal konieczne osobne sprawdzenie SHA,
fingerprintu i powiązaniaCP przez istniejące walidatory.

## Kolejność ewentualnej przyszłej oceny

Przygotowano DWIE oddzielne konfiguracje do przeglądu w
`evidence/ppo-repair-preparation-20261013/`: `kl-transaction.yaml` oraz
`persistent-recovery.yaml`. SchematRunSpec poprawny. To NIE są uruchamialne
przypięte kolejki: wymagają świeżego runtime zawierającego nowe moduły,
nowego fingerprintu/pinów/planu i autoryzacji startu. Bezresume zmienionego
pipeline/updatera naCP50 lubCP100.

Najpierw izolowany KLguard vs niezmienionybaseline, następnie osobno persistent
recovery vsbaseline. Matching freshseed17/29, identyczny budżet204800 i oceny50/100.
Bez łączenia zmian przed rozstrzygnięciem pojedynczych mechanizmów, bez siatek.
Zachować wszystkie outcomes; nie wybierać wyłącznie najlepszego checkpointu.
Oceniać ukończenia/czasy, zachowanie78–82% i odzyskanie ruchu oraz koszt guardów.
Odrzucić wariant przy pogorszeniu jazdy/niepoprawnym runtime; częste skippedupdates
to problem uczenia, nie sukces samej kontroliKL. Parametrów nie dobierać tylko
do tych trzech zapisanych porażek. Formalny wybór checkpointu i niezależna
kwalifikacja muszą używać oddzielnych prób; full pozostajeBLOCKED.

## Walidacja bez treningu

14 testówPASS: pełnyrollbackmodel/Adam/scaler/RNG, zachowanie zaakceptowanej
epoki, odrzucenieovershoot/NaN/Inf, wyjątek, poprawny fullbatchKL bezRNG,
persistentmemory/resetwarunków, niezależność quality/timing i wykrywanie regresji.
Testy mutują sztuczne stany kontrolne; NIE wykonują rzeczywistych optimizersteps.
RuffPASS, mypy(follow-imports silent)PASS. Kontrola20historycznychśladów oraz
SHA obucheckpointów i351sourcepinsPASS. Nowych procesów gry/uczenia nie uruchomiono.
Pierwsza seria testów raportu brakowała pól telemetry/controllererror w fixture;
poprawionofixture i dodano odrzucanie brakujących pól, finalnie14/14PASS.

Dowody: `repair-candidates-offline.json`, `verification.json`,
`repair-proposal.json` w `evidence/ppo-repair-preparation-20261013/`.
