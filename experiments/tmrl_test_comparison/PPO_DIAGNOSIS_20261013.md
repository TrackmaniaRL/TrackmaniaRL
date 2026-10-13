# PPO: diagnoza 3/10 i warunki przed długim treningiem

Badany run: `tmrl-ppo-p13-prefix-r3-s17`, W&B `brczyy2x`, frozen source
`2258d5c7ee60c9a91f2517a0778ae960431ecedd`. Trening i ocena zakończone normalnie.
To diagnoza istniejących danych, **bez nowego treningu, jazdy lub aktualizacji wag**.

## Co spowodowało odrzucenie pilota

Wszystkie siedem nieukończeń to utrata postępu, skupiona w trzech fragmentach:

| Fragment trasy | Próby (indeksy od 0) | Końcowy postęp | Zakończenie |
| --- | --- | --- | --- |
| Początek | 4, 9 | 5,028%; 4,981% | slow_progress; no_progress |
| Środek | 1, 8 | 40,867%; 40,867% | no_progress |
| Końcówka | 0, 2, 6 | 81,181%; 81,181%; 81,134% | no_progress |

Trzy ukończenia: 56,60 / 55,25 / 55,51 s. Powtarzalność miejsc nieukończeń
jest dowodem niedostatecznej niezawodności zapisanej polityki w tych fragmentach.
Nie wystarcza do wskazania konkretnego błędu sterowania: ewaluacja zachowała
agregaty i wyniki, ale nie pełne sekwencje obserwacji/akcji/pozycji krok po kroku.
Nie udowodniono, czy polityka źle skręca, źle hamuje, zbyt późno reaguje,
czy nie potrafi wyjść z incydentu. Nie nazywamy tego awarią Windows lub PPO API.

Runtime: prawidłowe UID 10/10, 7756 poprawnych pomiarów, max odstęp decyzji50 ms,
brak błędów kontrolera/telemetrii. Skips12899/max7 są ujawnione, a ich liczba nie
jest liczbą utraconych decyzji50 ms: producent wysyła częściej niż działają decyzje.
Poprawne SHA, fingerprint, finite, pełne145408 kroków/71 aktualizacji i closure.

## Jak przebiegało uczenie

Pierwsza meta wystąpiła dopiero w rolloucie42, czyli w przedziale kroków
83969–86016. Łącznie zachowano 21 met, 23 no_progress, 23 slow_progress
oraz **71 osobnych rollout_boundary**. Nie wolno traktować tych ostatnich
jak 71 błędów jazdy ani pomijać ich przy interpretacji skuteczności.

Ostatnie11 rolloutów (61–71) zawiera10 met, 2 naturalne nieukończenia i11
przerw na granicy rolloutu. Nie raportujemy tego jako „83% skuteczności”:
10/12 liczyłoby wyłącznie naturalne zakończenia i ignorowało11 uciętych wyścigów.
Czasy ostatnich met treningowych maleją od101,79 do74,45 s. W ocenie polityka
kończy w około55,8 s, ale tylko3/10. Trening losuje akcje z Gaussa; ocena używa
`tanh(latent_mean)`. To różne sposoby działania tej samej polityki, nie test
przyczynowy wpływu losowości. Hipoteza zbyt agresywnej jazdy bez losowości wymaga
osobnego kontrolowanego pomiaru i nie jest jeszcze potwierdzona.

W późnych nieukończeniach treningowych widać także problem odzyskiwania jazdy:
ostatnie100 kroków rolloutu64 przy38,33% trasy miało średni gaz0,839,
hamulec0,305 i prędkość wzdłuż trasy−0,0076 m/s, przy odległości od geometrii
około10,74 m. Nie jest to dowód komendy kontrolera równej zeru. To jeden
rzeczywisty przykład utknięcia mimo wydawania akcji; nie odtwarza oceny trial1/8.

Pilot to **1,775%** budżetu kampanii8192000. Krzywa treningu nie uzasadnia tezy,
że PPO już przestało się uczyć; nie uzasadnia też automatycznego uruchomienia
kilkutygodniowej kampanii. Pozostało tylko29 aktualizacji od pierwszej mety.

Reset co2048 kroków przerywa wyścig po około102,4 s. Osiem takich przerw
wystąpiło już po80% trasy, w tym98,89% i99,58%. To ograniczenie zbierania danych,
nie błąd bootstrappingu: rollout_boundary pozostaje truncation z bootstrapem.
Kontynuacja wyścigu przez granicę rolloutu jest kandydatem do osobnej kontroli;
nie jest poprawką wyjaśniającą bezpośrednio wszystkie porażki krótszej oceny.

## Stabilność aktualizacji: sygnały, których nie należy nadinterpretować

- Limit KL zakończył optymalizację wcześniej w63/71 aktualizacjach. Od aktualizacji21
  dotyczy to każdej. Średni KL ostatnich bloków około0,016, clipping około0,15.
  Logowane KL jest średnią kolejnych minibatchy/epok, nie końcowym KL całego batcha;
  wynik poniżej0,02 nie przeczy zadziałaniu ograniczenia w ostatniej epoce.
- Loss value wzrósł z0,121 w aktualizacji1 do2,940 w71. Pojawiły się nowe stany
  i nagrody za metę, więc sam wzrost MSE nie dowodzi rozjechania krytyka.
  Brak explained variance i rozkładu targetów uniemożliwia taki wniosek.
- Spadek entropii z0,618 do−0,774 nie oznacza zaniku wariancji eksploracji.
  Końcowe latent std dla gaz/hamulec/skręt to0,942/0,873/0,920. Entropia ciągłego,
  przekształconego rozkładu może być ujemna; to nie entropia kategorii akcji.
- LR na końcu0,00029475: zgodny z początkiem długiego harmonogramu, nie wygasł
  przedwcześnie. GAE i terminal/truncation w sprawdzonych ścieżkach kodu nie
  ujawniły oczywistej zamiany etykiet lub wyzerowania bootstrapu granicy rolloutu.
- Przy50 ms i gamma0,995 e-folding dyskonta wynosi9,975 s; nagroda odległa o55 s
  ma wagę0,00403. Bezpośredni ślad reszt GAE (gamma*lambda0,95) ma e-folding0,888 s.
  To własność konfiguracji, nie granica informacji wyuczonego V. Nie dowodzi,
  że zwiększenie gamma naprawi jazdę. Zmiana wymaga spójnego reward_gamma i kontroli.

Wspólny optymalizator actor/value przerywa również uczenie value po limicie KL
aktora. To sensowna hipoteza ograniczenia fitu value, ale nie dowód błędu lub
uzasadnienie automatycznego rozdzielenia optimizerów.

## Co poprawić i sprawdzić przed kampanią

1. **Uzupełnić widoczność danych.** `wandb_metrics._update_metrics` pomija `state/*`:
   KL, clip_fraction, early_stop, entropy i learning_rate istnieją we wszystkich71
   lokalnych events, ale nie trafiają do W&B. Dodać jawne metryki PPO, liczbę
   wykonanych epok/minibatchy, końcowe full-batch KL, explained variance,
   statystyki return/value oraz normę gradientów actor/value. To pierwsza
   poprawka techniczna, bez zmiany algorytmu i wspólnych cech.
2. **Zapisywać ograniczony ślad oceny.** Co najmniej10 przejazdów: race time,
   pozycja, postęp, prędkość, issued gas/brake/steer, terminal reason i odległość
   od geometrii, z trial/step oraz identycznością checkpointu. Zapis asynchroniczny
   lub buforowany, test wpływu na50 ms. Nie używać post-action logów jako current
   input. Osobno reset snapshot przed pierwszą akcją. To pozwoli rozstrzygnąć
   trzy fragmenty zamiast zgadywać po końcowym procencie.
3. **Najpierw sprawdzić hipotezę krótkiego treningu.** Propozycja, NIE uruchomiony
   plan: świeży seed17,204800 kroków (100 kompletnych rolloutów), ten sam PPO/LR
   horizon/reward/model/GNN/gamma, cap4 h. Z góry10 deterministycznych ocen przy
   102400 i204800; wyłącznie końcowe10 stanowi bramkę8/10, bez wybierania checkpointu.
   Pierwsza ocena pokazuje trend. Limit czasu nadal obowiązuje, nie extend.
   Nie restartować r3 i nie wznawiać go na zmienionym kodzie.
4. **Dopiero wynik decyduje o kolejnym wariancie.** Jeżeli final nadal powtarza
   te same miejsca, kontrola jednej zmiany, wybranej na podstawie śladu jazdy.
   Jeżeli explained variance potwierdzi niedofitowanie value i limit actorKL
   urywa większość jego treningu, test osobnego budżetu value. Jeżeli dominują
   ucięcia wyścigów pod koniec, kontrola kontynuacji przez rollout_boundary.
   Gamma/entropy/normalizacja nie są teraz uzasadnioną siatką do strojenia.
5. **Kwalifikacja i infrastruktura.** Po udanym ograniczonym wariancie sprawdzić
   seedy29/43, kryteria ustalone przed startem, bez selekcji po wyniku. PPO full
   nadal BLOCKED. Dla długiej kampanii potrzebne>=100 GiB wolnego miejsca,
   nowy zweryfikowany plan/pins po zmianach loggera oraz supervisor korzystający
   z PowerShell7; obecny stary campaign helper nadal wskazuje Windows PowerShell5.1.
   Nie zmieniać starego przypiętego campaign.json ani frozen runtime'u w miejscu.

Nie uruchomiono nowego pilota ani nie zastosowano proponowanych zmian treningowych.
Zachowano oryginalne checkpointy, ich modele i Adam; SHA wszystkich trzech
odczytanych checkpointów sprawdzono przed i po analizie.

## Dowody i odtworzenie

- `evidence/ppo-pilot-20261013-r3/diagnosis.json`: komplet wyników zbiorczego odczytu
 71 rolloutów,71 aktualizacji,10 ocen i checkpointów60/70/71.
- `pilot_tools/audit_ppo_result.py`: audyt CPU1thread, zero optimizer steps,
  zero inference/control, walidacja zgodności liczników i niezmienionych SHA.
- Oryginalne dane: `BASE/tmrl-ppo-p13-prefix-r3-s17/{events.jsonl,evaluation.json,episodes,checkpoints}`.
- Formuła PPO i ochrona przez KL: [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
  Dokumentacja potwierdza rolę KL i rozróżnienie losowej kolekcji od oceny;
  nie stanowi dowodu przyczyny wyniku tego pilota.

Audyt uruchomiono z cwd/PYTHONPATH frozen r3, Python3.12 z przypiętego środowiska,
PYTHONNOUSERSITE1, CUDA_VISIBLE_DEVICES−1. Jedyny nowy kod w editable/pilot_tools;
nie zmieniano wspólnego modelu, nagrody, GNN, replay ani starej kampanii.
