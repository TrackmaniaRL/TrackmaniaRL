# Walidacja kampanii v5 — 22 października 2026

Status: przygotowana, nieuruchomiona. Plan wymaga biblioteki 1.2.16 i tagu
`v1.2.16` w historii checkoutu przed jakimkolwiek użyciem gry. Nie aktualizuje
rozpoczętych prób ani checkpointów poprzednich kampanii.

Bibliotekę scalono w PR #61 wraz z poprawkami z PR #54 i #60. Tag `v1.2.16`
wskazuje scalony commit `dd8cc1f1b81c46f9ed9e823f52da17382046878d` i należy
do historii tej gałęzi. Wszystkie cztery kontrole CI wydania przeszły: pełne
zestawy Windows i Linux po 1283 testy PASS i jednym pominiętym, kontrola kodu
oraz budowanie paczek. Lokalny pełny zestaw biblioteki: 1279 PASS, 5 pominiętych.

- 15 konfiguracji: QR, IQN, SAC, TQC i PPO; seedy 17, 29, 43.
- Wspólny budżet 7 168 000 kroków na seed, 12,5% mniej od 8 192 000.
- Końcowa ocena: 30 prób na ostatnim poprawnie zapisanym checkpointcie.
- Modele etapowe: 2 048 000 / 4 096 000 / 6 144 000 / 7 168 000.
- Jawnie włączony gaz podczas odliczania; poprawione opróżnienie przyjętych
  aktualizacji przed zapisem po STOP. Awarie nadal uruchamiają ograniczone
  czasowo zamknięcie; plan nie wykonuje automatycznych wznowień.
- Weryfikacja planu, konfiguracji, źródeł i zasobów: PASS.
- 75 lokalnych testów kontraktów kampanii i nadzoru procesów: PASS, w tym
  rzeczywiste wykluczanie kontrolerów przez mutex między procesami.
- Historyczne plany v2/v3/v4 zachowane bez nadpisywania i przepinania źródeł.
  Testy historycznych kontraktów używają zapisanych tożsamości; rzeczywisty
  launcher v4 odrzuca bibliotekę 1.2.16.
- SD-SAC pozostaje wariantem eksperymentalnym poza wspólnym porównaniem.
- Headless nie jest hostem tej kampanii; brak kwalifikacji zgodności z grą.

Zmniejszenie budżetu skraca liczbę kroków, nie gwarantuje identycznego czasu
ściennego na różnych komputerach lub dla różnych algorytmów. Historyczny QR17
przy 8,192 mln kroków pozostaje osobnym wynikiem rozwojowym.
