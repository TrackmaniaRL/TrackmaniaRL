# Walidacja kampanii v4 — 21 października 2026

Status: przygotowana, nieuruchomiona. Biblioteka 1.2.14 została scalona w PR #56
i oznaczona tagiem wydania. Publikację paczek oraz pełną walidację tej gałęzi
śledzą procedury GitHub Actions; gałąź zespołu zostanie scalona po ich powodzeniu.
Launcher blokuje użycie gry, dopóki tag v1.2.14 nie należy do historii checkoutu.

- Zamrożony plan: 15 konfiguracji, pięć metod, trzy seedy każdej metody.
- Weryfikacja źródeł, konfiguracji i limitów: PASS. Historycznych v2/v3 nie przepinano.
- Budżet: 7 168 000; modele etapowe 2 048 000 / 4 096 000 / 6 144 000 / 7 168 000.
- Końcowa ocena: 30 prób, ostatni checkpoint; timing pozostaje raportowany osobno.
- Lokalna walidacja CPU: 51 testów PASS; dwa testy rzeczywistego mutexu pominięto,
  ponieważ host ma aktywny kontroler QR. Uruchamia je CI na pustym hoście.
- Sprawdzono kontrakty v2/v3/v4, tożsamości procesów przy ponownym użyciu PID,
  checkpointy etapowe, zgodność aktualizacji scalar/QR i kontrolę fokusu.
- Dodatkowe 64 testy przenośności dawnych pilotów, odsyłaczy, aliasów SD-SAC
  i kontraktu przechwytywania obrazu: PASS. Dawne testy używają danych w katalogach
  tymczasowych; historyczne zabezpieczenia źródeł nie zostały poluzowane.
- Ruff i kontrola lockfile: PASS. Testy nie sterowały grą ani aktywnym treningiem.
- Kandydat biblioteki: wszystkie cztery zadania CI PASS; Windows i Linux po
  1225 testów PASS i jednym pominiętym. Zbudowano i zweryfikowano wheel oraz sdist.

Nie wykonano nowego treningu ani porównania jakości po zmianach. Oszczędność
12,5% dotyczy limitu kroków; czas rzeczywisty zależy od metody i hosta.
Historyczny QR17 przy 8,192 mln nie zastępuje seeda nowej kohorty.
