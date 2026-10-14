# Przygotowanie paczki — 14 października 2026

- Nowy plan weeks-20261014-v2:15niezależnych konfiguracji, 8 192 000kroków,
  30ocen/seed; seedy17/29/43; SD-SAC i aliasy wyłączone.
- Jawna decyzja człowieka: opóźnienia raportujemy, nie stosujemy progu100ms
  do dopuszczenia nowej kampanii. Nie zmieniono historycznych ocen strictFAIL.
- PPO persistent recovery, oryginalny updater; pozostałe komponenty/nagroda
  wspólne z dotychczasowym generatorem. Żadnego KLguard ani cudzego checkpointu.
-36testów planu, STOP i ocenPASS; po końcowej korekcie typów3testy nowej wersjiPASS.
- Ruff i mypy nowego modułuPASS, parser obu skryptów Windows PowerShellPASS.
- Wszystkie15przydzielonych konfiguracji przeszły config-check bez gry.
- Nowy i stary plan source/config verifyPASS; nie edytowano starych planów/pins.
- Nie uruchomiono treningu, preflightu gry, benchmarku ani kalibracji.
- Nie potwierdzono środowisk kolegów. Każdy potrzebuje własnej kalibracji,
  klucza W&B, czystego checkoutu, CUDA i raportu host-check.
- Windows PowerShell autora nadal zgłaszaRestricted; nie zmieniono ustawień
  systemu podczas przygotowania. Instrukcja wyjaśnia konfigurację po stronie operatora.
- Nadzór istniejącej kampanii jest procesem lokalnym, nie niezależną usługą.
  STOP i limit336h pozostają; nie ma automatycznego retry/resume.
