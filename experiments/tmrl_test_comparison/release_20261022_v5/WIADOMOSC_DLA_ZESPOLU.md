# Paczka treningowa v5

Nowe próby uruchamiamy na wspólnym commicie gałęzi `codex/tmrl-algorithm-runs`,
z biblioteką 1.2.16. Każdy seed ma teraz **7 168 000 kroków**, czyli o 12,5% mniej,
oraz dotychczasowe 30 ocen końcowych. Podział osób i algorytmów się nie zmienia.

Paczka zawiera poprawione metryki po wznowieniu, oszczędniejsze obliczenia QR/IQN,
dokładniejsze raporty opóźnień i clippingu oraz zachowane checkpointy etapowe.
Obejmuje też wcześniejsze poprawki zgodności modeli, PB w W&B, fokusu i precyzji.
Wersja 1.2.16 dodaje gaz podczas odliczania i kończy przyjęte aktualizacje przed
zapisem po STOP. SD-SAC nadal nie należy do wspólnej kampanii porównawczej.

**Trwające treningi dokończcie na ich oryginalnym kodzie.** V5 jest osobną kampanią;
nie przenosimy do niej checkpointów, replayu, pinów ani wyników starego budżetu.
Przy awarii zachowajcie STOP i artefakty; kolejki same się nie wznawiają.

Pełna instalacja, komendy dla Windows/Linux i podział:
[START_HERE.md](START_HERE.md). Zaczynamy od `verify`, `jobs` i `host-check`.
Samo przygotowanie tej paczki nie uruchomiło żadnej próby.
