Cześć, przygotowana jest wspólna kampania treningowa Trackmanii:

- Jakub: QR, potem IQN.
- Borys: SAC ciągły.
- Kamil: TQC.
- Kuba P.: PPO recovery.
- SD-SAC pomijamy.

Każdy algorytm: seedy 17/29/43, po 8 192 000 kroków i 30 ocen końcowych.
To około 2 tygodnie samej jazdy na 3 seedy (u Jakuba około 4 tygodnie), plus uczenie
oraz zapisy. Windows + NVIDIA/CUDA, co najmniej 100 GiB miejsca na start.

Branch: codex/tmrl-algorithm-runs. Dokładny commit podany w RELEASE.json obok
instrukcji; wszyscy używamy tego samego. Nie instalujcie wersji z PyPI.

Instrukcja: experiments/tmrl_test_comparison/release_20261014/START_HERE.md.
Są tam instalacja, gra/mapa/plugin, własny W&B i kalibracja, kontrola środowiska,
polecenie startu dla każdej osoby, STOP, backup i lista oddawanych wyników.
Przed pierwszym startem odeślijcie hash commita i raport host-check;
po starcie link W&B. Nie wysyłajcie kluczy ani .env.

Trening od zera, bez checkpointów/replayu pilota. Nie aktualizujcie kodu,
ustawień ani zależności w trakcie. Przekroczenia 100 ms raportujemy, nie blokują
same w sobie startu; błędy sterowania, zła mapa i niepełne wyniki nadal blokują.
Przy problemie zachowajcie logi i STOP, bez automatycznych powtórek.
