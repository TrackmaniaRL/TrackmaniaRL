# Treningi na kilku komputerach

SD-SAC jest **experimental** i nie uczestniczy w tej kampanii. Jego automatyczne
naprawy są wstrzymane. Przygotowano 15 niezależnych runów; treningów nie uruchomiono.

| Osoba | Algorytmy kolejno | Seedy każdego |
| --- | --- | --- |
| Jakub | QR, IQN | 17, 29, 43 |
| Borys | ciągły SAC | 17, 29, 43 |
| Kamil | TQC | 17, 29, 43 |
| Kuba P. | PPO persistent recovery | 17, 29, 43 |

Nowy profil ma 8 192 000 kroków/seed: nominalnie około 14 dni jazdy na algorytm
z trzema seedami, około 28 dni u Jakuba. Sprzęt, uczenie i restarty wydłużą czas.
Historyczny profil full 2 048 000 pozostaje oddzielnym protokołem.

[Instalacja, start, STOP i przerwy na Windows oraz Linux X11](experiments/tmrl_test_comparison/team_campaign/START_HERE.md).
[Historia wyników i ograniczeń](https://github.com/TrackmaniaRL/TrackmaniaRL/tree/263fe78bbaa8c42ee86c38ae6f8b90e505f80536/experiments/tmrl_test_comparison).

Najpierw w czystym checkoutcie identycznego commita sprawdź swój plan bez jazdy:

```text
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.team_campaign.campaign prepare
uv run --frozen --group dev python -m experiments.tmrl_test_comparison.team_campaign.campaign jobs --worker borys
```

Podstaw `jakub`, `borys`, `kamil` lub `kuba-p`. Dopiero opisane w pełnym planie
lokalne sprawdzenia i jawny `run --launch` rozpoczynają trening. Nie uruchamiaj kilku
kontrolerów jednej gry. Nie kopiuj `.env`, starych checkpointów ani cudzej kalibracji.
