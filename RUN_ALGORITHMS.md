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

## Własne treningi

Gotowe konfiguracje bazowe biblioteki dostaniesz przez:

```text
uv run trackmaniarl init my-agent --template trackmania
```

Projekt zawiera m.in. `run-ppo.yaml`, `run-sac.yaml`, `run-tqc.yaml` oraz warianty
Q/QR/IQN/FQF z lidarem lub obrazem. README wygenerowanego projektu opisuje mapę,
geometrię i uruchomienie. Wybierz konfigurację, ustaw własny `run_id`, `seed`
i budżet; nie trzeba kopiować pliku dla każdego seeda do repozytorium.

Jeżeli chcesz dokładnie model i mapę z naszego porównania, wygeneruj jeden plik:

```text
uv run python -m experiments.tmrl_test_comparison.generate --algorithm ppo --seed 17 --stage pilot --output artifacts/configs/my-ppo.yaml
```

To tylko zapis konfiguracji, bez uruchamiania gry. Generator odmawia nadpisania
pliku. Ten profil wymaga lokalnej kalibracji skrętu opisanej w instrukcji zespołu;
przed własnym treningiem dostosuj też projekt W&B. W kampanii zespołowej używaj
`prepare`: generuje wszystkie seedy lokalnie, a wynikowe YAML-e są ignorowane przez Git.
