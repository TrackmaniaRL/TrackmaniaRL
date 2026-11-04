# Prototyp porównania headless

Status: przygotowanie, nie uruchamiać. `manifest.json` opisuje pięć niezależnych
treningów od zera, seed 17, budżet 7 168 000 przejść i 30 końcowych prób.
Kolejność w manifeście jest robocza. Pliki w `reference_configs` są oryginalnymi
konfiguracjami kampanii v5 z commita 1925336a, zachowanymi jako punkt odniesienia;
korzystają z OpenPlanet i nie są konfiguracjami headless. Ich SHA-256 zgadzają
się z manifestem źródłowym. Nie wznawiać starych treningów w ich katalogach.

Branch kampanii `codex/headless-five-algorithm-campaign` powstał z lokalnego
`codex/headless-training-parity` na commicie `b633d56f`. Następnie włączono
migawkę lokalnej pracy w toku `3ef45679`, obejmującą również eksperymenty.
Nie oznacza to kwalifikacji kodu do kampanii. Lokalny branch rozszedł się ze
zdalnym: tamtejsze dwa commity wymagają osobnej integracji i weryfikacji.
Przed kampanią należy uwzględnić dalsze poprawki headless oraz ponownie
sprawdzić zgodność, testy i stan obu branchy.

Przed przygotowaniem wykonywalnej kolejki:

1. Ustalić docelowy commit poprawionego headless i zbudować osobny deployment;
   zapisać identyfikatory źródeł, assetów, mapy i konfiguracji.
2. Dostosować pięć konfiguracji do tego backendu, w tym zweryfikować ciągłe
   sterowanie SAC, TQC i PPO. Nie zastępować go po cichu akcjami dyskretnymi.
3. Zapewnić wspólny licznik przejść, checkpointy na progach z manifestu,
   osobne katalogi, kolejkę i bezpieczne wznowienie pełnego stanu. Zatrzymać
   kolejkę przy błędzie i zachować przyczynę oraz niekompletne wyniki.
4. Sprawdzić krótkie treningi i końcową ewaluację dla każdej metody, odczyt
   checkpointów, zamykanie procesów, miejsce na dyskach i zasilanie/uśpienie.
   Ostatnia znana czterogodzinna kwalifikacja headless zakończyła się błędem.
5. Zamrozić konfiguracje, opisać różnice względem oryginalnego środowiska i
   dopiero na wyraźne polecenie użytkownika uruchomić kampanię.

Jeden seed to wstępne porównanie, nie pomiar zmienności między treningami.

`prepare.py` przygotowuje oddzielną kopię źródeł V8 i kandydackie konfiguracje
QR, IQN, Q, FQF oraz discrete SAC. Zapisuje skróty źródeł, konfiguracji, mapy
i referencji; nie uruchamia gry. SAC, TQC, PPO i REDQ pozostają jawnie
zablokowane do czasu kwalifikacji sterowania ciągłego. Dodatkowe Q, FQF,
discrete SAC i REDQ pochodzą ze starszych konfiguracji 2 048 000 kroków;
generator oznacza je jako eksploracyjne i zmienia budżet na 7 168 000 oraz
nagrodę na wspólną dla zespołowego QR. Discrete SAC nie należał do głównej
piątki kampanii zespołowej.

Przykładowe przygotowanie z katalogu repo w WSL, przy użyciu zachowanego profilu
na tym komputerze (wybierz **nowe** katalogi docelowe):

```bash
python -m experiments.headless_comparison_prototype.prepare \
  --source "$PWD" \
  --native-template PATH_TO_NATIVE_TEMPLATE/qr.yaml \
  --destination PATH_TO_NEW_COMPARISON \
  --artifacts /mnt/c/Users/szulc/nowy-headless-comparison/artifacts
```

Przygotowanie nie kwalifikuje środowiska ani nie tworzy kolejki do odpalenia.
Należy powtórzyć je po zamrożeniu końcowej wersji kodu oraz referencji.

Prototyp kolejki jest w `tools/headless_campaign_queue`. Z katalogu repo można
sprawdzić obecny plan poleceniem:

```powershell
python -m tools.headless_campaign_queue experiments/headless_comparison_prototype/manifest.json --output artifacts/headless-five-algorithm-campaign
```

Obecnie celowo zgłosi brak gotowości, również z `--execute`: manifest nie
zawiera zweryfikowanych komend dla headless. Ogólna kolejka wymaga osobnego,
kompletnego manifestu i zaufanego wrappera potwierdzającego wynik. Wznowienie
jest ograniczone do przerw między ukończonymi zadaniami po sprawdzeniu dowodów;
nie wznawia przerwanego treningu. Eksperymentalny wrapper Windows zapobiega
bezczynnemu uśpieniu podczas pracy, ale nie wymuszonemu restartowi ani uśpieniu.
Rzeczywisty trening i jego odtworzenie nadal wymagają kwalifikacji.
