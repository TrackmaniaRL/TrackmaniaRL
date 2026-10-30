# Prototyp porównania headless

Status: przygotowanie, nie uruchamiać. `manifest.json` opisuje pięć niezależnych
treningów od zera, seed 17, budżet 7 168 000 przejść i 30 końcowych prób.
Kolejność w manifeście jest robocza. Pliki w `reference_configs` są oryginalnymi
konfiguracjami kampanii v5 z commita 1925336a, zachowanymi jako punkt odniesienia;
korzystają z OpenPlanet i nie są konfiguracjami headless. Ich SHA-256 zgadzają
się z manifestem źródłowym. Nie wznawiać starych treningów w ich katalogach.

Branch kampanii `codex/headless-five-algorithm-campaign` powstał z lokalnego
`codex/headless-training-parity` na commicie `b633d56f`. W chwili utworzenia
tamten katalog zawierał liczne niezacommitowane zmiany, a lokalny branch był
9 commitów przed i 2 za swoim odpowiednikiem zdalnym. Te zmiany nie są częścią
tej migawki. Przed integracją kampanii trzeba uwzględnić zakończoną i sprawdzoną
wersję prac headless, włącznie ze zmianami obecnie w toku, i ponownie zweryfikować
zgodność oraz stan obu branchy. Nie kopiować ich w ciemno z aktywnego katalogu.

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
