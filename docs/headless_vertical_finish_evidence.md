> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Ostatni rekord V7: rzeczywisty EOF i synchroniczna blokada

Badanie dotyczyło końca oryginalnej taśmy4152. Nie uruchamiano treningu ani
kontrolera Windows. Nie zmieniano stanu fizycznego, zegara, shima V7, pinów,
limitów czasu epizodu lub kontraktu terminalnego.

## Wejścia i granice badania

`scripts/headless_native_finish_probe.py` używa oddzielnego, przypiętego
launchera V7. Po bootstrap2190 jawnie wykonuje jeden administracyjny neutralny
interwał do2200. Każdy kolejny rekord odczytuje przed pierwszym załadowaniem.
Pełne32B muszą być identyczne z niezmiennym `parent.inputs.bin`, SHA256
`9f5c8fc74e59875954d63299bc933e81f9dac32be3658ec36226ca79581ea64f`.
Op6 zapisuje te same oryginalne bajty SGB. Potwierdzenie, wszystkie audyty
rekordu i shadow oraz rzeczywiste POST muszą pozostać zgodne. Te zapisy nie
otrzymują fikcyjnych identyfikatorów akcji polityki.

Badania mieszczą się w120s i64MiB; zmierzony czas obejmuje równoległe prace na
komputerze i nie jest benchmarkiem. Każdy katalog przechowuje własne źródło,
surowe żądania/odpowiedzi, POST, audyty, receipt launchera i raport cleanupu.

| Badanie pod `<USER_HOME>/tmrl-headless-parity-20261021-003/` | Wynik |
| --- | --- |
| `finish001/` | 3999 oryginalnych interwałów +1 neutralny, native42190; skończony budżet przed ostatnim rekordem, bez EOF i bez dowodu mety; cleanup0, `remainingOwned=[]`, 6,67s |
| `finish002/` | próba skonsumowania wszystkich4000 oryginalnych interwałów po neutralnym resecie; ostatnie op6 N5 zaakceptowane, następnie reader-stage EOF i odrzucenie audytu dodatkowego POST; zachowany błąd, `remainingOwned=[]`, 6,79s |
| `finish-guard001/` | poprawiony produkcyjny preflight odrzucony przy rzeczywistym końcowym readerze, bez odczytu natywnego i bez żądania; normalny cleanup0, `remainingOwned=[]`, 7,58s |

## Dlaczego finish002 nie jest dowodem mety

Przy readerze p4147/count4152 op6 N5 zachował rekordy4147..4151. Otrzymano
boundary ACK native42200. Kolejne ARM/op3 oczekiwało następnego readera, ale
oryginalny silnik po wyczerpaniu tablicy omija tę ścieżkę. Powstał kolejny
rzeczywisty POST native42210, wykraczający poza zaakceptowane N5.

To nie jest naturalny koniec walidacji. Zachowany
`runtime/build/server-probe/finish002/work/root/srv/server.log` kończy się
literalnym `FKPOST reject 337`. W źródle V7
`source/tools/search/forkshim/src/post_observe/native_ipc.rs::at_boundary`
warunek `ARMED != 0` wywołuje `fail(337)` po zapisaniu POST. `fail` zapisuje
komunikat i wykonuje exit_group95. Jest to źródłowe wyjaśnienie przerwania;
oryginalny driver nie posiada waitpid status tego dziecka.

Pełny stdout walidatora zawiera wyłącznie `b'[\n'`, bez zamkniętej tablicy JSON.
Istnieje40 pośrednich rekordów TMRESULT po768B. Ostatni ma native42000,
Time4294967295, zero checkpointów (liczba184 i długość danych192). Nie odpowiada
końcowemu POST i nie ustanawia mety. Cleanup drivera po błędzie diagnostycznym
miał returncode−9, bo supervisor zatrzymał pozostałych właścicieli; nie jest to
status naturalnego ukończenia walidatora.

Dodatkowego POST nie pominięto, nie przypisano do poprzedniej decyzji i nie
uznano za nagrodę. Produkcyjna kontrola overshoot już wcześniej odrzucała ten
przypadek prawidłowo. Nie ma podstaw do oznaczenia `terminated=True`.

## Poprawka i rzeczywista weryfikacja

W `_runtime/immediate.py::ImmediateSession.advance` obowiązuje teraz
`p + n < input_count`, sprawdzane przed odczytem kanałów i przed op6. Żądanie
zużywające ostatni rekord jest odrzucane w całości z błędem
`V7 unsupported-final-reader-boundary`. V7 wymaga jeszcze jednego readera do
zwrócenia synchronicznego stanu; nie ma zatwierdzonego kontraktu drain/release.
Ta poprawka nie skraca N po cichu i nie zmienia zakresu kwalifikacji terminalnej.

Cztery testy wykorzystują dosłowny reader z finish002, sprawdzają odrzucenie
całego N5 i pojedynczego ostatniego rekordu przed operacjami natywnymi oraz
przepuszczenie N4 pozostawiającego reader. Razem z dotychczasowymi testami V7
i FP:35PASS w Linux/ext4; Ruff bez błędów.

`finish-guard001` odtworzył3995 oryginalnych interwałów, zatrzymał żywe dziecko
przy p4147/native42150 i wywołał **dokładną produkcyjną metodę advance** na tym
rzeczywistym kontekście. Nie zastępowano jej warunku atrapą. Jest to test metody
z eksperymentalnym, zweryfikowanym launcherem, a nie osobna kwalifikacja całej
procedury provisioningu produkcji. Odrzucenie miało:

- zero odczytów natywnych i zero nowych żądań podczas wywołania;
- identyczne przed/po: sekwencja, epoch, RX, liczba żądań/odpowiedzi, długość
  POST, reader, boundary i SHA całej tablicy wejść;
- tę samą tożsamość PID/birth/UID/PGID/SID i żywy driver;
- normalne zamknięcie dopiero po weryfikacji, bez sygnałów cleanupu.

Pełna fizyczna zgodność klient/serwer i naturalna meta pozostają niepotwierdzone.
