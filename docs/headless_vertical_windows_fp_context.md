> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Odczyt FP klienta Windows przez Process Snapshot API

Diagnostyczny `scripts/headless_windows_fp_context.py` jest osobnym czytnikiem.
Nie zmienia profilu V7 ani przypiętych hashy shima. Odczytuje kontrolę FP
wątków konkretnego klienta przy jednym zatrzymanym przebiegu testu. Sam taki
zapis **nie identyfikuje wątku fizyki ani momentu wykonania fizyki**.

## Zakres i tożsamość

Dozwolony plik to `D:\Epic Games\Trackmania\Trackmania.exe`, SHA256
`3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`.
Czytnik otwiera wskazany PID z `PROCESS_QUERY_INFORMATION`, sprawdza ścieżkę,
hash pliku, AMD64 bez WOW64, czas utworzenia procesu oraz jego żywotność.
Właściwy capture wymaga literalnego czasu utworzenia FILETIME z wcześniejszego
wywołania `--inspect`. Proces jest utrzymywany otwartym uchwytem i ponownie
sprawdzany po odczycie. Nic nie podnosi uprawnień ani nie prosi o all-access.

Przed snapshotem metadane Toolhelp ograniczają liczbę wątków do 256. Snapshot
również musi mieć 1–256 wątków; zmiana liczby podczas capture może zakończyć
się odrzuceniem wyniku. Jeden oddzielny worker ma pięciosekundowy limit,
wewnętrzna iteracja limit czterech sekund. Timeout oznacza wynik błędu,
zakończenie workera i oczekiwanie na jego zamknięcie.

## Rzeczywiste API i układ

Użyte są wyłącznie `PSS_CAPTURE_THREADS | PSS_CAPTURE_THREAD_CONTEXT = 0x180`
oraz `CONTEXT_AMD64 | CONTROL | FLOATING_POINT = 0x100009`. Nie ma VA clone,
przechwytywania stron pamięci, uchwytów ani stosów. Nie ma manualnego
SuspendThread/ResumeThread, GetThreadContext ani SetThreadContext. Przejściowym
snapshotem zarządza Windows; program nie utrzymuje własnego suspend count.
Definicje flag podają [Microsoft PSS_CAPTURE_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/processsnapshot/ne-processsnapshot-pss_capture_flags)
i [WinNT.h](https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/um/winnt.h).

Struktury ctypes odzwierciedlają nagłówki Microsoftu. AMD64 CONTEXT ma 1232 B:
ContextFlags +48, MxCsr +52, Rip +248, FltSave +256. W obszarze FXSAVE:
control word +0, status word +2, MXCSR +24 i mask +28. x64 PSS_THREAD_ENTRY
ma 120 B, wskaźnik ContextRecord +112. Układ obejmuje czterobajtowe wyrównanie
FILETIME po WORD LastSyscallNumber; nie zakłada wyrównania FILETIME do 8 B.
[CONTEXT](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-context)
oraz [ProcessSnapshot.h](https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/um/ProcessSnapshot.h)
są źródłami definicji.

Każdy kontekst musi zawierać żądane flagi, co najmniej 1232 B i wskaźnik
wyrównany do 16 B. Wskaźnik pochodzi z lokalnego snapshotu; jego żywotność
wiąże się z walk marker. Odczyt nie jest ReadProcessMemory na kliencie.
Top-level MxCsr musi być identyczny z FltSave.MxCsr.
[Microsoft opisuje ContextRecord i jego żywotność](https://learn.microsoft.com/en-us/windows/win32/api/processsnapshot/ns-processsnapshot-pss_thread_entry).

Do raportu trafiają wyłącznie PID/tożsamość, identyfikatory i czasy wątków,
kontrolne/statusowe słowa FP wraz z jawnie dekodowanymi bitami oraz nazwa
modułu i RVA dla RIP. Pełny kontekst, rejestry ogólne, adres RIP, stos i
pamięć klienta nie są eksportowane. Moduły pochodzą z metadanych Toolhelp;
brak dopasowania jest jawny, nakładanie się mapowania jest błędem.

Marker i lokalny snapshot są zwalniane w zagnieżdżonym finally, także po
błędzie walidacji lub iteracji. `PssFreeSnapshot` otrzymuje
**GetCurrentProcess()**, gdyż snapshot należy do workera, a nie do klienta.
To rozróżnienie wynika z [dokumentacji PssFreeSnapshot](https://learn.microsoft.com/en-us/windows/win32/api/processsnapshot/nf-processsnapshot-pssfreesnapshot).
Raport błędu po capture zachowuje kody cleanupu. Zakończenie samego workera
zwalnia zasoby jego procesu; kod nie obiecuje kontroli nad czasem wewnętrznych
operacji jądra.

## Uruchomienie i interpretacja

Najpierw `--pid PID --inspect --output NOWY_PLIK_JSON`. Następnie, po
sprawdzeniu, że kontroler testu już nie prowadzi auta: `--pid PID
--expected-creation-filetime LITERAL_Z_INSPECT --output KOLEJNY_NOWY_PLIK_JSON`.
Pliki otwierane są wyłącznie jako nowe; wcześniejsze dowody nie są nadpisywane.
Access denied lub brak obsługi PSS kończy badanie bez szukania obejścia.

Na etapie przygotowania przeprowadzono 9 testów bez dostępu do gry: układ,
dekoder, brak eksportu pełnych rejestrów oraz zwolnienie lokalnego snapshotu
po błędach query, alokacji markera, iteracji, tożsamości i dekodowania.
Są to testy obsługi zapisu, a nie dowód wyniku FP klienta.

Zachowano źródłowe nagłówki Microsoftu i receipt pobrania w
`<USER_HOME>/tmrl-headless-parity-20261021-003/vertical-windows-fp-source001/`.
Oba pochodzą z commita `2f767671802301cf043c717d66bf004ed7863487`:
WinNT.h SHA256 `d66df443218279ed2ffe7718b5347e64bf712dbfa5d25a3b0692787169405c69`,
ProcessSnapshot.h SHA256
`946ea96d560fb2644b460b04f7494b5d7aa0ec1c687d2b53ec6e419b61b2b42e`.

## Pierwszy rzeczywisty odczyt

Po zwolnieniu kontrolera testowego wykonano identity-only inspect, a następnie
jeden capture. PID50068, creation FILETIME134368745122451526, ścieżka i SHA
przeszły kontrolę. Capture został odrzucony jako brakujący, zbyt krótki lub
niewyrównany CONTEXT. Oba kody cleanupu wynoszą0, worker zakończył się po0,438s.
Nie eksportowano niezweryfikowanych słów FP i nie poluzowano tego wyniku.
Dowody zachowano w
`<USER_HOME>/tmrl-headless-parity-20261021-003/vertical-windows-fp001/`.

Pierwszy komunikat połączył trzy warunki i nie zachował ich poszczególnych
wartości. Kolejna wersja źródła dodaje do odrzucenia sam rozmiar kontekstu,
resztę wyrównania i flagi wątku, bez adresu wskaźnika. Na podstawie pierwszego
raportu nie da się stwierdzić, który z warunków był przyczyną. Odczyt PSS
potwierdził dostępność API i poprawne zwolnienie zasobów; stan FP klienta
pozostaje nieustalony.

## Kontynuacja r4

Rzeczywiste metadane wskazały zakończony wątek (`Flags=1`) bez kontekstu.
`PSS_THREAD_FLAGS_TERMINATED=0x0001` jest w przypiętym ProcessSnapshot.h.
Czytnik zachowuje ten wpis osobno; nie odczytuje nieistniejącego kontekstu.
Wszystkie aktywne konteksty nadal przechodzą dotychczasową ścisłą walidację,
a nieznane flagi, brak aktywnych wątków i niespójna enumeracja są odrzucane.

Capture r4 potwierdził139 wpisów,135 aktywnych kontekstów i4 zakończone.
Wszystkie135 aktywne mają CW `0x027f`, czyli53 bity x87; kontrolne bity MXCSR
odpowiadają nearest/FTZ-off/DAZ-off i maskowaniu wyjątków. Kody cleanupu oba0.
Nie ustalono wątku/etapu fizyki. Eksperyment native V9 z samą precyzją53 nie
zmienił200 stanów fizycznych względem V7, więc nie usunął różnicy pionowej.
Szczegóły i ograniczenia są w [raporcie r4](headless-parity-diagnostics-r4.md).
