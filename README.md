# RAG-Studienarbeit
##### Erstellt von Tristan Muri

Dieses Repository ist eine Sammlung von verschiedenen Ansätzen für das Preprocessing von Dokumenten.
Der Output der meisten dieser Skripts ist Markdown. 
Grund dafür ist das einfachere Prüfen des Resultats im Vergleich zu json.
Eines dieser Skripts(pdf_to_image_3.py) braucht einen Google AI studio API key. 
Der rest kann einfach eine lokale Library installiert werden.

### Beschreibung Skripts
Die einzelnen Skripts haben alle einen eigenen Ausschnitt über ihre Details.

RAG_test.ipynb kann ignoriert werden da es noch keine Funktion erfüllt.

### Auswertung
Die Auswertung der verschiedenen Skripts ist auf Confluence unter 
"Task Preprocessing für LLMs" zu finden.

## PDF-Prozessor für die Vorbereitung von Contriever
### contriver_process.py

###### Output: src/processed_chunks.json

Dieses Skript dient der Verarbeitung von PDF-Dokumenten, um deren Inhalte für die weitere Nutzung im Contriever-System aufzubereiten. 
Es extrahiert den Text aus PDF-Dateien, entfernt wiederholte Kopf- und Fußzeilen und teilt den bereinigten Text in satzbasierte Abschnitte (Chunks) auf.
Funktionsweise

    Textextraktion: Das Skript verwendet die Bibliothek PyMuPDF, um den Text aus den Seiten der PDF-Dokumente zu extrahieren.

    Erkennung von Kopf- und Fusszeilen: Durch die Analyse der ersten und letzten Zeilen jeder Seite identifiziert 
    das Skript wiederholte Muster, die als Kopf- oder Fusszeilen fungieren. Diese werden basierend auf einem festgelegten 
    Schwellenwert als solche erkannt und können entfernt werden.

    Textbereinigung: Nach der Extraktion wird der Text bereinigt, indem überflüssige Leerzeichen entfernt, Zeilenumbrüche 
    korrigiert und erkannte Kopf- und Fusszeilen eliminiert werden.

    Satzbasierte Aufteilung: Unter Verwendung des spaCy-Sprachmodells wird der bereinigte Text in einzelne Sätze segmentiert. 
    Diese Sätze werden anschließend zu Abschnitten (Chunks) zusammengefasst, wobei die Wortanzahl pro Chunk durch definierte Parameter gesteuert wird.

    Speicherung der Ergebnisse: Die erzeugten Chunks werden mit Metadaten wie der Quell-PDF, 
    den Seitenzahlen und einer eindeutigen Chunk-ID versehen und in einer JSON-Datei gespeichert.

Anpassbare Parameter

    DEFAULT_MAX_WORDS: Maximale Anzahl von Wörtern pro Chunk (Standardwert: 300).

    MIN_CHUNK_SIZE: Minimale Wortanzahl für den letzten Chunk, um zu kleine Abschnitte zu vermeiden (Standardwert: 100).

    HEADER_FOOTER_THRESHOLD: Prozentsatz der Seiten, auf denen ein Muster auftreten muss, um als Kopf- oder Fußzeile erkannt zu werden (Standardwert: 0,75).

Hinweis:

    Es ist nicht so gut.

Diese Parameter können direkt im Skript an die spezifischen Anforderungen angepasst werden.
Fehlerbehandlung

Das Skript ist mit umfassenden Fehlerbehandlungsmechanismen ausgestattet, die Probleme bei der PDF-Verarbeitung erkennen 
und protokollieren. Fehler werden im Logfile contriever_process.log festgehalten, während die Verarbeitung der verbleibenden 
Dateien fortgesetzt wird, um eine möglichst vollständige Datenextraktion sicherzustellen.

Durch diese strukturierte Vorgehensweise stellt das Skript sicher, dass die aus den PDF-Dokumenten extrahierten Texte 
optimal für die weitere Verarbeitung im Contriever-System vorbereitet sind.

## Weniger Basic PDF prozessor 2
### process_pdfs.py

###### Output: src/json/*

Dieses Skript dient der Extraktion von Text und Tabellen aus PDF-Dokumenten. Es verarbeitet alle PDFs in einem angegebenen Verzeichnis, extrahiert den unstrukturierten Text sowie die strukturierten Tabellendaten und speichert die kombinierten Ergebnisse als JSON-Dateien.
Funktionsweise

    Verzeichniskontrolle: Das Skript überprüft, ob das Ausgabeverzeichnis existiert, und erstellt es bei Bedarf.

    PDF-Verarbeitung: Für jede PDF-Datei im Eingabeverzeichnis wird Folgendes durchgeführt:

        Textextraktion: Mit PyMuPDF wird der Text aus jeder Seite der PDF extrahiert und in einer Liste gespeichert.

        Tabellenextraktion: Mit PDFTableExtractor werden Tabellen erkannt und extrahiert.

    Datenzusammenführung: Die extrahierten Texte und Tabellen werden in einem Dictionary zusammengeführt.

    Speicherung: Das kombinierte Ergebnis wird als JSON-Datei im Ausgabeverzeichnis gespeichert, wobei der Dateiname der ursprünglichen PDF-Datei entspricht.

    Fehlerbehandlung: Tritt während der Verarbeitung ein Fehler auf, wird dieser im Log festgehalten, und das Skript fährt mit der nächsten Datei fort.

Hinweise

    Es ist nicht so gut.    

    Abhängigkeiten: Das Skript verwendet die Bibliotheken PyMuPDF für die Textextraktion und PDFTableExtractor für die Tabellenextraktion.

    Pfadanpassung: Passen Sie die Variablen pdf_folder und output_dir an Ihre spezifischen Eingabe- und Ausgabeverzeichnisse an.

    Logging: Das Skript erstellt eine Log-Datei namens contriever_process.log, in der Informationen und Fehler während der Verarbeitung festgehalten werden.

Dieses Skript bietet eine effiziente Methode zur Extraktion und Speicherung von Text- und Tabellendaten aus PDF-Dokumenten für weitere Analysen oder Verarbeitungen.

## PDF to img to Markdown
### pdf_to_image_3.py

###### Output: output_images/* && console

Dieses Skript ermöglicht die Umwandlung von PDF-Dokumenten in das Markdown-Format, indem es jede Seite des PDFs als Bild 
rendert und anschliessend ein Sprachmodell zur Texterkennung und -konvertierung verwendet.
Funktionsweise

    Initialisierung: Das Skript lädt das spezifizierte Sprachmodell und legt Parameter wie Temperatur, maximale Tokenanzahl, 
    Timeout und maximale Anzahl von Wiederholungsversuchen fest.

    Bildextraktion: Für jede Seite des PDF-Dokuments wird ein Bild erstellt, das als PNG-Datei gespeichert und 
    zusätzlich in einen Base64-kodierten String umgewandelt wird.

    Texterkennung und Konvertierung: Das Sprachmodell erhält die Base64-kodierten Bilder und wandelt den erkannten Text in Markdown um.

    Ergebnis: Die generierten Markdown-Inhalte werden für jede Seite des PDFs gespeichert und können weiterverwendet werden.

Hinweise

    Abhängigkeiten: Das Skript verwendet Bibliotheken wie PyMuPDF für die PDF-Verarbeitung und langchain_google_genai für die Integration des Sprachmodells.

    API-Schlüssel: Stellen Sie sicher, dass der GOOGLE_API_KEY in den Umgebungsvariablen gesetzt ist oder beim Ausführen des Skripts eingegeben wird.

    Fehlerbehebung: Bei der Warnung "grpc_wait_for_shutdown_with_timeout() timed out." kann ein Downgrade von grpcio auf Version 1.60.1 hilfreich sein:

    pip install grpcio==1.60.1

Dieses Skript bietet eine effiziente Methode, um PDF-Inhalte in das Markdown-Format zu überführen, indem es die Stärken moderner Sprachmodelle nutzt.

## Unstructured Document to Markdown
### O3_unstructured.py

###### Output: src/Unstructured/*

Dieses Skript automatisiert die Konvertierung verschiedener Dateiformate 
wie PDF, DOCX, HTML, TXT und PPTX in das Markdown-Format. 
Es durchsucht ein angegebenes Eingabeverzeichnis nach unterstützten Dateien, 
verarbeitet deren Inhalte und speichert die resultierenden Markdown-Dateien im Ausgabeverzeichnis.

Funktionsweise

    Verzeichniseinrichtung: Das Skript stellt sicher, dass das Ausgabeverzeichnis existiert, und erstellt es bei Bedarf.

    Dateiverarbeitung: Für jede Datei im Eingabeverzeichnis mit einer der unterstützten Erweiterungen wird Folgendes durchgeführt:

        Dateityp-Erkennung: Anhand der Dateierweiterung wird der entsprechende Verarbeitungsprozess ausgewählt.

        Inhaltsaufbereitung: Mit Hilfe der unstructured-Bibliothek werden die Inhalte der Datei in verarbeitbare Elemente zerlegt.

        Markdown-Konvertierung: Die extrahierten Elemente werden in Markdown umgewandelt. 
        Tabellen werden dabei speziell behandelt, indem sie von HTML in Markdown-Tabellen konvertiert werden.

        Speicherung: Der resultierende Markdown-Text wird in einer .md-Datei im Ausgabeverzeichnis gespeichert, 
        wobei der Dateiname dem der ursprünglichen Datei entspricht.

    Fehlerbehandlung: Tritt während der Verarbeitung einer Datei ein Fehler auf, wird dieser protokolliert, 
    und das Skript setzt die Verarbeitung mit der nächsten Datei fort.

Hinweise

    Abhängigkeiten: Das Skript verwendet die unstructured-Bibliothek zur Datei-Partitionierung und BeautifulSoup zur HTML-Verarbeitung. 
    Stellen Sie sicher, dass alle erforderlichen Pakete installiert sind.

    Eingabe- und Ausgabepfade: Passen Sie die Variablen INPUT_FOLDER und OUTPUT_FOLDER an 
    die gewünschten Verzeichnisse für die Eingabedateien und die Ausgabe-Markdown-Dateien an.

    Unterstützte Dateiformate: Das Skript unterstützt die Formate PDF, DOCX, HTML/HTM, TXT und PPTX. 
    Für nicht erkannte Dateitypen wird ein automatischer Partitionierungsansatz verwendet.

Dieses Skript bietet eine effiziente Methode zur Stapelverarbeitung und Konvertierung verschiedener Dateiformate in das 
Markdown-Format, was insbesondere für die Erstellung von Dokumentationen und die Datenaufbereitung nützlich ist.

## Markitdown PDF to Markdown
### pdf_to_md_converter.py

###### Output: src/md_versions/*

Dieses Skript automatisiert die Umwandlung von PDF-Dateien in das Markdown-Format mit Markitdown. 
Es durchsucht das Verzeichnis Docs nach PDF-Dateien, konvertiert jede gefundene Datei und speichert die resultierenden 
Markdown-Dateien im Verzeichnis src/md_versions.

Funktionsweise

    Initialisierung: Das Skript erstellt eine Instanz von MarkItDown, 
    einer Python-Bibliothek zur Konvertierung verschiedener Dateiformate in Markdown.
    github.com

    Verzeichniseinrichtung: Es stellt sicher, dass das Ausgabeverzeichnis src/md_versions existiert, und erstellt es bei Bedarf.

    Dateisuche: Im Verzeichnis Docs werden alle PDF-Dateien ermittelt.

    Konvertierung: Für jede gefundene PDF-Datei wird:

        Der Dateiname für die resultierende Markdown-Datei festgelegt.

        Die PDF-Datei mit MarkItDown in Markdown konvertiert.

        Der konvertierte Inhalt in einer .md-Datei im Ausgabeverzeichnis gespeichert.

    Fehlerbehandlung: Tritt während der Konvertierung oder Speicherung ein Fehler auf, 
    wird eine entsprechende Fehlermeldung ausgegeben, und das Skript fährt mit der nächsten Datei fort.

Hinweise

    Abhängigkeiten: Stellen Sie sicher, dass die MarkItDown-Bibliothek installiert ist. Sie kann über PyPI installiert werden:

    pip install markitdown

    Pfadanpassung: Passen Sie die Pfade docs_dir und md_dir im Skript an Ihre spezifischen Verzeichnisse an, falls diese von den Standardwerten abweichen.

    Speicherung: Die resultierenden Markdown-Dateien werden im Verzeichnis src/md_versions gespeichert.

Dieses Skript bietet eine effiziente Methode zur Stapelverarbeitung von PDF-Dateien und deren Umwandlung in das 
Markdown-Format, was insbesondere für die Erstellung von Dokumentationen und die Datenaufbereitung nützlich ist.

## Mark any down
### mark_any_down.py

###### Output: console

Schreibt Markdown Output von beliebigen File in die Konsole.

Hinweis:

    Am besten für Docx und Pptx Dokumente von den nicht LLM varianten.