# RAG-Studienarbeit

## PDF-Prozessor für die Vorbereitung von Contriever
### contriver_process.py

Dieses Skript dient der Verarbeitung von PDF-Dokumenten, um deren Inhalte für die weitere Nutzung im Contriever-System aufzubereiten. Es extrahiert den Text aus PDF-Dateien, entfernt wiederholte Kopf- und Fußzeilen und teilt den bereinigten Text in satzbasierte Abschnitte (Chunks) auf.
Funktionsweise

    Textextraktion: Das Skript verwendet die Bibliothek PyMuPDF, um den Text aus den Seiten der PDF-Dokumente zu extrahieren.

    Erkennung von Kopf- und Fußzeilen: Durch die Analyse der ersten und letzten Zeilen jeder Seite identifiziert das Skript wiederholte Muster, die als Kopf- oder Fußzeilen fungieren. Diese werden basierend auf einem festgelegten Schwellenwert als solche erkannt und können entfernt werden.

    Textbereinigung: Nach der Extraktion wird der Text bereinigt, indem überflüssige Leerzeichen entfernt, Zeilenumbrüche korrigiert und erkannte Kopf- und Fußzeilen eliminiert werden.

    Satzbasierte Aufteilung: Unter Verwendung des spaCy-Sprachmodells wird der bereinigte Text in einzelne Sätze segmentiert. Diese Sätze werden anschließend zu Abschnitten (Chunks) zusammengefasst, wobei die Wortanzahl pro Chunk durch definierte Parameter gesteuert wird.

    Speicherung der Ergebnisse: Die erzeugten Chunks werden mit Metadaten wie der Quell-PDF, den Seitenzahlen und einer eindeutigen Chunk-ID versehen und in einer JSON-Datei gespeichert.

Anpassbare Parameter

    DEFAULT_MAX_WORDS: Maximale Anzahl von Wörtern pro Chunk (Standardwert: 300).

    MIN_CHUNK_SIZE: Minimale Wortanzahl für den letzten Chunk, um zu kleine Abschnitte zu vermeiden (Standardwert: 100).

    HEADER_FOOTER_THRESHOLD: Prozentsatz der Seiten, auf denen ein Muster auftreten muss, um als Kopf- oder Fußzeile erkannt zu werden (Standardwert: 0,75).

Diese Parameter können direkt im Skript an die spezifischen Anforderungen angepasst werden.
Fehlerbehandlung

Das Skript ist mit umfassenden Fehlerbehandlungsmechanismen ausgestattet, die Probleme bei der PDF-Verarbeitung erkennen und protokollieren. Fehler werden im Logfile contriever_process.log festgehalten, während die Verarbeitung der verbleibenden Dateien fortgesetzt wird, um eine möglichst vollständige Datenextraktion sicherzustellen.

Durch diese strukturierte Vorgehensweise stellt das Skript sicher, dass die aus den PDF-Dokumenten extrahierten Texte optimal für die weitere Verarbeitung im Contriever-System vorbereitet sind.

## PDF to img to Markdown
### pdf_to_image_3.py

Dieses Skript ermöglicht die Umwandlung von PDF-Dokumenten in das Markdown-Format, indem es jede Seite des PDFs als Bild rendert und anschließend ein Sprachmodell zur Texterkennung und -konvertierung verwendet.
Funktionsweise

    Initialisierung: Das Skript lädt das spezifizierte Sprachmodell und legt Parameter wie Temperatur, maximale Tokenanzahl, Timeout und maximale Anzahl von Wiederholungsversuchen fest.

    Bildextraktion: Für jede Seite des PDF-Dokuments wird ein Bild erstellt, das als PNG-Datei gespeichert und zusätzlich in einen Base64-kodierten String umgewandelt wird.

    Texterkennung und Konvertierung: Das Sprachmodell erhält die Base64-kodierten Bilder und wandelt den erkannten Text in Markdown um.

    Ergebnis: Die generierten Markdown-Inhalte werden für jede Seite des PDFs gespeichert und können weiterverwendet werden.

Hinweise

    Abhängigkeiten: Das Skript verwendet Bibliotheken wie PyMuPDF für die PDF-Verarbeitung und langchain_google_genai für die Integration des Sprachmodells.

    API-Schlüssel: Stellen Sie sicher, dass der GOOGLE_API_KEY in den Umgebungsvariablen gesetzt ist oder beim Ausführen des Skripts eingegeben wird.

    Fehlerbehebung: Bei der Warnung "grpc_wait_for_shutdown_with_timeout() timed out." kann ein Downgrade von grpcio auf Version 1.60.1 hilfreich sein:

    pip install grpcio==1.60.1

Dieses Skript bietet eine effiziente Methode, um PDF-Inhalte in das Markdown-Format zu überführen, indem es die Stärken moderner Sprachmodelle nutzt.

## Unstructured Document to Markdown
### unstructured_converter.py

Dieses Skript dient der automatisierten Verarbeitung von PDF-Dokumenten. Es durchsucht ein angegebenes Eingabeverzeichnis nach PDF-Dateien, extrahiert deren Inhalte und speichert die Ergebnisse als Textdateien im Ausgabeverzeichnis.
Funktionsweise

    Verzeichniseinrichtung: Das Skript legt sicherheitshalber das Ausgabeverzeichnis an, falls es noch nicht existiert.

    PDF-Verarbeitung: Für jede PDF-Datei im Eingabeverzeichnis wird Folgendes durchgeführt:

        Extraktion von Elementen: Mit der Funktion partition_pdf aus der unstructured-Bibliothek werden verschiedene Elemente wie Text, Bilder und Tabellen aus der PDF extrahiert. Die strategy-Option ist auf "auto" gesetzt, was bedeutet, dass die Methode basierend auf den Dokumenteigenschaften automatisch gewählt wird. Die Option infer_table_structure ist aktiviert, um Tabellenstrukturen beizubehalten.

        Textzusammenführung: Die extrahierten Elemente werden zu einem zusammenhängenden Text kombiniert, wobei jedes Element durch zwei Zeilenumbrüche getrennt wird.

        Speicherung: Der kombinierte Text wird in einer .txt-Datei im Ausgabeverzeichnis gespeichert, wobei der Dateiname dem der ursprünglichen PDF entspricht.

    Fehlerbehandlung: Tritt während der Verarbeitung einer PDF-Datei ein Fehler auf, wird dieser abgefangen und eine entsprechende Fehlermeldung ausgegeben, während das Skript mit der nächsten Datei fortfährt.

Hinweise

    Abhängigkeiten: Dieses Skript verwendet die unstructured-Bibliothek zur PDF-Verarbeitung. Stellen Sie sicher, dass alle erforderlichen Pakete installiert sind.

    Eingabe- und Ausgabepfade: Passen Sie die Variablen INPUT_DIR und OUTPUT_DIR an die gewünschten Verzeichnisse für die Eingabe-PDFs und die Ausgabe-Texte an.

    Partitionierungsstrategie: Die strategy-Option in der Funktion partition_pdf ist auf "auto" gesetzt, was bedeutet, dass die Methode basierend auf den Dokumenteigenschaften automatisch gewählt wird. Je nach Bedarf kann diese Einstellung angepasst werden.

Dieses Skript bietet eine effiziente Methode zur Stapelverarbeitung von PDF-Dokumenten und zur Extraktion ihrer Inhalte in Textform.

