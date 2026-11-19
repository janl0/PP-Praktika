# Verbindung und Ausführung der Labs auf dem Lichtenberg
## Verbindung per SSH

Um sich per SSH mit dem Lichtenberg-Cluster zu verbinden gebt ihr im Terminal eurer Wahl zunächst folgendes ein:

`
ssh <tu-id>@<login-knoten>
`

Dabei ersetzt ihr `<tu-id>` mit eurer TU-ID und `<login-knoten>` mit einem der Loginknoten, die ihr [hier](https://www.hrz.tu-darmstadt.de/hlr/betrieb_hlr/hardware_hlr_1/aktuell_verfuegbar_hlr/index.de.jsp) findet (am besten einen ungeraden wählen).
Es kann sein, dass ihr den Knoten noch als vertrauenswürdigen Host hinzufügen müsst. 

Sollte die Verbindung erfolgreich sein, werdet ihr nach eurem Passwort und OTP gefragt.

## Auf dem Lichtenberg
Nach erfolgreichem Login solltet ihr Folgendes sehen:

`
[<tu-id>@<login-knoten-id> ~]$
`

Hierbei bedeutet `~`, dass ihr euch in eurem Home-Verzeichnis befindet. 

**IM HOME-VERZEICHNIS SOLLTET IHR KEINE PROJEKT-DATEIEN ABLEGEN, DA SICH ALLE PP-TEILNEHMER DORT NUR 60 GB TEILEN**

Die Aufgaben für die Praktika legt ihr am besten ins Scratch-Verzeichnis. Dieses erreicht ihr mit `cd $HPC_SCRATCH`.

Danach solltet ihr Folgendes sehen: `[<tu-id>@<login-knoten-id> <tu-id>]$`.

## Dateien aufs Lichtenberg spielen

Um Dateien der Praktika aufs Lichtenberg zu spielen, nutzt ihr am besten `scp` in einem neuen Terminal. Eine Einspielung sähe wie folgt aus:

`
scp -r <pfad-zu-eurem-lokalen-projekt-git-ordner> <tu-id>@<login-knoten>:/work/scratch/kurse/kurs00107/<tu-id>/<projekt-ordner>
`

Nach Ersetzen der Platzhalter werdet ihr wieder nach Passwort und OTP gefragt.

Wenn ihr schon Projektdateien auf dem Lichtenberg habt, wird im Discord empfohlen mit `rm -rf <projekt-ordner>` diese vor dem erneuten Einspielen zu entfernen, um Probleme zu vermeiden.

## Ausführen auf dem Lichtenberg

Wenn ihr euch nun auf dem Lichtenberg einloggt, könnt ihr mit `cd $HPC_SCRATCH/<projekt-ordner>` in den Projekt-Ordner wechseln und die Praktika ausführen. (Die Befehle vom Arbeitsblatt)

## Probleme auf dem Lichtenberg

Solltet ihr beim Ausführen von Tests Probleme mit Berechtigungen haben, müsst ihr diese mit `chmod +x <pfad-zum-test>` erweitern.
