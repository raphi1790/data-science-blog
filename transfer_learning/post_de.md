# Transfer-Learning in Forecasting
## How Transfer-Learning can help with shifts in Time-Series

Verschiebungen in den Daten können jederzeit auftreten. Oftmals führen sie dazu, dass Zeitreihen-Modelle von heute auf morgen keine präzsien Vorhersagen mehr treffen.

Um diese Problematik besser zu verstehen, schauen wir uns die folgende Situation im Detail an (welche bei Adtrac aufgetreten ist)

Das Unternehmen, für das du arbeitest, sammelt Daten zu Personenflüssen an spezifischen Stellen. Konkret werden für jede einzelne Person, die vorbeigeht, das Alter sowie das Geschlecht anhand eines Computer Vision Algorithmus bestimmt und abgespeichert. 
Dein Job als Data Scientist ist es diese seit Jahren gesammelten Daten zu nutzen und ein Modell zu entwickeln, das  die Personenflüssen für jede Alters-Geschlechts-Gruppe für die Zukunft vorhersagt; also ein Zeitreihen-Vorhersage für alle Alters-Geschlechts-Gruppen einzeln.
Als erfahrener Data Scientist erfüllst du diese Aufagbe mifhilfe eines Neuronalen Netzwerken, wobei du mit Zufriedenheit feststellts, dass das Modell auch die jährlichen Saisonalitäten abbildet. Da alles in Ordnung ist, wird dieses Vorhersage-Modell in Produktion genommen.

Nach einiger Zeit stellts du fest, dass die vorhergesagte Alters-Geschlechts-Verteilung nicht mehr der Realität entspricht. Deine Nachforschungen ergeben, dass diese Verwerfungen mit der Aufhebung der Corona-Maskenpflicht zusammenhängen. Es scheint, dass der Computer-Vision-Algorithmus, die Personen anders einschätzt wenn keine Maske getragen wird. Er erkennt zwar immer noch die gleiche Anzahl an Personen im Total, aber ihre Verteilung auf einzelne Gruppen unterscheidet sich zu vorher. 
Dein Auftrag ist es nun die Vorhersagen an die neue Situation anzupassen. Unglücklicherweise, gibt es erst ein paar Wochen an neuen Daten und du fragst dich, wie du damit die jährliche Saisonalität abbilden kannst.

Was wie ein Rätsel tönt ist in tat und wahrheit genau so bei Adtrac aufgetreten. Dies unterstreicht, dass diese Szenario keineswegs fiktional ist, sondern tatsächlich in der Realtität auftritt. Die Frage bleibt allerdings, wie können wir damit umgehen?

Wie immer gibt es verschiedene Ansätze. So So kann man die "alten" Daten an die neue Situation anpassen, oder den Output des bestehenden Modells mit einer Heuristik überschreiben.
Hier wollen wir uns aber auf einen Transfer-Learning-Ansatz fokussieren, der noch nicht breit bekannt ist und trotzdem die gesetzen Ziele erfüllt hat, die da wären:
-   Alters-Geschlechtsverteilung passt sich den neuen Daten an
-   Jährliche Saisonalität wird ähnlich vorhergesagt wie beim bestehenden Modell


### Was ist Transfer-Learning
![alt text](assets/transfer_learning.png "example of how transfer learning works. Source: [kdnuggets.com](https://www.kdnuggets.com/2017/09/databricks-vision-making-deep-learning-simple.html)")
Transfer-Learning bedeutet im Grunde, dass Wissen aus einem wohlbekannten Bereich auf ein Gebiet übertragen wird, von dem man noch nicht sehr viele Daten hat. 
Im Bereich des Maschine-Learning wird diese Technik häufig für Klassifizierungsproblmene im Computer-Vision-Bereich eingesetzt. 
Möchte man z.B. ein Modell trainieren, das verschiedene Hunderassen unterscheiden kann, so startet man am besten mit einem allgemeinen Objekterkennungsmodell (z.B. ResNet50). Diese grossen Modelle können bereits gut ein Hund von einer Katze oder einem Pferd unterscheiden. Um diesem Modell nun aber die verschiedenen Hunderassen anzutrainieren, benötigt man lediglich ein paar Bilder der verschiedenen Rassen. Mit diesen trainiert man die letzten Layers des Basismodells neu. Die ersten Layer des Basismodells bleiben dabei unangetastet und dienen lediglich dazu, den Input in einzelne Features zu zerlegen. Nach diesem Training erhält man hoffentlich ein Modell das einem Hundebild, die entsprechende Rasse zuordnen kann.
Der grösste Vorteil dieser Methode ist, dass man mit ungleich weniger Trainingsdaten auskommt, da das Basismodell bereits viele Features enthält und z.B. weiss, wie sich ein Tier von einem Menschen unterscheidet.

Gibt es vielleicht eine Möglichkeit unser Zeitreihen-Problem mithilfe dieser Methode zu lösen? Die kurze Antwort ist: Ja, wie wir bei Adtrac zeigen konnten.

### Anwendung von Transfer-Learning auf Zeitreihen
Obschon sich unser Problem von einem Klassifizierungsproblm unterscheidet, so haben wir analog der Hunderasse-Klassifzierung auch nur wenige Daten zur Verfügung, welche für sich alleine nicht ausreichen um ein Neuronales Netz von Grund auf zu trainieren. 
Würden wir das mit unserer Zeitreihe versuchen, so würden wir unweigerlich die jährliche Saisonalität verlieren, da nur ein beschränkter Zeitraum in den Trainingsdaten vorhanden ist.
Wir wissen allerdings, dass unser bestehendes Modell diese Saisonalitäten exakt vorhersagen kann. Wir können nun versuchen, diese Modell als Basismodell zu verwenden, und mit den neuen Daten (welche eine geänderte Verteilung aufweisen) nachzutrainieren.

In Tensorflow kann das mit wenigen Zeilen Code bewerkstelligt werden:

Um zu analysieren, ob wir unser Ziel erreicht haben, schauen wir zuerst auf die Alters-Geschlechts-Verteilung. Wie das Chart zeigt, wurde diese vollständig von den neuen Daten übernommen. Somit ist unser erstes Ziel erfüllt. Wie sieht es nun mit der Saisonalität aus. Unsere Hoffnung war, dass sich die langfristigen Zeitreihen-Features vom Basismodell übernommen wird. Das Chart beweist nun, dass diese saisonalen Effekte auch im neuen Modell vorhanden sind. Folglich wurde auch das zweite gesetzte Ziel erfüllt.

Zu erwähnen bleibt, dass trotz dieses Erfolgs kein Patentrezept gefunden wurde, welches erklärt wie man Transfer Learning am Besten anwendet. Unsere Erfahrung zeigt vielmehr, dass es für jedes Problem einzel evaluiert werden muss, wie viele Layers nachtrainiert werden,...etc.


### Schlussfolgerung
Gerade bei Zeitreihen, welche eine längere Saisonalität aufweisen, können Änderungen an den Labels aufgrund von verschiedensten Faktoren (Aufhebung Coronamassnahmen, Software-Update,...) grosse Folgen haben, denn das Trainieren von solchen Modellen benötigt mindesten die Daten aus einem kompletten Zyklus. Oft möchte man aber nicht darauf warten, dass genug neue Daten vorhanden sind, sondern möchte das bestehende Modell einfach an die neue Begebenheiten anpassen. Als ein möglicher Ansatz kommt hier das Transfer-Learning ins Spiel, welches sich bereits in Klassifikationsproblemen als vielversprechnede Methode herausgestellt hat. Angewendet auf Zeitreihen-Probleme kann diese Methode eine Möglichkeit darstellen, schneller als ein Saisonzyklus ein Update in Produktion zu bringen. Dies haben wir bei Adtrac unter Beweis gestellt. 