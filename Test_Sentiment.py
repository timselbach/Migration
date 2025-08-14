# sentiment_de.py
from textblob_de import TextBlobDE as TextBlob

text = "Das heißt, Migration und Klimawandel werden mittlerweile unter sicherheitspolitischen Aspekten bewertet, obwohl es eigentlich globale Probleme sind, die ökonomischer und sozialer Natur sind."


blob = TextBlob(text)
polarity = blob.sentiment.polarity      # range: [-1.0, 1.0]
subjectivity = blob.sentiment.subjectivity  # range: [0.0, 1.0]

print(f"Text: {text}")
print(f"Polarity: {polarity:.3f}")
print(f"Subjectivity: {subjectivity:.3f}")

# If you want per-sentence sentiment:
for i, s in enumerate(blob.sentences, 1):
    print(f"Satz {i}: {s} -> {s.sentiment.polarity:.3f}")


