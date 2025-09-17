from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentences = [
    "Comprei o tamanho P e serviu perfeitamente.",
    "Achei o zíper um pouco difícil de usar.",
    "A cor e o material são lindos pessoalmente.",
    "Incrívelmente confortável!"
]

# Faz análise
for s in sentences:
    result = classifier(s)[0]
    print(f"Frase: {s}")
    print(f"Sentimento: {result['label']} (score={result['score']:.2f})")
    print("---")
