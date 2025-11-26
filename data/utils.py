from medical_rag_classifier import MedicalRAGClassifier

# Initialize classifier
classifier = MedicalRAGClassifier()

# Load training data
classifier.load_training_data("training_data.xlsx")

# Classify single text
result = classifier.classify("Рентгенологических признаков патологии не выявлено")
print(result)

# Process entire dataset
results_df = classifier.process_dataset("input_data.xlsx")
