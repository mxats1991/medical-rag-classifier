"""
Medical RAG Classifier for Radiology Reports
Automated quality control system for AI services in radiology and fluorography
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Tuple, Dict, Optional
import warnings
import os
from tqdm import tqdm


class MedicalRAGClassifier:
    """
    Retrieval-Augmented Generation system for medical text classification.
    
    This system combines semantic search (Sentence Transformers), lexical search (BM25),
    and generative language models (DialoGPT) to classify radiology reports as 
    "pathology" or "normal" based on established medical criteria.
    
    Attributes:
        embedding_model: SentenceTransformer for semantic embeddings
        tokenizer: Tokenizer for the language model
        model: Generative language model for classification
        training_texts: List of training medical texts
        training_labels: List of corresponding labels
        training_embeddings: Precomputed embeddings for training texts
        bm25: BM25 search index
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Medical RAG Classifier.
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.setup_environment()
        self.setup_components()
        
    def setup_environment(self):
        """Configure environment variables for optimal performance."""
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if self.device == 'cpu' else '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        warnings.filterwarnings('ignore')
        
    def setup_components(self):
        """Initialize all system components."""
        print("Loading models...")
        try:
            # Load embedding model
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            print("Embedding model loaded")

            # Load language model
            print("Loading language model...")
            model_name = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Language model loaded: {model_name}")

            # Initialize training data storage
            self.training_texts = []
            self.training_labels = []
            self.training_embeddings = None

        except Exception as e:
            print(f"Critical error loading models: {e}")
            raise

    def load_training_data(self, excel_path: str, text_column: str = None, 
                          label_column: str = None) -> None:
        """
        Load training data from Excel file.
        
        Args:
            excel_path: Path to Excel file with training data
            text_column: Name of column containing medical texts
            label_column: Name of column containing labels (0=normal, 1=pathology)
        """
        print(f"Loading training data from {excel_path}...")

        try:
            df = pd.read_excel(excel_path)

            # Auto-detect columns if not specified
            if text_column is None:
                text_column = self._detect_text_column(df)
            if label_column is None:
                label_column = self._detect_label_column(df)

            texts = df[text_column].fillna('').astype(str).tolist()
            labels = df[label_column].fillna(0).astype(int).tolist()

            # Clean texts and store
            self.training_texts = [self.clean_text(text) for text in texts]
            self.training_labels = labels

            print(f"Using columns: text='{text_column}', labels='{label_column}'")
            print(f"Loaded {len(self.training_texts)} training examples")

            # Create embeddings and search index
            self._create_embeddings()
            self._create_search_index()

        except Exception as e:
            print(f"Error loading training data: {e}")
            raise

    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Auto-detect text column in DataFrame."""
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['заключение', 'исследование', 'текст', 'описание']):
                return col
        return df.columns[0]

    def _detect_label_column(self, df: pd.DataFrame) -> str:
        """Auto-detect label column in DataFrame."""
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['патология', 'класс', 'метка', 'label']):
                return col
        # Create default label column if none found
        if 'Pатология' not in df.columns:
            df['Pатология'] = 0
        return 'Pатология'

    def _create_embeddings(self):
        """Create embeddings for training texts."""
        if len(self.training_texts) == 0:
            return

        print("Creating embeddings...")
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(self.training_texts), batch_size), 
                     desc="Creating embeddings"):
            batch = self.training_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)

        self.training_embeddings = np.array(embeddings)
        print("Embeddings created")

    def _create_search_index(self):
        """Create BM25 search index."""
        if len(self.training_texts) == 0:
            return

        tokenized_texts = [self.tokenize(text) for text in self.training_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        print("Search index created")

    def clean_text(self, text: str) -> str:
        """
        Clean medical text by removing HTML tags and extra spaces.
        
        Args:
            text: Raw medical text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""

        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^<]+?>', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove medical formatting tags
        text = re.sub(r'<br>|</br>|<br/>', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 search.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.lower().split()

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        Perform hybrid semantic and lexical search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, label) tuples
        """
        if not self.training_texts:
            return []

        clean_query = self.clean_text(query)

        try:
            # Semantic search
            query_embedding = self.embedding_model.encode([clean_query])[0]
            dense_scores = cosine_similarity([query_embedding], self.training_embeddings)[0]

            # Lexical search
            tokenized_query = self.tokenize(clean_query)
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Normalize scores
            dense_scores_norm = self._normalize_scores(dense_scores)
            bm25_scores_norm = self._normalize_scores(bm25_scores)

            # Hybrid scoring (60% semantic, 40% lexical)
            hybrid_scores = 0.6 * dense_scores_norm + 0.4 * bm25_scores_norm

        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

        # Get top-k results
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        return [
            (self.training_texts[idx], hybrid_scores[idx], self.training_labels[idx])
            for idx in top_indices if idx < len(self.training_texts)
        ]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_val = scores.min()
        max_val = scores.max()
        return (scores - min_val) / (max_val - min_val + 1e-8)

    def classify(self, medical_text: str) -> Dict[str, any]:
        """
        Classify medical text as pathology or normal.
        
        Args:
            medical_text: Medical report text
            
        Returns:
            Dictionary with classification results
        """
        if not medical_text or medical_text.strip() == "":
            return {
                "verdict": "НОРМА",
                "confidence": 0.0,
                "explanation": "Empty medical text",
                "pathology": 0
            }

        clean_text = self.clean_text(medical_text)

        try:
            # Find similar examples
            similar_examples = self.hybrid_search(clean_text, top_k=3)
            
            # Classify with LLM
            result = self._classify_with_llm(clean_text, similar_examples)
            result["pathology"] = 1 if result["verdict"] == "ПАТОЛОГИЯ" else 0
            
            return result
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "verdict": "НОРМА",
                "confidence": 0.5,
                "explanation": f"Classification error: {str(e)}",
                "pathology": 0
            }

    def _classify_with_llm(self, medical_text: str, 
                          similar_examples: List[Tuple]) -> Dict[str, any]:
        """
        Classify using LLM with RAG context.
        
        Args:
            medical_text: Medical text to classify
            similar_examples: Similar examples from search
            
        Returns:
            Classification results
        """
        prompt = self._build_classification_prompt(medical_text, similar_examples)

        # Tokenize and generate
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", 
                                     max_length=1024, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse response
        verdict = self._parse_verdict(response)
        explanation = self._parse_explanation(response)
        confidence = self._calculate_confidence(similar_examples, verdict)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "explanation": explanation
        }

    def _build_classification_prompt(self, medical_text: str, 
                                   similar_examples: List[Tuple]) -> str:
        """
        Build classification prompt with medical criteria and examples.
        
        Args:
            medical_text: Text to classify
            similar_examples: Similar examples for context
            
        Returns:
            Formatted prompt string
        """
        prompt = """Ты опытный врач-рентгенолог. Проанализируй медицинское заключение и определи, есть ли патология.

Критерии патологии:
- ПНЕВМОНИЯ, ИНФИЛЬТРАЦИЯ, ОЧАГИ
- ОБРАЗОВАНИЕ
- ГИДРОТОРАКС, ПНЕВМОТОРАКС
- АТЕЛЕКТАЗ, АБСЦЕСС, КАВЕРНА
- ВОСПАЛИТЕЛЬНЫЕ ИЗМЕНЕНИЯ

Критерии нормы:
- "ПАТОЛОГИЧЕСКИХ ИЗМЕНЕНИЙ НЕ ВЫЯВЛЕНО"
- "ОЧАГОВЫЕ И ИНФИЛЬТРАТИВНЫЕ ИЗМЕНЕНИЯ НЕ ВЫЯВЛЕНЫ"
- "БЕЗ СВЕЖИХ ОЧАГОВЫХ ИЗМЕНЕНИЙ"
- СТАРЫЕ РУБЦЫ, КАЛЬЦИНАТЫ БЕЗ АКТИВНЫХ ИЗМЕНЕНИЙ

Примеры классификации из базы знаний:
"""

        # Add similar examples
        for i, (example_text, score, label) in enumerate(similar_examples):
            status = "ПАТОЛОГИЯ" if label == 1 else "НОРМА"
            short_text = example_text[:120] + "..." if len(example_text) > 120 else example_text
            prompt += f"\n{i + 1}. {short_text} → {status}"

        prompt += f"""

ЗАКЛЮЧЕНИЕ ДЛЯ АНАЛИЗА: {medical_text}

Проанализируй заключение на основе приведенных примеров и критериев.

Ответь строго в формате:
ВЕРДИКТ: [ПАТОЛОГИЯ/НОРМА]
ОБОСНОВАНИЕ: [краткое объяснение на основе медицинских критериев]

Ответ:"""

        return prompt

    def _parse_verdict(self, response: str) -> str:
        """Extract verdict from LLM response."""
        response_upper = response.upper()
        
        if "ВЕРДИКТ: ПАТОЛОГИЯ" in response_upper:
            return "ПАТОЛОГИЯ"
        elif "ВЕРДИКТ: НОРМА" in response_upper:
            return "НОРМА"
        elif "ПАТОЛОГИЯ" in response_upper and "НОРМА" not in response_upper:
            return "ПАТОЛОГИЯ"
        else:
            return "НОРМА"  # Conservative default

    def _parse_explanation(self, response: str) -> str:
        """Extract explanation from LLM response."""
        if "ОБОСНОВАНИЕ:" in response.upper():
            parts = response.upper().split("ОБОСНОВАНИЕ:")
            if len(parts) > 1:
                return parts[1].strip()
        return response.strip()

    def _calculate_confidence(self, similar_examples: List[Tuple], 
                            verdict: str) -> float:
        """Calculate confidence based on similar examples."""
        if not similar_examples:
            return 0.5

        target_label = 1 if verdict == "ПАТОЛОГИЯ" else 0
        total_weight = sum(score for _, score, _ in similar_examples)
        matched_weight = sum(score for _, score, label in similar_examples 
                           if label == target_label)

        confidence = matched_weight / total_weight if total_weight > 0 else 0.5

        # Boost confidence for strong matches
        max_score = max(score for _, score, _ in similar_examples)
        if max_score > 0.8:
            confidence = min(confidence * 1.2, 0.95)

        return confidence

    def process_dataset(self, input_path: str, output_path: str = None,
                       text_column: str = None) -> pd.DataFrame:
        """
        Process entire dataset and add classification columns.
        
        Args:
            input_path: Path to input Excel file
            output_path: Path for output file (optional)
            text_column: Name of text column (auto-detected if None)
            
        Returns:
            DataFrame with classification results
        """
        print(f"Processing file: {input_path}")

        try:
            df = pd.read_excel(input_path)

            # Auto-detect text column
            if text_column is None:
                text_column = self._detect_text_column(df)

            print(f"Processing {len(df)} records using column: {text_column}")

            # Add classification columns
            classification_columns = ['RAG_Вердикт', 'RAG_Патология', 
                                    'RAG_Уверенность', 'RAG_Обоснование']
            
            for col in classification_columns:
                df[col] = ""

            # Classify each record
            successful = 0
            for idx in tqdm(range(len(df)), desc="Classifying records"):
                if idx % 10 == 0 and idx > 0:
                    print(f"Processed {idx}/{len(df)} records...")

                medical_text = str(df.iloc[idx][text_column])
                result = self.classify(medical_text)

                # Store results
                df.at[idx, 'RAG_Вердикт'] = result['verdict']
                df.at[idx, 'RAG_Патология'] = result['pathology']
                df.at[idx, 'RAG_Уверенность'] = result['confidence']
                df.at[idx, 'RAG_Обоснование'] = result['explanation']

                if result['confidence'] > 0.5:
                    successful += 1

            # Save results
            if output_path is None:
                base_name = os.path.splitext(input_path)[0]
                output_path = f"{base_name}_classified.xlsx"

            df.to_excel(output_path, index=False)
            print(f"Results saved to: {output_path}")

            # Print statistics
            self._print_statistics(df, successful)
            
            return df

        except Exception as e:
            print(f"Error processing dataset: {e}")
            raise

    def _print_statistics(self, df: pd.DataFrame, successful: int):
        """Print classification statistics."""
        pathology_count = len(df[df['RAG_Патология'] == 1])
        normal_count = len(df[df['RAG_Патология'] == 0])
        avg_confidence = df['RAG_Уверенность'].mean()

        print(f"\nClassification Statistics:")
        print(f"   • Pathology: {pathology_count} records")
        print(f"   • Normal: {normal_count} records")
        print(f"   • Successful classifications: {successful}/{len(df)}")
        print(f"   • Average confidence: {avg_confidence:.3f}")
        print(f"   • Total records: {len(df)}")


def main():
    """Example usage of the Medical RAG Classifier."""
    # Initialize classifier
    classifier = MedicalRAGClassifier()

    # Load training data
    classifier.load_training_data("training_data.xlsx")

    # Process new dataset
    result_df = classifier.process_dataset("input_data.xlsx")

    # Show sample results
    if result_df is not None:
        print("\nSample Results:")
        sample_cols = [col for col in result_df.columns if 'RAG' in col or 'заключение' in col.lower()]
        print(result_df[sample_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
