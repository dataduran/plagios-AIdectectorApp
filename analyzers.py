# analyzers.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM



class PlagiarismAnalyzer:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

    def analyze_documents(self, documents: Dict[str, str]) -> Dict:
        """
        Analyze multiple documents for plagiarism using TF-IDF and cosine similarity.
        """
        if len(documents) < 2:
            return {
                "similarity_scores": {},
                "detailed_matches": {},
                "warning": "At least two documents are required for comparison"
            }

        # Prepare document texts and file paths
        texts = list(documents.values())
        file_paths = list(documents.keys())

        # Calculate TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate pairwise similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Prepare results
        similarity_scores = {}
        detailed_matches = {}

        # Analyze each pair of documents
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                file1 = file_paths[i]
                file2 = file_paths[j]
                similarity = similarity_matrix[i][j]

                pair_key = f"{file1} vs {file2}"
                similarity_scores[pair_key] = float(similarity)

                if similarity > self.threshold:
                    detailed_matches[pair_key] = self._find_similar_passages(
                        texts[i], texts[j]
                    )

        return {
            "similarity_scores": similarity_scores,
            "detailed_matches": detailed_matches
        }

    def _find_similar_passages(self, text1: str, text2: str) -> List[Dict]:
        """
        Find similar passages between two texts using sentence-level comparison.
        """
        sentences1 = self._split_into_sentences(text1)
        sentences2 = self._split_into_sentences(text2)
        
        matches = []
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Compare each sentence from text1 with each sentence from text2
        for i, sent1 in enumerate(sentences1):
            if len(sent1.split()) < 5:  # Skip very short sentences
                continue
                
            tfidf_matrix = vectorizer.fit_transform([sent1] + sentences2)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            for j, score in enumerate(similarity_scores[0]):
                if score > self.threshold:
                    matches.append({
                        "text1_sentence": sent1,
                        "text2_sentence": sentences2[j],
                        "similarity": float(score)
                    })
        
        return matches

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class AIContentAnalyzer:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.perplexity_threshold = 0.85
        self.burstiness_threshold = 0.8

        if model_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        elif model_name == "deepseek":
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek")
            self.model = AutoModelForCausalLM.from_pretrained("deepseek")
        elif model_name == "gemma2":
            self.tokenizer = AutoTokenizer.from_pretrained("gemma2")
            self.model = AutoModelForCausalLM.from_pretrained("gemma2")
        elif model_name == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained("llama3")
            self.model = AutoModelForCausalLM.from_pretrained("llama3")
        elif model_name == "mixtral":
            self.tokenizer = AutoTokenizer.from_pretrained("mixtral")
            self.model = AutoModelForCausalLM.from_pretrained("mixtral")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def analyze_text(self, text: str) -> float:
        """
        Analyze text for potential AI-generated content.
        Returns a probability score between 0 and 1.
        """
        # Calculate various metrics
        perplexity_score = self._calculate_perplexity(text)
        burstiness_score = self._calculate_burstiness(text)
        
        # Combine scores with weights
        weighted_score = (0.6 * perplexity_score + 0.4 * burstiness_score)
        
        return min(1.0, max(0.0, weighted_score))

    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity score based on word predictability patterns.
        Higher scores indicate more likely AI-generated content.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        # Normalize perplexity to a 0-1 range
        normalized_perplexity = min(1.0, perplexity / 100)
        return normalized_perplexity

    def _calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness score based on word distribution patterns.
        Higher scores indicate more likely AI-generated content.
        """
        words = text.split()
        if not words:
            return 0.0

        # Analyze word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Calculate distribution uniformity
        frequencies = list(word_freq.values())
        if not frequencies:
            return 0.0

        # Calculate coefficient of variation
        mean = np.mean(frequencies)
        std = np.std(frequencies)
        cv = std / mean if mean > 0 else 0

        # Normalize to 0-1 range
        score = 1.0 - min(1.0, cv)
        return score

    def _check_repetitive_patterns(self, words: List[str]) -> float:
        """Check for unusually repetitive patterns in text."""
        if len(words) < 2:
            return 0.0

        repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetitions += 1

        return min(1.0, repetitions / (len(words) - 1))

    def _check_unusual_combinations(self, words: List[str]) -> float:
        """Check for unusual word combinations that might indicate AI generation."""
        if len(words) < 3:
            return 0.0

        # Simple analysis of trigram transitions
        unusual_count = 0
        for i in range(len(words) - 2):
            # This is a simplified check. In a real implementation,
            # you would use a proper language model or frequency database
            trigram = ' '.join(words[i:i+3]).lower()
            if len(trigram) > 50:  # Arbitrary length check
                unusual_count += 1

        return min(1.0, unusual_count / (len(words) - 2))


class AITeacherAnalyzer:
    def __init__(self, training_texts: List[str]):
        self.vectorizer = TfidfVectorizer()
        self.reference_text = " ".join(training_texts)

    def evaluate_exercise(self, text: str) -> float:
        vectors = self.vectorizer.fit_transform([text, self.reference_text]).toarray()
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0][1]  # Return similarity score

    def compare_pdfs(self, documents: Dict[str, str]) -> Dict[str, float]:
        scores = {}
        for pdf_path, text in documents.items():
            score = self.evaluate_exercise(text)
            scores[pdf_path] = score
        return scores

    def assign_best_qualification(self, scores: Dict[str, float]) -> Tuple[str, float]:
        best_pdf = max(scores, key=scores.get)
        best_score = scores[best_pdf]
        return best_pdf, best_score