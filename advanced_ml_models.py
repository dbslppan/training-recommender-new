"""
Advanced ML Models for Training Recommender System
Includes: TF-IDF, Cosine Similarity, K-Means Clustering, NMF, Content-Based Filtering
Integration with Training Catalog Database
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
import json
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TextSimilarityEngine:
    """
    Engine untuk text similarity menggunakan TF-IDF dan Cosine Similarity
    Digunakan untuk matching learning objectives dengan competency gaps
    """
    
    def __init__(self, max_features=100, ngram_range=(1, 2)):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        self.fitted = False
        self.document_vectors = None
        self.document_ids = None
        
    def fit(self, documents: List[str], document_ids: List[str] = None):
        """
        Fit TF-IDF vectorizer pada corpus documents
        
        Args:
            documents: List of text documents (training objectives, descriptions)
            document_ids: List of document identifiers (training_ids)
        """
        # Clean and preprocess documents
        cleaned_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Fit and transform
        self.document_vectors = self.tfidf_vectorizer.fit_transform(cleaned_docs)
        self.document_ids = document_ids if document_ids else list(range(len(documents)))
        self.fitted = True
        
        return self
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text: lowercase, remove special chars, etc."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def calculate_similarity(self, query_text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Calculate cosine similarity between query and all documents
        
        Args:
            query_text: Query text (e.g., employee's competency gaps description)
            top_n: Number of top similar documents to return
            
        Returns:
            List of tuples (document_id, similarity_score)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Preprocess and vectorize query
        cleaned_query = self._preprocess_text(query_text)
        query_vector = self.tfidf_vectorizer.transform([cleaned_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = [
            (self.document_ids[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results
    
    def get_feature_importance(self, document_idx: int, top_n: int = 10):
        """Get most important features (keywords) for a document"""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        doc_vector = self.document_vectors[document_idx].toarray()[0]
        
        top_indices = np.argsort(doc_vector)[::-1][:top_n]
        
        return [(feature_names[idx], doc_vector[idx]) for idx in top_indices]


class CompetencyClusteringModel:
    """
    K-Means Clustering untuk mengelompokkan karyawan berdasarkan profil kompetensi
    Digunakan untuk group training recommendations
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        self.scaler = StandardScaler()
        self.cluster_profiles = None
        self.cluster_labels = None
        self.feature_names = None
        
    def fit(self, competency_df: pd.DataFrame) -> np.ndarray:
        """
        Fit K-Means clustering model
        
        Args:
            competency_df: DataFrame dengan competency scores
            
        Returns:
            Array of cluster labels
        """
        self.feature_names = competency_df.columns.tolist()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(competency_df)
        
        # Fit K-Means
        self.cluster_labels = self.kmeans.fit_predict(scaled_data)
        
        # Calculate cluster profiles (centroids in original scale)
        competency_df_copy = competency_df.copy()
        competency_df_copy['cluster'] = self.cluster_labels
        self.cluster_profiles = competency_df_copy.groupby('cluster').mean()
        
        return self.cluster_labels
    
    def predict_cluster(self, employee_competencies: np.ndarray) -> int:
        """Predict cluster for new employee"""
        scaled_data = self.scaler.transform(employee_competencies.reshape(1, -1))
        return self.kmeans.predict(scaled_data)[0]
    
    def get_cluster_characteristics(self, cluster_id: int) -> Dict:
        """
        Get detailed characteristics of a cluster
        
        Returns:
            Dictionary dengan strengths, weaknesses, avg scores
        """
        if self.cluster_profiles is None:
            raise ValueError("Model not fitted.")
        
        profile = self.cluster_profiles.loc[cluster_id]
        
        # Sort competencies
        sorted_comps = profile.sort_values(ascending=False)
        
        return {
            'cluster_id': cluster_id,
            'size': np.sum(self.cluster_labels == cluster_id),
            'average_profile': profile.to_dict(),
            'top_strengths': sorted_comps.head(5).to_dict(),
            'top_weaknesses': sorted_comps.tail(5).to_dict(),
            'overall_avg': profile.mean()
        }
    
    def find_optimal_clusters(self, competency_df: pd.DataFrame, 
                             max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method
        
        Returns:
            Dictionary with inertia scores for different k values
        """
        scaled_data = self.scaler.fit_transform(competency_df)
        
        inertias = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        return {
            'k_values': list(k_range),
            'inertias': inertias
        }


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering menggunakan Matrix Factorization (NMF)
    Menemukan pola tersembunyi dalam data employee-competency
    """
    
    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.nmf = NMF(
            n_components=n_components,
            init='random',
            random_state=random_state,
            max_iter=500,
        )
        self.user_factors = None
        self.item_factors = None
        self.scaler = StandardScaler()
        self.employee_ids = None
        self.competency_names = None
        
    def fit(self, competency_matrix: pd.DataFrame):
        """
        Fit NMF model
        
        Args:
            competency_matrix: DataFrame with index=employee_id, columns=competencies
        """
        self.employee_ids = competency_matrix.index.tolist()
        self.competency_names = competency_matrix.columns.tolist()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(competency_matrix)
        
        # Make non-negative (NMF requirement)
        scaled_data = scaled_data - scaled_data.min() + 0.1
        
        # Fit NMF
        self.user_factors = self.nmf.fit_transform(scaled_data)
        self.item_factors = self.nmf.components_
        
        return self
    
    def get_similar_employees(self, employee_idx: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Find employees with similar competency profiles
        
        Returns:
            List of (employee_index, similarity_score) tuples
        """
        if self.user_factors is None:
            raise ValueError("Model not fitted.")
        
        # Get employee vector
        emp_vector = self.user_factors[employee_idx].reshape(1, -1)
        
        # Calculate cosine similarity with all employees
        similarities = cosine_similarity(emp_vector, self.user_factors)[0]
        
        # Get top N (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def recommend_competency_focus(self, employee_idx: int, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend which competencies to focus on based on latent factors
        """
        if self.user_factors is None:
            raise ValueError("Model not fitted.")
        
        # Reconstruct competency scores
        reconstructed = np.dot(
            self.user_factors[employee_idx],
            self.item_factors
        )
        
        # Find gaps (low reconstructed scores suggest areas for improvement)
        gaps = 5.0 - reconstructed  # Assuming max score is 5.0
        
        # Get top competencies to improve
        top_indices = np.argsort(gaps)[::-1][:top_n]
        
        return [(self.competency_names[idx], gaps[idx]) for idx in top_indices]


class HybridRecommenderEngine:
    """
    Hybrid Recommendation Engine yang mengintegrasikan multiple algorithms:
    1. Content-Based Filtering (TF-IDF + Cosine Similarity)
    2. Collaborative Filtering (NMF)
    3. Competency Gap Analysis
    4. PESTLE Context Awareness
    5. Clustering-based Recommendations
    """
    
    def __init__(self, config: Dict = None):
        # Default weights
        self.weights = config or {
            'content_based': 0.30,      # TF-IDF similarity dengan learning objectives
            'competency_gap': 0.25,     # Gap analysis score
            'collaborative': 0.15,      # Collaborative filtering score
            'pestle': 0.15,            # PESTLE relevance
            'cluster_based': 0.10,      # Cluster-based recommendation
            'cost_efficiency': 0.05     # Cost vs impact ratio
        }
        
        # Sub-models
        self.text_engine = None
        self.cf_model = None
        self.clustering_model = None
        self.fitted = False
        
    def fit(self, employee_data: pd.DataFrame, training_catalog: pd.DataFrame):
        """
        Fit all sub-models
        
        Args:
            employee_data: Employee competency data
            training_catalog: Training catalog with objectives, costs, etc.
        """
        print("Fitting Hybrid Recommender Engine...")
        
        # 1. Fit Text Similarity Engine
        print("  - Fitting TF-IDF model...")
        training_texts = training_catalog.apply(
            lambda x: f"{x['training_name']} {x['learning_objectives']} {x['job_family']}", 
            axis=1
        ).tolist()
        
        self.text_engine = TextSimilarityEngine(max_features=200, ngram_range=(1, 3))
        self.text_engine.fit(training_texts, training_catalog['training_id'].tolist())
        
        # 2. Fit Collaborative Filtering
        print("  - Fitting Collaborative Filtering model...")
        competency_cols = [col for col in employee_data.columns 
                          if col.startswith(('core_', 'managerial_', 'leadership_'))]
        competency_matrix = employee_data[competency_cols].copy()
        competency_matrix.index = employee_data['employee_id']
        
        self.cf_model = CollaborativeFilteringRecommender(n_components=10)
        self.cf_model.fit(competency_matrix)
        
        # 3. Fit Clustering Model
        print("  - Fitting K-Means clustering...")
        self.clustering_model = CompetencyClusteringModel(n_clusters=5)
        employee_data['cluster'] = self.clustering_model.fit(competency_matrix)
        
        self.fitted = True
        print("✓ All models fitted successfully!")
        
        return self
    
    def generate_recommendations(self, 
                                employee_id: str,
                                employee_data: pd.DataFrame,
                                training_catalog: pd.DataFrame,
                                top_n: int = 10,
                                filters: Dict = None) -> List[Dict]:
        """
        Generate comprehensive recommendations using hybrid approach
        
        Args:
            employee_id: Target employee
            employee_data: All employee data
            training_catalog: Available trainings
            top_n: Number of recommendations
            filters: Optional filters (division, level, cost_max, etc.)
            
        Returns:
            List of recommendation dictionaries with scores
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get employee info
        employee = employee_data[employee_data['employee_id'] == employee_id].iloc[0]
        employee_idx = employee_data[employee_data['employee_id'] == employee_id].index[0]
        
        # Apply filters to training catalog
        filtered_catalog = self._apply_filters(training_catalog, employee, filters)
        
        if len(filtered_catalog) == 0:
            return []
        
        recommendations = []
        
        for _, training in filtered_catalog.iterrows():
            scores = self._calculate_all_scores(
                employee, employee_idx, training, employee_data
            )
            
            # Weighted final score
            final_score = sum(
                scores[key] * self.weights[key]
                for key in self.weights.keys()
            )
            
            recommendations.append({
                'training_id': training['training_id'],
                'training_name': training['training_name'],
                'school': training['school'],
                'duration_days': training['duration_days'],
                'cost': training['cost'],
                'job_family': training['job_family'],
                'learning_objectives': training['learning_objectives'],
                'target_division': training['target_division'],
                'target_level': training['target_level'],
                'final_score': final_score,
                'score_breakdown': scores,
                'priority': self._get_priority(final_score),
                'roi_estimate': self._estimate_roi(scores, training['cost'])
            })
        
        # Sort by final score
        recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _calculate_all_scores(self, employee, employee_idx, training, all_employees):
        """Calculate all component scores"""
        
        # 1. Content-Based Score (TF-IDF Similarity)
        employee_query = self._create_employee_query(employee)
        text_similarities = self.text_engine.calculate_similarity(employee_query, top_n=100)
        text_score = next((score for tid, score in text_similarities 
                          if tid == training['training_id']), 0.0)
        
        # 2. Competency Gap Score
        gap_score = self._calculate_competency_gap_score(employee, training)
        
        # 3. Collaborative Score
        collab_score = self._calculate_collaborative_score(employee_idx, training, all_employees)
        
        # 4. PESTLE Score
        pestle_score = self._calculate_pestle_score(employee, training)
        
        # 5. Cluster-Based Score
        cluster_score = self._calculate_cluster_score(employee, training, all_employees)
        
        # 6. Cost Efficiency Score
        cost_score = self._calculate_cost_efficiency(training, gap_score)
        
        return {
            'content_based': text_score * 5.0,  # Scale to 0-5
            'competency_gap': gap_score,
            'collaborative': collab_score,
            'pestle': pestle_score,
            'cluster_based': cluster_score,
            'cost_efficiency': cost_score
        }
    
    def _create_employee_query(self, employee) -> str:
        """Create search query from employee's weak competencies"""
        competency_cols = [col for col in employee.index 
                          if col.startswith(('core_', 'managerial_', 'leadership_'))]
        
        # Get weakest competencies
        weak_comps = []
        for comp in competency_cols:
            if employee[comp] < 3.5:  # Below satisfactory
                comp_name = comp.replace('_', ' ')
                weak_comps.append(comp_name)
        
        query = f"{employee['current_position']} {employee['division']} " + " ".join(weak_comps)
        return query
    
    def _calculate_competency_gap_score(self, employee, training) -> float:
        """Calculate how well training addresses competency gaps"""
        # Parse impacted competencies from PESTLE data or learning objectives
        # This is a simplified version
        competency_cols = [col for col in employee.index 
                          if col.startswith(('core_', 'managerial_', 'leadership_'))]
        
        # Calculate average gap
        gaps = [max(0, 4.0 - employee[col]) for col in competency_cols]
        avg_gap = np.mean(gaps)
        
        # Training targeting specific competencies gets higher score
        # This should be customized based on training metadata
        
        return min(avg_gap * 1.2, 5.0)
    
    def _calculate_collaborative_score(self, employee_idx, training, all_employees) -> float:
        """Score based on what similar employees need"""
        similar_employees = self.cf_model.get_similar_employees(employee_idx, top_n=5)
        
        # Check if similar employees have similar gaps
        # Simplified: return moderate score
        return 3.0
    
    def _calculate_pestle_score(self, employee, training) -> float:
        """Calculate PESTLE relevance"""
        # Division match
        if pd.notna(training['target_division']):
            if str(employee['division']).lower() in str(training['target_division']).lower():
                return 5.0
            return 3.0
        return 2.5
    
    def _calculate_cluster_score(self, employee, training, all_employees) -> float:
        """Score based on cluster characteristics"""
        if 'cluster' not in employee.index:
            return 3.0
        
        cluster_id = employee['cluster']
        cluster_chars = self.clustering_model.get_cluster_characteristics(cluster_id)
        
        # Training should address cluster weaknesses
        return 3.5  # Simplified
    
    def _calculate_cost_efficiency(self, training, gap_score) -> float:
        """Calculate cost-effectiveness"""
        # Normalize cost (assuming max 20M IDR)
        cost_normalized = min(training['cost'] / 20000000, 1.0)
        
        # Higher gap justifies higher cost
        efficiency = (gap_score / 5.0) / (cost_normalized + 0.1)
        
        return min(efficiency * 5.0, 5.0)
    
    def _apply_filters(self, catalog, employee, filters):
        """Apply filtering logic"""
        filtered = catalog.copy()
        
        if filters:
            if 'max_cost' in filters:
                filtered = filtered[filtered['cost'] <= filters['max_cost']]
            
            if 'max_duration' in filters:
                filtered = filtered[filtered['duration_days'] <= filters['max_duration']]
            
            if 'job_family' in filters:
                filtered = filtered[filtered['job_family'] == filters['job_family']]
        
        return filtered
    
    def _get_priority(self, score):
        """Determine priority level"""
        if score >= 4.0:
            return 'Critical'
        elif score >= 3.0:
            return 'High'
        elif score >= 2.0:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_roi(self, scores, cost):
        """Estimate ROI of training"""
        impact = scores['competency_gap'] + scores['content_based']
        roi_ratio = (impact / 10.0) / (cost / 10000000)
        return min(roi_ratio * 100, 500)  # ROI percentage


# Utility Functions
def prepare_training_catalog(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training catalog for ML processing
    
    Handles:
    - JSON parsing of learning_objectives
    - Text normalization
    - Feature engineering
    """
    catalog = catalog_df.copy()
    
    # Parse learning objectives if it's JSON string
    if catalog['learning_objectives'].dtype == 'object':
        catalog['learning_objectives'] = catalog['learning_objectives'].apply(
            lambda x: ' '.join(json.loads(x)) if isinstance(x, str) and x.startswith('[') else str(x)
        )
    
    # Create combined text feature
    catalog['combined_text'] = catalog.apply(
        lambda x: f"{x['training_name']} {x['learning_objectives']} {x['job_family']}", 
        axis=1
    )
    
    return catalog


def export_recommendations(recommendations: List[Dict], filename: str = 'recommendations.csv'):
    """Export recommendations to CSV"""
    df = pd.DataFrame(recommendations)
    df.to_csv(filename, index=False)
    print(f"✓ Recommendations exported to {filename}")


# Example Usage
if __name__ == "__main__":
    print("Advanced ML Models for Training Recommender System")
    print("=" * 60)
    print("\nModules loaded successfully!")
    print("\nAvailable models:")
    print("  1. TextSimilarityEngine - TF-IDF + Cosine Similarity")
    print("  2. CompetencyClusteringModel - K-Means Clustering")
    print("  3. CollaborativeFilteringRecommender - NMF Matrix Factorization")
    print("  4. HybridRecommenderEngine - Complete hybrid system")