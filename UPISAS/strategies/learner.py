import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logging.getLogger().setLevel(logging.INFO)


class SituationLearner:
    def __init__(self, num_cars_ranges: list, candidate_k_values: list = None):
        self.num_cars_ranges = num_cars_ranges
        self.candidate_k_values = candidate_k_values or list(range(2, 10))
        
        self.samples_per_range = {}
        
        self.features = []
        self.range_indices = []
        self.situation_labels = {}
        self.optimal_k = None
        self.silhouette_scores = {}
        
    
    def load_from_monitored_data(self, monitored_data: dict):
        self.samples_per_range = {}
        
        if 'car_stats' not in monitored_data:
            logging.error("No 'car_stats' in monitored_data")
            return
        
        car_stats_list = monitored_data['car_stats']
        
        for stats in car_stats_list:
            num_cars = stats.get('driving_car_counter')
            trip_overhead = stats.get('total_trip_overhead_average')
            
            if num_cars is None or trip_overhead is None:
                continue
            
            range_idx = self._get_range_index(num_cars)
            if range_idx == -1:
                continue
            
            if range_idx not in self.samples_per_range:
                self.samples_per_range[range_idx] = []
            
            self.samples_per_range[range_idx].append(trip_overhead)
        
        total_samples = sum(len(samples) for samples in self.samples_per_range.values())
        logging.info(f"Loaded {total_samples} samples across {len(self.samples_per_range)} ranges")
    
    def _get_range_index(self, num_cars: int) -> int:
        for i, (low, high) in enumerate(self.num_cars_ranges):
            if i == len(self.num_cars_ranges) - 1:
                if low <= num_cars <= high:
                    return i
            else:
                if low <= num_cars < high:
                    return i
        
        logging.warning(f"num_cars={num_cars} not in any defined range")
        return -1
    
    def extract_features(self):
        self.features = []
        self.range_indices = []
        
        for range_idx in sorted(self.samples_per_range.keys()):
            samples = self.samples_per_range[range_idx]
            
            if len(samples) == 0:
                continue
            
            values = np.array(samples)
            
            features = [
                np.mean(values),
                np.median(values),
                np.percentile(values, 75),
                np.percentile(values, 90),
                np.std(values),
                np.var(values)
            ]
            
            self.features.append(features)
            self.range_indices.append(range_idx)
        
        logging.info(f"Extracted 6 features for {len(self.features)} ranges")
        return np.array(self.features)
    
    def perform_clustering(self):
        features_array = self.extract_features()
        
        if len(features_array) < 2:
            logging.error("Not enough ranges with data to cluster (need at least 2)")
            return 0
        
        best_k = None
        best_score = -1
        self.silhouette_scores = {}
        
        for k in self.candidate_k_values:
            if k >= len(features_array):
                continue
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_array)
            
            score = silhouette_score(features_array, labels)
            self.silhouette_scores[k] = score
            
            logging.info(f"k={k}: Silhouette score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        if best_k is None:
            logging.error("Could not find optimal k")
            return 0
        
        self.optimal_k = best_k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_array)
        
        self.situation_labels = {}
        for range_idx, label in zip(self.range_indices, labels):
            self.situation_labels[range_idx] = int(label)
        
        logging.info(f"Optimal k: {best_k} (Silhouette score: {best_score:.4f})")
        for range_idx in sorted(self.situation_labels.keys()):
            situation = self.situation_labels[range_idx]
            low, high = self.num_cars_ranges[range_idx]
            logging.info(f"  Range [{low}, {high}] -> Situation {situation}")
        
        return best_k
    
    def get_situation(self, num_cars: int) -> int:
        range_idx = self._get_range_index(num_cars)
        if range_idx == -1:
            return -1
        return self.situation_labels.get(range_idx, -1)
    
    def get_statistics(self) -> dict:
        total_samples = sum(len(samples) for samples in self.samples_per_range.values())
        num_ranges = len(self.samples_per_range)
        num_situations = self.optimal_k or 0
        
        return {
            'total_samples': total_samples,
            'num_ranges': num_ranges,
            'num_situations': num_situations,
            'optimal_k': self.optimal_k,
            'silhouette_scores': self.silhouette_scores,
            'situation_labels': self.situation_labels
        }

