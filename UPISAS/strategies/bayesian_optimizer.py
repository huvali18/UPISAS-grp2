import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logging.getLogger().setLevel(logging.INFO)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.stats import norm
from scipy.stats import qmc
from scipy.optimize import differential_evolution

@dataclass
class Configuration:
    exploration_percentage: float  # [0, 0.3]
    route_random_sigma: float  # [0, 0.3]
    max_speed_and_length_factor: float  # [1, 2.5]
    average_edge_duration_factor: float  # [1, 2.5]
    freshness_update_factor: int  # [5, 20]
    freshness_cut_off_value: int  # [100, 700]
    re_route_every_ticks: int  # [10, 70]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for UPISAS execute"""
        return {
            'exploration_percentage': self.exploration_percentage,
            'route_random_sigma': self.route_random_sigma,
            'max_speed_and_length_factor': self.max_speed_and_length_factor,
            'average_edge_duration_factor': self.average_edge_duration_factor,
            'freshness_update_factor': self.freshness_update_factor,
            'freshness_cut_off_value': self.freshness_cut_off_value,
            're_route_every_ticks': self.re_route_every_ticks
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for GP"""
        return np.array([
            self.exploration_percentage,
            self.route_random_sigma,
            self.max_speed_and_length_factor,
            self.average_edge_duration_factor,
            self.freshness_update_factor,
            self.freshness_cut_off_value,
            self.re_route_every_ticks
        ])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'Configuration':
        """Create configuration from array"""
        return Configuration(
            exploration_percentage=float(arr[0]),
            route_random_sigma=float(arr[1]),
            max_speed_and_length_factor=float(arr[2]),
            average_edge_duration_factor=float(arr[3]),
            freshness_update_factor=int(round(arr[4])),
            freshness_cut_off_value=int(round(arr[5])),
            re_route_every_ticks=int(round(arr[6]))
        )


class BayesianOptimizer:
    # (from Table II in paper)
    BOUNDS = np.array([
        [0.0, 0.3],# exploration_percentage
        [0.0, 0.3],# route_random_sigma
        [1.0, 2.5],# max_speed_and_length_factor
        [1.0, 2.5],# average_edge_duration_factor
        [5, 20],# freshness_update_factor
        [100, 700],# freshness_cut_off_value
        [10, 70]# re_route_every_ticks
    ])
    
    def __init__(
        self,
        budget: int = 100,
        n_initial_samples: int = 10,
        ehvi_mc_samples: int = 128,
        rng: Optional[np.random.Generator] = None
    ):
        self.budget = budget
        self.n_initial_samples = n_initial_samples
        self.ehvi_mc_samples = ehvi_mc_samples
        self._rng = rng or np.random.default_rng()
        
        # storage for evaluated
        self.X_evaluated = []  # configs
        self.y_trip_overhead = []  # trip overhead values
        self.y_routing_cost = []  # routing cost values
        
        self.gp_trip_overhead = None
        self.gp_routing_cost = None
        
        self.pareto_front_X = []
        self.pareto_front_y = []
        
        self.ref_point = np.array([3.0, 100.0])  # conservative worst-case
        self._initial_candidates = None
        
    def generate_initial_samples(self) -> List[Configuration]:
        sampler = qmc.LatinHypercube(d=self.BOUNDS.shape[0], seed=int(self._rng.integers(0, 2**32 - 1)))
        unit_samples = sampler.random(self.n_initial_samples)
        scaled = qmc.scale(unit_samples, self.BOUNDS[:, 0], self.BOUNDS[:, 1])
        return [Configuration.from_array(sample) for sample in scaled]
    
    def add_evaluation(self, config: Configuration, trip_overhead: float, routing_cost: float):
        self.X_evaluated.append(config.to_array())
        self.y_trip_overhead.append(trip_overhead)
        self.y_routing_cost.append(routing_cost)
        
        logging.info(f"{len(self.X_evaluated)}/{self.budget}: trip_overhead={trip_overhead:.4f}, routing_cost={routing_cost:.2f}")
        
        self._update_reference_point()
        self._update_pareto_front()
    
    def _update_pareto_front(self):
        if len(self.X_evaluated) == 0: return
        
        X = np.array(self.X_evaluated)
        y = np.column_stack([self.y_trip_overhead, self.y_routing_cost])
        
        # find non-dominated solutions
        pareto_mask = self._is_pareto_efficient(y)
        
        self.pareto_front_X = X[pareto_mask].tolist()
        self.pareto_front_y = y[pareto_mask].tolist()
        
        logging.info(f"Pareto front size: {len(self.pareto_front_X)}")
    
    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i in range(costs.shape[0]):
            if is_efficient[i]:
                # mark points dominated by point i as inefficient
                dominated_by_i = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
                is_efficient[dominated_by_i] = False
                is_efficient[i] = True
        
        logging.info(f"is_efficient: {is_efficient}")
        logging.info(f"costs: {costs}")
        logging.info(f"dominated_by_i: {dominated_by_i}")
        logging.info(f"i: {i}")
        logging.info(f"is_efficient[i]: {is_efficient[i]}")
        logging.info(f"is_efficient[dominated_by_i]: {is_efficient[dominated_by_i]}")
        logging.info(f"is_efficient[i]: {is_efficient[i]}")
        return is_efficient

    def _update_reference_point(self):
        if len(self.y_trip_overhead) == 0: return
        margin_overhead = 0.1
        margin_cost = 1.0
        self.ref_point = np.array([float(np.max(self.y_trip_overhead) + margin_overhead), float(np.max(self.y_routing_cost) + margin_cost),])
    
    def _compute_hypervolume(self, points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        
        ref = self.ref_point
        # clip to reference point to avoid negative widths/heights
        clipped = np.minimum(points, ref)
        sorted_indices = np.argsort(clipped[:, 0])
        sorted_points = clipped[sorted_indices]
        
        hypervolume = 0.0
        prev_obj2 = ref[1]
        
        for point in sorted_points:
            width = max(ref[0] - point[0], 0.0)
            height = max(prev_obj2 - point[1], 0.0)
            hypervolume += width * height
            prev_obj2 = min(prev_obj2, point[1])
        
        return hypervolume
    
    def _expected_hypervolume_improvement(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        pareto_points: np.ndarray,
        hv_base: float
    ) -> float:
        safe_std = np.where(std < 1e-9, 1e-9, std)
        samples = self._rng.normal(
            loc=mean,
            scale=safe_std,
            size=(self.ehvi_mc_samples, mean.shape[0])
        )
        
        if pareto_points.size == 0:
            hv_fn = lambda sample: self._compute_hypervolume(sample.reshape(1, -1))
        else:
            hv_fn = lambda sample: self._compute_hypervolume(
                np.vstack((pareto_points, sample))
            )
        
        improvements = [
            max(0.0, hv_fn(sample) - hv_base)
            for sample in samples
        ]
        return float(np.mean(improvements))
    
    def fit_gp_models(self):
        if len(self.X_evaluated) < 2:
            logging.warning("Not enough samples to fit GP models")
            return
        
        X = np.array(self.X_evaluated)
        
        # kernel: constant * rbf
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # gp for trip overhead
        self.gp_trip_overhead = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Noise level (CrowdNav has noisy outputs)
            n_restarts_optimizer=10,
            normalize_y=True
        )
        self.gp_trip_overhead.fit(X, self.y_trip_overhead)
        
        # gp for routing cost
        self.gp_routing_cost = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        self.gp_routing_cost.fit(X, self.y_routing_cost)
        
        logging.info("✓ GP models fitted")
    
    def propose_next_configuration(self) -> Configuration:
        if len(self.X_evaluated) < self.n_initial_samples:
            # still in initial sampling phase (use deterministic LHS batch)
            if self._initial_candidates is None:
                self._initial_candidates = self.generate_initial_samples()
            idx = len(self.X_evaluated)
            if idx < len(self._initial_candidates):
                return self._initial_candidates[idx]
            logging.warning("No initial candidates available, generating new batch")
            return self.generate_initial_samples()[0]
        
        self.fit_gp_models()
        
        pareto_points = (
            np.array(self.pareto_front_y)
            if self.pareto_front_y else
            np.empty((0, 2))
        )
        hv_base = self._compute_hypervolume(pareto_points)
        
        def acquisition(x):
            x = x.reshape(1, -1)
            
            mu_overhead, std_overhead = self.gp_trip_overhead.predict(x, return_std=True)
            mu_cost, std_cost = self.gp_routing_cost.predict(x, return_std=True)
            
            mean = np.array([mu_overhead[0], mu_cost[0]])
            std = np.array([std_overhead[0], std_cost[0]])
            ehvi = self._expected_hypervolume_improvement(mean, std, pareto_points, hv_base)
            return -ehvi
        
        result = differential_evolution(
            acquisition,
            bounds=self.BOUNDS,
            maxiter=100,
            seed=int(self._rng.integers(0, 2**32 - 1)),
            workers=1
        )
        
        next_config = Configuration.from_array(result.x)
        return next_config
    
    def _expected_improvement(self, mu: float, sigma: float,  best: float, xi: float = 0.01) -> float:
        if sigma < 1e-9:
            return 0.0
        
        improvement = best - mu - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return max(0.0, ei)
    
    def calculate_hypervolume(self) -> float:
        if len(self.pareto_front_y) == 0: return 0.0
        pareto_y = np.array(self.pareto_front_y)
        return self._compute_hypervolume(pareto_y)
    
    def get_best_configuration(self) -> Tuple[Configuration, float, float]:
        if len(self.pareto_front_X) == 0:
            best_idx = np.argmin(self.y_trip_overhead)
            return (
                Configuration.from_array(self.X_evaluated[best_idx]),
                self.y_trip_overhead[best_idx],
                self.y_routing_cost[best_idx]
            )
        
        pareto_y = np.array(self.pareto_front_y)
        best_idx = np.argmin(pareto_y[:, 0])
        
        return (
            Configuration.from_array(self.pareto_front_X[best_idx]),
            pareto_y[best_idx, 0],
            pareto_y[best_idx, 1]
        )
    
    def get_results_summary(self) -> Dict:
        if len(self.y_trip_overhead) == 0: return {}
        pareto_y = np.array(self.pareto_front_y) if self.pareto_front_y else np.column_stack([self.y_trip_overhead, self.y_routing_cost])
        return {
            'evaluations': len(self.X_evaluated),
            'pareto_size': len(self.pareto_front_X),
            'min_trip_overhead': np.min(pareto_y[:, 0]),
            'min_routing_cost': np.min(pareto_y[:, 1]),
            'avg_trip_overhead': np.mean(pareto_y[:, 0]),
            'avg_routing_cost': np.mean(pareto_y[:, 1]),
            'hypervolume': self.calculate_hypervolume(),
            'pareto_front': pareto_y.tolist()
        }
