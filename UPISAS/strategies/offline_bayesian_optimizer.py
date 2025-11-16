import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict

from UPISAS.strategies.bayesian_optimizer import (
    BayesianOptimizer, 
    Configuration
)
from UPISAS.exemplars.crowdnav_d import CrowdNav

logging.basicConfig(level=logging.INFO)


class OfflineBayesianOptimizer:
    """
    Runs offline optimization to pre-compute optimal configurations.
    
    Each run:
    1. Starts fresh CrowdNav instance
    2. Runs optimization for ONE situation
    3. Stores Pareto front to Knowledge Base
    4. Shuts down CrowdNav
    """
    
    def __init__(self, situation_car_count: int, budget: int = 100, 
                 stabilization_time: int = 30, sampling_time: int = 60):
        self.situation_car_count = situation_car_count
        self.budget = budget
        self.stabilization_time = stabilization_time
        self.sampling_time = sampling_time
        self.optimizer = BayesianOptimizer(budget=budget, n_initial_samples=10)
        self.exemplar = None
        logging.info(f"OfflineBayesianOptimizer initialized for {self.situation_car_count} cars")

    def start_crowdnav(self, config: Dict | None = None):
        self.exemplar = CrowdNav(auto_start=True, start_services=True)
        logging.info("CrowdNav started")
        
        if not self.exemplar.wait_for_api_ready(max_retries=30): raise RuntimeError("CrowdNav failed to start")
        
        config = config or {
            'exploration_percentage': 0.1,
            'route_random_sigma': 0.2,
            'max_speed_and_length_factor': 1.0,
            'average_edge_duration_factor': 1.0,
            'freshness_update_factor': 10,
            'freshness_cut_off_value': 500,
            're_route_every_ticks': 60,
            'edge_average_influence': 140,
            'total_car_counter': self.situation_car_count
        }
        
        import requests
        url = f"{self.exemplar.base_endpoint}/execute"
        # get response
        requests.put(url, json=config)
        time.sleep(10)
    
    def stop_crowdnav(self):
        if self.exemplar:
            logging.info("Stopping CrowdNav...")
            self.exemplar.stop_container()
            self.exemplar = None
            logging.info("✓ CrowdNav stopped")
    
    def evaluate_configuration(self, config: Configuration) -> Tuple[float, float]:
        config_dict = config.to_dict()
        # restart crowdnav
        self.stop_crowdnav()
        config_dict['total_car_counter'] = self.situation_car_count
        config_dict['edge_average_influence'] = 140
        self.start_crowdnav(config_dict)
        # wait for api to be ready
        if not self.exemplar.wait_for_api_ready(max_retries=30): raise RuntimeError("CrowdNav failed to start")

        
        import requests
        # url = f"{self.exemplar.base_endpoint}/execute"
        # response = requests.put(url, json=config_dict)
        
        # if response.status_code != 200:
        #     logging.error(f"Failed to apply config: {response.text}")
        #     return float('inf'), float('inf')
        
        logging.info(f"Waiting {self.stabilization_time}s for stabilization...")
        # time.sleep(self.stabilization_time)
        
        monitor_url = f"{self.exemplar.base_endpoint}/monitor"
        # wait while step is less than 1000
        while True:
            response = requests.get(monitor_url)
            if response.status_code != 200:
                logging.error("Failed to get monitoring data")
                time.sleep(1)
                continue
            data = response.json()
            car_stats = data.get('car_stats', {})
            step = car_stats.get('step', 0)
            if step >= 1000: break
            time.sleep(0.1)

        end_time = time.time() + self.sampling_time
        overhead_samples: List[float] = []
        routing_samples: List[float] = []
        num_samples = 0
        while num_samples < 200:
            response = requests.get(monitor_url)
            if response.status_code != 200:
                logging.error("Failed to get monitoring data")
                time.sleep(1)
                continue
            data = response.json()
            num_samples += 1
            car_stats = data.get('car_stats', {})
            tovh = car_stats.get('total_trip_overhead_average', None)
            rc = car_stats.get('routing_duration', None)
            if isinstance(tovh, (int, float)): overhead_samples.append(float(tovh))
            if isinstance(rc, (int, float)): routing_samples.append(float(rc))
            time.sleep(0.05)
        
        trip_overhead = float(np.mean(overhead_samples)) if overhead_samples else float('inf')
        routing_cost = float(np.sum(routing_samples)) if routing_samples else float('inf')
        logging.info(f"Overhead samples: {len(overhead_samples)}")
        logging.info(f"Routing samples: {len(routing_samples)}")
        logging.info(f"  Measured (avg over {self.sampling_time}s): trip_overhead={trip_overhead:.4f}, routing_cost={routing_cost:.2f}")
        
        return trip_overhead, routing_cost
    
    def run_optimization(self) -> Dict:
        try:
            self.start_crowdnav()
            for iteration in range(self.budget):
                logging.info(f"\n--- Iteration {iteration + 1}/{self.budget} ---")
                
                # next configuration
                next_config = self.optimizer.propose_next_configuration()
                logging.info(f"Proposed config: {next_config.to_dict()}")
                
                trip_overhead, routing_cost = self.evaluate_configuration(next_config)
                logging.info(f"Evaluated configuration: trip_overhead={trip_overhead:.4f}, routing_cost={routing_cost:.2f}")
                self.optimizer.add_evaluation(next_config, trip_overhead, routing_cost)
                
                results = self.optimizer.get_results_summary()
                logging.info(f"Progress: {results['evaluations']}/{self.budget}")
                logging.info(f"  Best trip_overhead: {results['min_trip_overhead']:.4f}")
                logging.info(f"  Best routing_cost: {results['min_routing_cost']:.2f}")
                logging.info(f"  Hypervolume: {results['hypervolume']:.2f}")
                logging.info(f"  Pareto size: {results['pareto_size']}")
            
            results = self.optimizer.get_results_summary()
            
            logging.info(f"Pareto front size: {results['pareto_size']}")
            logging.info(f"Best trip_overhead: {results['min_trip_overhead']:.4f}")
            logging.info(f"Best routing_cost: {results['min_routing_cost']:.2f}")
            logging.info(f"Hypervolume: {results['hypervolume']:.2f}")
            
            return {
                'situation_car_count': self.situation_car_count,
                'budget': self.budget,
                'stabilization_time': self.stabilization_time,
                'optimization_results': results,
                'pareto_front_configs': [
                    Configuration.from_array(x).to_dict() 
                    for x in self.optimizer.pareto_front_X
                ]
            }
            
        finally:
            self.stop_crowdnav()


def run_offline_optimization_for_all_situations(
    situations: List[int] = [500, 700, 800],
    budget: int = 100,
    stabilization_time: int = 60,
    output_file: Path = Path("knowledge_base.json")
):
    knowledge_base = {
        'metadata': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'budget_per_situation': budget,
            'stabilization_time': stabilization_time,
            'optimizer': 'Bayesian Optimization (SMS-EGO variant)'
        },
        'situations': {}
    }
    
    for situation_car_count in situations:
        logging.info("\n" + "="*80)
        logging.info(f"OPTIMIZING FOR SITUATION: {situation_car_count} CARS")
        logging.info("="*80)
        
        optimizer = OfflineBayesianOptimizer(
            situation_car_count=situation_car_count,
            budget=budget,
            stabilization_time=stabilization_time
        )
        
        try:
            results = optimizer.run_optimization()
            knowledge_base['situations'][str(situation_car_count)] = results
            with open(output_file, 'w') as f:
                json.dump(knowledge_base, f, indent=2)
            
            logging.info(f"\n✓ Saved results to {output_file}")
            time.sleep(30)
            
        except Exception as e:
            logging.error(f"✗ Optimization failed for {situation_car_count} cars: {e}")
            continue
    
    return knowledge_base
