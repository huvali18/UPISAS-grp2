from enum import Enum
from typing import Dict, Optional, List
from UPISAS.strategy import Strategy
from UPISAS.strategies.learner import SituationLearner
import logging
from pathlib import Path
import time
logging.getLogger().setLevel(logging.INFO)
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
class State(Enum):
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    OPERATIONAL = "operational"

class CrowdNavBaseStrategy(Strategy):

    SITUATIONS = {
        0: {'range': (101, 500), 'representative': 500, 'name': 'low_traffic'},
        1: {'range': (501, 700), 'representative': 700, 'name': 'medium_traffic'},
        2: {'range': (701, 800), 'representative': 800, 'name': 'high_traffic'}
    }

    IGNORE_FIRST_N_RESULTS = 100 # Paper uses 1000
    SAMPLE_SIZE = 1000 # Paper uses 1000
    N_CALLS = 20 # Number of Bayesian optimization iterations
    def __init__(self, exemplar, state: str):
        super().__init__(exemplar)
        self.learner = SituationLearner(num_cars_ranges=[(100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 650), (650, 700), (700, 750), (750, 800)], candidate_k_values=[2, 3, 4, 5, 6, 7, 8, 9])
        self.is_warming_up = True
        self.state: State | None = State(state)
        logging.info(f"State set to: {self.state}")
        self.knowledge_base_file = Path("knowledge_base.json")
        self.ignore_first_n = 100
        self.sample_size = 1000
        self.config: dict | None = None
        self.current_situation: int | None = None
        self.num_cars: int | None = None
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict:
        import json
        if not self.knowledge_base_file.exists():
            raise FileNotFoundError(f"Knowledge Base not found: {self.knowledge_base_file}")
        
        with open(self.knowledge_base_file, 'r') as f:
            kb = json.load(f)
        
        logging.info(f"{kb['metadata']['created_at']}")
        return kb
    
    def set_config(self, config): self.config = config
    def set_state(self, state: str): self.state = State(state)

    def detect_current_situation(self) -> Optional[int]:
        monitored = self.knowledge.monitored_data
        configs = monitored.get('configs', [])
        
        if isinstance(configs, list):
            if len(configs) == 0:
                return None
            configs = configs[-1]
        
        current_cars = configs.get('total_car_counter', 0)
        logging.info(f"Current cars: {current_cars}")
        
        if current_cars == 0:
            return None
        
        for situation_id, info in self.SITUATIONS.items():
            low, high = info['range']
            if low <= current_cars <= high:
                return situation_id
        
        logging.warning(f"Car count {current_cars} doesn't match any situation")
        return None

    def analyze(self):
        data = self.knowledge.monitored_data
        match self.state:
            case State.OPTIMIZATION:
                return self._analyze_optimization(data)
            case State.LEARNING:
                return self._analyze_learning(data)
            case State.OPERATIONAL:
                return self._analyze(data)
            case _:
                raise ValueError(f"Invalid state: {self.state}")

    def _analyze_optimization(self, data, sample_counter = 'total_trips'):
        # return true is optimization should end, false otherwise, populate info for planning
        # last_sample = data['car_stats'][-1]
        # num_samples = last_sample[sample_counter] # total number of cars completed trips
        # logging.info(f"num_samples={num_samples}")
        # if self.is_warming_up:
        #     if num_samples >= self.ignore_first_n: 
        #         self.is_warming_up = False
        #         self.reset_statistics()
        #         time.sleep(3) # might take some time to get new config
        #     return False
        
        # if num_samples >= self.sample_size:
        #     # get overhead average
        #     logging.info(f"enough samples, last_sample={last_sample}")
        #     overhead_average = last_sample['total_trip_overhead_average']
        #     self.knowledge.analysis_data['num_samples'] = num_samples
        #     self.knowledge.analysis_data['opt_phase_finished'] = True
        #     self.knowledge.analysis_data['overhead_average'] = overhead_average
        #     return True

        # return False
        return False # TODO: ither delete or slightly change abstraction

    def _analyze_learning(self, data): return False

    def _analyze(self, data): 
        # get the range we are in
        # num_cars = data['configs']['total_car_counter']
        situation_id = self.detect_current_situation()
        
        if situation_id is None:
            logging.warning("No situation detected")
            return False
        
        # situation changed? -> run optimization
        if situation_id == self.current_situation: return False
        self.current_situation = situation_id
        return True

    def reset_statistics(self):
        import requests
        knob_object = self.config.copy() if self.config else {}
        knob_object['reset_statistics'] = True
        try:
            url = f"{self.exemplar.base_endpoint}/execute"
            response = requests.put(url, json=knob_object)
            logging.info(f"Statistics reset: {response.status_code}")
        except Exception as e:
            raise Exception(f"ERROR resetting statistics: {e}")

    def plan(self):
        # situation changed, and new config is needed
        # try to get config from knowledge base
        logging.info(f"current situation: {self.current_situation}")
        optimal_config = self.get_optimal_config_for_situation(self.current_situation)
        if optimal_config is not None:
            self.set_config(optimal_config)
            optimal_config['reset_statistics'] = True
            self.knowledge.plan_data['config'] = optimal_config
            logging.info(f"Optimal config found in knowledge base: {optimal_config}")
            return True
        else:
            self.set_state("optimization")
            self.optimize(self.current_situation)
            optimal_config = self.get_optimal_config_for_situation(self.current_situation).copy()
            logging.info(f"Optimal config found after optimization: {optimal_config}")
            if optimal_config is not None:
                self.set_config(optimal_config)
                optimal_config['reset_statistics'] = True
                self.knowledge.plan_data['config'] = optimal_config
                logging.info(f"Optimal config set: {optimal_config}")
                return True
            else:
                return False
        
        # match self.state:
        #     case State.OPTIMIZATION:
        #         return self._plan_optimization()
        #     case State.LEARNING:
        #         return self._plan_learning()
        #     case State.OPERATIONAL:
        #         return self._plan_operational()
        #     case _:
        #         return False

    def optimize(self, car_count: int):
        logging.info(f"Optimizing for {car_count} cars")
        search_spaces = [
            Real(0.0, 0.3, name='exploration_percentage'), # range from paper
            Real(0.0, 0.3, name='route_random_sigma'), # range from paper 2
            Real(1, 5, name='max_speed_and_length_factor'), # static info weight, aka w_dynamic in paper 1
            Real(1, 5, name='average_edge_duration_factor'), # dynamic info weight, aka w_static in paper 1
            Real(10, 20, name='freshness_update_factor'), # w_benefit, aka "exploration weight" in second paper (5-20 in paper 2)
            Integer(100, 700, name='freshness_cut_off_value'), # data freshness threshold
            Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], name='re_route_every_ticks'),
            Integer(1, 100, name='edge_average_influence'), # not used in any of the papers
        ]
        logging.info(f"Search spaces: {search_spaces}")
        variables = ['exploration_percentage', 'route_random_sigma', 'max_speed_and_length_factor', 'average_edge_duration_factor', 'freshness_update_factor', 'freshness_cut_off_value', 're_route_every_ticks', 'edge_average_influence']
        optimizer_result = gp_minimize(
            func=lambda opti_values: self.experiment_function(variables, opti_values),
            dimensions=search_spaces,
            n_calls=self.N_CALLS,
            random_state=car_count,
            verbose=True
        )
        
        self.optimization_history = {
            'best_params': self.recreate_knob_from_optimizer_values(variables, optimizer_result.x),
            'best_overhead': optimizer_result.fun,
            'all_results': [(self.recreate_knob_from_optimizer_values(variables, x), y) 
                           for x, y in zip(optimizer_result.x_iters, optimizer_result.func_vals)]
        }
        self.set_config(self.optimization_history['best_params'])
        logging.info(f"Best configuration: {self.optimization_history['best_params']}")
        logging.info(f"Best overhead: {self.optimization_history['best_overhead']:.3f}")
        # set knowledge base
        self.knowledge_base['situations'][self.current_situation]['optimal_config'] = self.optimization_history['best_params']

    def recreate_knob_from_optimizer_values(self, variables: List[str], opti_values: List[float]) -> Dict:
        knob_object = {}
        for idx, val in enumerate(variables):
            knob_object[val] = float(opti_values[idx])
        return knob_object

    def experiment_function(self, variables: List[str], opti_values: List[float]) -> float:
        import requests
        # Recreate knob object
        knob_object = self.recreate_knob_from_optimizer_values(variables, opti_values)
        car_count = self.num_cars
        logging.info(f"Car count: {car_count}")
        logging.info(f"Testing configuration: {knob_object}")
        knob_object['total_car_counter'] = car_count
        knob_object['reset_statistics'] = True  # Reset CarRegistry stats
        
        # Apply configuration and reset statistics
        try:
            url = f"{self.exemplar.base_endpoint}/execute"
            response = requests.put(url, json=knob_object)
            logging.info(f"Configuration applied with reset: {response.status_code}")
        except Exception as e:
            logging.error(f"ERROR applying configuration: {e}")
        
        # Wait a moment for reset to take effect
        time.sleep(0.5)
        
        # Phase 1: Warm-up period - wait for system to stabilize
        logging.info(f"Phase 1: Warming up (waiting for {self.IGNORE_FIRST_N_RESULTS} trips)...")
        monitor_url = f"{self.exemplar.base_endpoint}/monitor"
        logging.info(f"Monitor URL: {monitor_url}")
        
        while True:
            response = requests.get(monitor_url)
            if response.status_code == 200:
                data = response.json()
                logging.info(f"Data: {data}")
                total_trips = data['car_stats']['total_trips']
                logging.info(f"Total trips: {total_trips}")
                if total_trips >= self.IGNORE_FIRST_N_RESULTS:
                    logging.info(f"  Warm-up complete ({total_trips} trips)")
                    break
                if total_trips % 100 == 0 and total_trips > 0:
                    logging.info(f"  Warm-up: {total_trips}/{self.IGNORE_FIRST_N_RESULTS} trips")
            time.sleep(0.5)
        
        # Verify configuration
        if data['configs']['total_car_counter'] != car_count:
            logging.error(f"ERROR: Number of cars in config is not the same as the car count")
            raise Exception(f"ERROR: Number of cars in config is not the same as the car count")

        logging.info(f"Phase 1 complete. System stabilized.")
        
        # Phase 2: Collection period - wait for more trips to build average
        logging.info(f"Phase 2: Collecting data (waiting for {self.SAMPLE_SIZE} more trips)...")
        target_trips = self.SAMPLE_SIZE
        try:
            url = f"{self.exemplar.base_endpoint}/execute"
            response = requests.put(url, json=knob_object)
            logging.info(f"Configuration applied with reset: {response.status_code}")
        except Exception as e:
            logging.error(f"ERROR applying configuration: {e}")
            return 999.0  # Return high penalty
        response = requests.get(monitor_url)
        data = response.json()
        logging.info(f"Data1: {data}")
        time.sleep(3)
        response = requests.get(monitor_url)
        data = response.json()
        logging.info(f"Data2: {data}")


        while True:
            response = requests.get(monitor_url)
            if response.status_code == 200:
                data = response.json()
                total_trips = data['car_stats']['total_trips']
                # assert data['configs']['total_car_counter'] == car_count, f"Number of cars in config is not the same as the car count"
                if total_trips >= target_trips:
                    overhead_avg = data['car_stats']['total_trip_overhead_average']
                    logging.info(f"  Collection complete ({total_trips} total trips)")
                    break
                if (total_trips) % 100 == 0:
                    logging.info(f"Data: {data}")
                    current_overhead = data['car_stats'].get('total_trip_overhead_average', 0)
                    logging.info(f"  Collecting: {total_trips}/{self.SAMPLE_SIZE} trips (overhead: {current_overhead:.3f})")
            time.sleep(0.5)
        
        result = overhead_avg
        
        logging.info(f"Result: Average overhead = {result:.3f}")
        logging.info("=" * 70)
        
        return result
    

    def _plan_optimization(self):
        # let the optimizer choose the next config
        # return True
        self.optimize(self.num_cars)
        optimal_config = self.knowledge_base['situations'][self.current_situation]['optimal_config']
        logging.info(f"Optimal config found: {optimal_config}")
        if optimal_config is None:
            logging.error(f"No optimal config found for situation {self.current_situation}")
            return False
        else:
            self.set_config(optimal_config)
            optimal_config['reset_statistics'] = True
            self.knowledge.plan_data['config'] = optimal_config
            return True

    def _plan_learning(self): pass


    def get_optimal_config_for_situation(self, situation_id: int) -> Optional[Dict]:
        situation_info = self.SITUATIONS[situation_id]
        car_count = situation_info['representative']
        try:
            logging.info(f"Knowledge base: {self.knowledge_base}")
            situation_data = self.knowledge_base['situations'][situation_id]
            logging.info(f"Situation data: {situation_data}")
            pareto_configs = situation_data['optimal_config']
            logging.info(f"Pareto configs: {pareto_configs}")
            return pareto_configs # should be none if no optimal config found
        except Exception as e:
            logging.error(f"Error getting optimal config for situation {situation_id}: {e}")
            return None