from UPISAS.strategy import Strategy
import logging
import json
from pathlib import Path
from typing import Dict, Optional
import time

logging.getLogger().setLevel(logging.INFO)


class RuntimeLookupStrategy(Strategy):
    SITUATIONS = {
        0: {'range': (101, 500), 'representative': 500, 'name': 'low_traffic'},
        1: {'range': (501, 700), 'representative': 700, 'name': 'medium_traffic'},
        2: {'range': (701, 800), 'representative': 800, 'name': 'high_traffic'}
    }
    
    def __init__(self, exemplar, knowledge_base_file: Path):
        super().__init__(exemplar)
        
        self.knowledge_base_file = knowledge_base_file
        self.knowledge_base = self._load_knowledge_base()
        
        self.current_situation = None
        self.current_config = None
        self.last_adaptation_time = None
        self.min_adaptation_interval = 120
        
    def _load_knowledge_base(self) -> Dict:
        if not self.knowledge_base_file.exists():
            logging.error(f"Knowledge Base not found: {self.knowledge_base_file}")
            logging.error("Run offline optimization first:")
            logging.error("  python -m UPISAS.strategies.offline_bayesian_optimizer")
            raise FileNotFoundError(f"Knowledge Base not found: {self.knowledge_base_file}")
        
        with open(self.knowledge_base_file, 'r') as f:
            kb = json.load(f)
        
        logging.info(f"✓ Loaded Knowledge Base created at {kb['metadata']['created_at']}")
        return kb
    
    def detect_current_situation(self) -> Optional[int]:
        monitored = self.knowledge.monitored_data
        car_stats = monitored.get('car_stats', [])
        
        if isinstance(car_stats, list):
            if len(car_stats) == 0:
                return None
            car_stats = car_stats[-1]
        
        current_cars = car_stats.get('driving_car_counter', 0)
        
        if current_cars == 0:
            return None
        
        for situation_id, info in self.SITUATIONS.items():
            low, high = info['range']
            if low <= current_cars <= high:
                return situation_id
        
        logging.warning(f"Car count {current_cars} doesn't match any situation")
        return None
    
    def get_optimal_config_for_situation(self, situation_id: int) -> Optional[Dict]:
        situation_info = self.SITUATIONS[situation_id]
        car_count = situation_info['representative']
        
        if str(car_count) not in self.knowledge_base['situations']:
            logging.error(f"No optimization results for {car_count} cars in Knowledge Base")
            return None
        
        situation_data = self.knowledge_base['situations'][str(car_count)]
        
        pareto_configs = situation_data['pareto_front_configs']
        
        if not pareto_configs:
            logging.error(f"No Pareto front configs for {car_count} cars")
            return None
        
        optimization_results = situation_data['optimization_results']
        pareto_front = optimization_results['pareto_front']
        
        min_idx = min(range(len(pareto_front)), key=lambda i: pareto_front[i][0])
        optimal_config = pareto_configs[min_idx].copy()
        optimal_config['total_car_counter'] = car_count
        optimal_config['edge_average_influence'] = 140
        return optimal_config
    
    def analyze(self) -> bool:
        situation_id = self.detect_current_situation()
        
        if situation_id is None:
            return False
        
        # situation changed?
        if situation_id == self.current_situation:
            return False
        
        if self.last_adaptation_time:
            elapsed = time.time() - self.last_adaptation_time
            if elapsed < self.min_adaptation_interval:
                logging.info(f"Situation changed but waiting for stability "
                           f"({elapsed:.0f}s < {self.min_adaptation_interval}s)")
                return False
        
        logging.info(f"Situation changed: {self.current_situation} → {situation_id}")
        self.current_situation = situation_id
        
        return True
    
    def plan(self) -> bool:
        if self.current_situation is None:
            return False
        
        optimal_config = self.get_optimal_config_for_situation(self.current_situation)
        
        if optimal_config is None:
            logging.error(f"No optimal config found for situation {self.current_situation}")
            return False
        
        self.knowledge.plan_data['config'] = optimal_config
        self.knowledge.plan_data['situation'] = self.current_situation
        
        return True
    
    def execute(self, adaptation: Dict = None) -> bool:
        if adaptation is None:
            adaptation = self.knowledge.plan_data.get('config')
        
        if adaptation is None:
            logging.error("No configuration to execute")
            return False
        
        success = super().execute(adaptation)
        
        if success:
            self.current_config = adaptation
            self.last_adaptation_time = time.time()
            logging.info(f"✓ Configuration applied successfully")
        
        return success
    
    def get_current_status(self) -> Dict:
        return {
            'current_situation': self.current_situation,
            'current_config': self.current_config,
            'last_adaptation_time': self.last_adaptation_time,
            'knowledge_base_file': str(self.knowledge_base_file)
        }
