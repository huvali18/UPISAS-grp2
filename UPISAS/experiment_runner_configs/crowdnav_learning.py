from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import time
import json
import pickle

from UPISAS.strategies.crowdnav_base import CrowdNavBaseStrategy
from UPISAS.exemplars.crowdnav_d import CrowdNav


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    name: str = "crowdnav_learning_phase"
    results_output_path: Path = ROOT_DIR / 'experiments' / 'learning'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 10000  # 10 seconds between runs

    exemplar = None
    strategy = None
    
    SAMPLES_PER_RUN = 1000
    MONITOR_INTERVAL = 0.05
    
    all_monitoring_data = {}

    def __init__(self):
        """Executes immediately after program start, on config load"""
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment)
        ])
        self.run_table_model = None
        output.console_log("Learning Phase config loaded")

    def create_run_table_model(self) -> RunTableModel:
        factor_car_count = FactorModel(
            "car_count", 
            [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
        )
        
        self.run_table_model = RunTableModel(
            factors=[factor_car_count],
            exclude_variations=[],
            data_columns=[
                'samples_collected',      
                'avg_trip_overhead',
                'std_trip_overhead',  
                'median_trip_overhead',
                'collection_time_sec'
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log(" Data Collection Clustering")

    def before_run(self) -> None:
        self.exemplar = CrowdNav(
            auto_start=True,
            start_services=True
        )
        self.strategy = CrowdNavBaseStrategy(self.exemplar)
        time.sleep(5)

    def start_run(self, context: RunnerContext) -> None:
        car_count = context.run_variation['car_count']
        
        learning_config = {
            'exploration_percentage': 0.1,
            'route_random_sigma': 0.2,
            'max_speed_and_length_factor': 1,
            'average_edge_duration_factor': 1,
            'freshness_update_factor': 10,
            'freshness_cut_off_value': 90,
            're_route_every_ticks': 60,
            'edge_average_influence': 140,
            'total_car_counter': car_count
        }

        self.exemplar.start_run()
        if not self.exemplar.wait_for_api_ready():
            output.console_log("API not ready, skipping execution")
            return
        time.sleep(10)
        self.strategy.execute(learning_config)
        self.exemplar.get_current_status()
        context.collection_start_time = time.time()

    def start_measurement(self, context: RunnerContext) -> None:
        self.strategy.get_monitor_schema()
        self.strategy.get_adaptation_options_schema()
        self.strategy.get_execute_schema()

    def interact(self, context: RunnerContext) -> None:
        car_count = context.run_variation['car_count']
        
        data_file = self.results_output_path / self.name / 'collected_data.json'
        if data_file.exists():
            with open(data_file, 'r') as f:
                self.all_monitoring_data = json.load(f)
            output.console_log(f"Loaded existing data: {list(self.all_monitoring_data.keys())}")
        
        samples_collected = 0
        target_samples = self.SAMPLES_PER_RUN
        
        output.console_log(f"Collecting {target_samples} samples for {car_count} cars...")
        
        while samples_collected < target_samples:
            output.console_log(f"collected {samples_collected}/{target_samples} samples")
            self.strategy.monitor(verbose=True)
            
            monitored = self.strategy.knowledge.monitored_data
            if 'car_stats' in monitored:
                samples_collected = len(monitored['car_stats'])
            time.sleep(self.MONITOR_INTERVAL)
        
        context.collection_end_time = time.time()
        context.samples_collected = samples_collected
        
        self.all_monitoring_data[str(car_count)] = {
            'car_count': car_count,
            'samples': [
                {
                    'trip_overhead': s.get('total_trip_overhead_average'),
                    'car_count': s.get('driving_car_counter'),
                    'routing_cost': s.get('total_routing_time_average')
                }
                for s in monitored.get('car_stats', [])
                if s.get('total_trip_overhead_average') is not None
            ],
            'num_samples': samples_collected
        }
        
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'w') as f:  json.dump(self.all_monitoring_data, f, indent=2)
        
        for cc, data in self.all_monitoring_data.items():
            output.console_log(f"  {cc} cars: {len(data['samples'])} samples")

    def stop_measurement(self, context: RunnerContext) -> None:
        output.console_log("Stopping measurement...")

    def stop_run(self, context: RunnerContext) -> None:
        if self.exemplar:
            self.exemplar.stop_container()
        output.console_log("Run stopped")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        car_count = context.run_variation['car_count']
        
        data_file = self.results_output_path / self.name / 'collected_data.json'
        if data_file.exists():
            with open(data_file, 'r') as f:
                self.all_monitoring_data = json.load(f)
        
        if str(car_count) in self.all_monitoring_data:
            data = self.all_monitoring_data[str(car_count)]
            trip_overheads = [s['trip_overhead'] for s in data['samples'] 
                            if s['trip_overhead'] is not None]
            
            import numpy as np
            avg_overhead = np.mean(trip_overheads) if trip_overheads else 0
            std_overhead = np.std(trip_overheads) if trip_overheads else 0
            median_overhead = np.median(trip_overheads) if trip_overheads else 0
        else:
            avg_overhead = std_overhead = median_overhead = 0
        
        collection_time = (getattr(context, 'collection_end_time', 0) - 
                          getattr(context, 'collection_start_time', 0))
        
        return {
            'samples_collected': getattr(context, 'samples_collected', 0),
            'avg_trip_overhead': avg_overhead,
            'std_trip_overhead': std_overhead,
            'median_trip_overhead': median_overhead,
            'collection_time_sec': collection_time
        }

    def after_experiment(self) -> None:
        data_file = self.results_output_path / self.name / 'collected_data.json'
        if not data_file.exists():
            return
        
        with open(data_file, 'r') as f:
            self.all_monitoring_data = json.load(f)
        
        self.exemplar = CrowdNav(auto_start=False, start_services=False)
        self.strategy = CrowdNavBaseStrategy(self.exemplar)
        
        combined_data = {'car_stats': []}
        for car_count, data in self.all_monitoring_data.items():
            for sample in data['samples']:
                combined_data['car_stats'].append({
                    'driving_car_counter': sample['car_count'],
                    'total_trip_overhead_average': sample['trip_overhead'],
                    'total_routing_time_average': sample.get('routing_cost')
                })
        
        
        self.strategy.learner.load_from_monitored_data(combined_data)
        num_situations = self.strategy.learner.perform_clustering()
        situations_file = self.results_output_path / self.name / 'situations.json'
        situations_data = {
            'optimal_k': self.strategy.learner.optimal_k,
            'silhouette_scores': self.strategy.learner.silhouette_scores,
            'situation_labels': self.strategy.learner.situation_labels,
            'num_cars_ranges': [(low, high) for low, high in self.strategy.learner.num_cars_ranges],
            'statistics': self.strategy.learner.get_statistics()
        }
        
        with open(situations_file, 'w') as f:
            json.dump(situations_data, f, indent=2)
        
        pickle_file = situations_file.with_suffix('.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(situations_data, f)

        for range_idx, situation_id in sorted(self.strategy.learner.situation_labels.items()):
            low, high = self.strategy.learner.num_cars_ranges[range_idx]
            output.console_log(f"  [{low:3d}, {high:3d}] cars → Situation {situation_id}")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None

