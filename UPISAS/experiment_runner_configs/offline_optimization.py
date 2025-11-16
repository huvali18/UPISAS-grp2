from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ExtendedTyping.Typing import SupportsStr
from ProgressManager.Output.OutputProcedure import OutputProcedure as output

from typing import Dict, Optional
from pathlib import Path
from os.path import dirname, realpath
import time
import json

from UPISAS.strategies.offline_bayesian_optimizer import OfflineBayesianOptimizer


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    name: str = "offline_bayesian_optimization"
    results_output_path: Path = ROOT_DIR / 'experiments' / 'offline'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 30000

    BUDGET = 100
    STABILIZATION_TIME = 60
    SAMPLING_TIME = 30
    
    KNOWLEDGE_BASE_FILE = ROOT_DIR / 'experiments' / 'offline' / 'knowledge_base.json'

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
        
        # Load or initialize Knowledge Base
        self.knowledge_base = self._load_or_create_kb()
        
        output.console_log("Offline Optimization config loaded")

    def _load_or_create_kb(self) -> Dict:
        """Load existing KB or create new one"""
        if self.KNOWLEDGE_BASE_FILE.exists():
            with open(self.KNOWLEDGE_BASE_FILE, 'r') as f:
                kb = json.load(f)
            output.console_log(f"Loaded existing Knowledge Base from {self.KNOWLEDGE_BASE_FILE}")
            return kb
        else:
            kb = {
                'metadata': {
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'budget_per_situation': self.BUDGET,
                    'stabilization_time': self.STABILIZATION_TIME,
                    'optimizer': 'Bayesian Optimization (SMS-EGO variant)'
                },
                'situations': {}
            }
            self.KNOWLEDGE_BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
            return kb

    def create_run_table_model(self) -> RunTableModel:
        factor_situation = FactorModel("car_count", [500, 700, 800])
        
        self.run_table_model = RunTableModel(
            factors=[factor_situation],
            exclude_variations=[],
            data_columns=[
                'evaluations',
                'pareto_size',
                'min_trip_overhead',
                'min_routing_cost',
                'hypervolume',
                'optimization_time_sec'
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("Offline Optimization config loaded")


    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        car_count = context.run_variation['car_count']
        
        context.car_count = car_count
        context.optimization_start_time = time.time()

    def start_measurement(self, context: RunnerContext) -> None:
        pass

    def interact(self, context: RunnerContext) -> None:
        car_count = context.car_count
        
        optimizer = OfflineBayesianOptimizer(
            situation_car_count=car_count,
            budget=self.BUDGET,
            stabilization_time=self.STABILIZATION_TIME,
            sampling_time=self.SAMPLING_TIME
        )
        
        try:
            results = optimizer.run_optimization()
            context.optimization_results = results
            context.optimization_end_time = time.time()
            self.knowledge_base['situations'][str(car_count)] = results
            with open(self.KNOWLEDGE_BASE_FILE, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            
        except Exception as e:
            context.optimization_results = None
            context.optimization_end_time = time.time()

    def stop_measurement(self, context: RunnerContext) -> None:
        pass

    def stop_run(self, context: RunnerContext) -> None:
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        results = getattr(context, 'optimization_results', None)
        
        if results is None:
            return {
                'evaluations': 0,
                'pareto_size': 0,
                'min_trip_overhead': 0,
                'min_routing_cost': 0,
                'hypervolume': 0,
                'optimization_time_sec': 0
            }
        
        opt_results = results['optimization_results']
        optimization_time = (getattr(context, 'optimization_end_time', 0) - 
                           getattr(context, 'optimization_start_time', 0))
        
        return {
            'evaluations': opt_results['evaluations'],
            'pareto_size': opt_results['pareto_size'],
            'min_trip_overhead': opt_results['min_trip_overhead'],
            'min_routing_cost': opt_results['min_routing_cost'],
            'hypervolume': opt_results['hypervolume'],
            'optimization_time_sec': optimization_time
        }

    def after_experiment(self) -> None:
        output.console_log("optimization experiment complete")
    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None

