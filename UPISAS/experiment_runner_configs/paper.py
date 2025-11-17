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

from UPISAS.exemplars.crowdnav_d import CrowdNav
from UPISAS.strategies.base import CrowdNavBaseStrategy

class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    name: str = "crowdnav_bayesian_rtx"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

    IGNORE_FIRST_N_RESULTS = 100
    SAMPLE_SIZE = 1000
    N_CALLS = 20

    exemplar = None
    optimization_history = []
    
    def __init__(self):
        """Initialize experiment runner"""
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
        output.console_log("CrowdNav Bayesian RTX config loaded")

    def create_run_table_model(self) -> RunTableModel:
        factor_situation = FactorModel("car_count", [500, 700, 800])

        self.run_table_model = RunTableModel(
            factors=[factor_situation],
            exclude_variations=[],
            data_columns=[
                'best_exploration',
                'best_sigma', 
                'best_overhead',
                'mean_overhead',
                'iterations_completed'
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("CrowdNav Bayesian Optimization")

    def before_run(self) -> None:
        output.console_log("Connecting to CrowdNav exemplar...")
        self.exemplar = CrowdNav(
            auto_start=True,
            start_services=True
        )
        self.strategy = CrowdNavBaseStrategy(self.exemplar, "operational")
        time.sleep(2)

    def start_run(self, context: RunnerContext) -> None:
        num_cars = context.run_variation['car_count']
        self.strategy.num_cars = num_cars
        try:
            status = self.exemplar.get_current_status()
            output.console_log(f"CrowdNav connected! Status: {status}")
        except Exception as e:
            output.console_log(f"ERROR: Cannot connect to CrowdNav: {e}")
            raise
        time.sleep(1)

    def start_measurement(self, context: RunnerContext) -> None:
        """Start measurements"""
        output.console_log("Starting Bayesian optimization measurement...")

    def interact(self, context: RunnerContext) -> None:
        # set current situation 
        # wait for api ready
        if not self.exemplar.wait_for_api_ready():
            output.console_log("API not ready, skipping execution")
            return
        time.sleep(10)
        num_cars = context.run_variation['car_count']
        config = {
            'exploration_percentage': 0.1,
            'route_random_sigma': 0.1,
            'max_speed_and_length_factor': 1.5,
            'average_edge_duration_factor': 1.5,
            'freshness_update_factor': 10,
            'freshness_cut_off_value': 100,
            're_route_every_ticks': 60,
            'edge_average_influence': 10,
            "total_car_counter": num_cars,
            'reset_statistics': True
        }
        output.console_log(f"Applying config: {config}")
        self.strategy.set_config(config)
        self.strategy.execute(config, with_validation=False)
        time.sleep(4)
        self.strategy.get_monitor_schema()
        self.strategy.get_adaptation_options_schema()
        self.strategy.get_execute_schema()

        while True:
            self.strategy.monitor(verbose=True)
            if self.strategy.analyze():
                if self.strategy.plan():
                    self.strategy.execute()

            time.sleep(0.1)

    def stop_measurement(self, context: RunnerContext) -> None:
        """Stop measurements"""
        output.console_log("Measurement complete")

    def stop_run(self, context: RunnerContext) -> None:
        """Stop the run"""
        # Don't stop containers - they're external
        output.console_log("Run complete")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        # TODO: finish this for graphing
        output.console_log(f"Optimization history: {self.optimization_history}")
        output.console_log(f"{context}")
        return {
            'avg_trip_overhead': self.optimization_history['best_overhead']
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment"""
        output.console_log("=" * 70)
        output.console_log("Bayesian optimization experiment complete!")
        output.console_log("=" * 70)

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None

