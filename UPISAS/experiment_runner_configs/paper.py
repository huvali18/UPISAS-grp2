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
    name: str = "crowdnav_mape_bayesian"
    results_output_path: Path = ROOT_DIR / 'experiments'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 1000

    exemplar = None
    strategy = None
    
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
        output.console_log("CrowdNav MAPE-Integrated Bayesian config loaded")

    def create_run_table_model(self) -> RunTableModel:
        factor_situation = FactorModel("car_count", [500, 700, 800])

        self.run_table_model = RunTableModel(
            factors=[factor_situation],
            exclude_variations=[],
            data_columns=[
                'best_overhead',
                'worst_overhead',
                'mean_overhead',
                'iterations_completed',
                'optimization_duration'
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("crowdnav experiment starting...")

    def before_run(self) -> None:
        output.console_log("connecting to crowdnav exemplar...")
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
            output.console_log(f"crowdnav connected! status: {status}")
            output.console_log(f"testing with {num_cars} cars")
        except Exception as e:
            output.console_log(f"gg no connection: {e}")
            raise
        
        time.sleep(1)

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log("starting measurement")
        self.optimization_start_time = time.time()

    def interact(self, context: RunnerContext) -> None:
        if not self.exemplar.wait_for_api_ready():
            output.console_log("api not ready, skipping execution")
            return
        
        time.sleep(10)
        
        # Set initial configuration
        num_cars = context.run_variation['car_count']
        initial_config = {
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
        
        output.console_log(f"Applying initial config: {initial_config}")
        self.strategy.set_config(initial_config)
        self.strategy.execute(initial_config, with_validation=False)
        time.sleep(4)
        
        # Get schemas
        self.strategy.get_monitor_schema()
        self.strategy.get_adaptation_options_schema()
        self.strategy.get_execute_schema()

        output.console_log("starting MAPE loop")
        
        mape_iteration = 0
        while True:
            mape_iteration += 1
            self.strategy.monitor(verbose=False, with_validation=False)
            if self.strategy.analyze():
                output.console_log(f"\n[mape {mape_iteration}] analyze: action required")
                if self.strategy.plan():
                    output.console_log(f"[mape {mape_iteration}] plan: adaptation prepared")
                    self.strategy.execute(with_validation=False)
                    output.console_log(f"[mape {mape_iteration}] execute: adaptation applied\n")
                else:
                    output.console_log(f"[mape {mape_iteration}] plan: failed\n")
            
            if (hasattr(self.strategy, 'optimization_history') and 
                self.strategy.optimization_history and 
                self.strategy.state.value == "operational"):
                output.console_log("optimization complete - stopping loop")
                break
            
            time.sleep(0.5)

    def stop_measurement(self, context: RunnerContext) -> None:
        self.optimization_duration = time.time() - self.optimization_start_time
        output.console_log(f"measurement complete (duration: {self.optimization_duration:.1f}s)")

    def stop_run(self, context: RunnerContext) -> None:
        if self.exemplar:
            self.exemplar.stop_container()
        output.console_log("run stopped")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        if not hasattr(self.strategy, 'optimization_history') or not self.strategy.optimization_history:
            output.console_log("no optimization history available, should not happen")
            return {}
        
        history = self.strategy.optimization_history
        all_overheads = [result['overhead'] for result in history['all_results']]
        
        results = {
            'best_overhead': history['best_overhead'],
            'worst_overhead': max(all_overheads) if all_overheads else 0,
            'mean_overhead': sum(all_overheads) / len(all_overheads) if all_overheads else 0,
            'iterations_completed': len(history['all_results']),
            'optimization_duration': self.optimization_duration
        }
        
        output.console_log(f"results: {results}")
        return results

    def after_experiment(self) -> None:
        output.console_log("experiment complete")
        
        if hasattr(self.strategy, 'optimization_history') and self.strategy.optimization_history:
            history = self.strategy.optimization_history
            all_overheads = [result['overhead'] for result in history['all_results']]
            
            output.console_log(f"\nBest configuration: {history['best_params']}")
            output.console_log(f"Best overhead: {history['best_overhead']:.3f}")
            output.console_log(f"Worst overhead: {max(all_overheads):.3f}")
            output.console_log(f"Mean overhead: {sum(all_overheads)/len(all_overheads):.3f}")
            output.console_log(f"Total iterations: {len(history['all_results'])}")
            
            try:
                from shutil import copy2
                plot_source = Path("optimization_plots")
                if plot_source.exists() and self.experiment_path:
                    plot_dest = self.experiment_path / "plots"
                    plot_dest.mkdir(exist_ok=True)
                    for plot_file in plot_source.glob("*.png"): copy2(plot_file, plot_dest / plot_file.name)
                    output.console_log(f"plots copied to: {plot_dest}")
            except Exception as e:
                output.console_log(f"gg no plots: {e}")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None

