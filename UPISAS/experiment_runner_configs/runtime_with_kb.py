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

from UPISAS.strategies.runtime_lookup_strategy import RuntimeLookupStrategy
from UPISAS.exemplars.crowdnav_d import CrowdNav


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    name: str = "runtime_with_knowledge_base"
    results_output_path: Path = ROOT_DIR / 'experiments' / 'runtime'
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 5000

    exemplar = None
    strategy = None
    
    KNOWLEDGE_BASE_FILE = ROOT_DIR / 'experiments' / 'offline' / 'knowledge_base.json'
    RUN_DURATION = 600

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
        output.console_log("Runtime with Knowledge Base config loaded")

    def create_run_table_model(self) -> RunTableModel:
        # 300 cars
        # 600 cars
        # 750 cars
        factor_scenario = FactorModel("scenario", ["low", "medium", "high"])
        
        self.run_table_model = RunTableModel(
            factors=[factor_scenario],
            exclude_variations=[],
            data_columns=[
                'scenario',
                'adaptations_made',
                'avg_trip_overhead',
                'avg_routing_cost',
                'final_situation',
                'kb_lookups',
            ]
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        output.console_log("Runtime with Knowledge Base config loaded")

    def before_run(self) -> None:
        output.console_log("Initializing CrowdNav exemplar...")
        
        self.exemplar = CrowdNav(
            auto_start=True,
            start_services=True
        )
        
        self.strategy = RuntimeLookupStrategy(
            self.exemplar,
            knowledge_base_file=self.KNOWLEDGE_BASE_FILE
        )
        
        time.sleep(5)

    def start_run(self, context: RunnerContext) -> None:
        scenario = context.run_variation['scenario']
        
        scenario_cars = {
            'low': 300,
            'medium': 600,
            'high': 750
        }
        
        initial_car_count = scenario_cars[scenario]
        
        initial_config = {
            'exploration_percentage': 0.1,
            'route_random_sigma': 0.2,
            'max_speed_and_length_factor': 1.0,
            'average_edge_duration_factor': 1.0,
            'freshness_update_factor': 10,
            'freshness_cut_off_value': 500,
            're_route_every_ticks': 60,
            'edge_average_influence': 140,
            'total_car_counter': initial_car_count
        }
        
        self.exemplar.start_run()
        if not self.exemplar.wait_for_api_ready():
            output.console_log("API not ready, skipping execution")
            return
        
        time.sleep(10)
        output.console_log(f"{initial_car_count} cars")
        self.strategy.execute(initial_config)
        
        context.scenario = scenario
        context.start_time = time.time()
        context.adaptations = 0
        context.kb_lookups = 0

    def start_measurement(self, context: RunnerContext) -> None:
        output.console_log("starting loop...")
        self.strategy.get_monitor_schema()
        self.strategy.get_execute_schema()

    def interact(self, context: RunnerContext) -> None:
        iteration = 0
        
        while time.time() - context.start_time < self.RUN_DURATION:
            iteration += 1
            
            self.strategy.monitor(verbose=False)
            should_adapt = self.strategy.analyze()
            
            if should_adapt:
                output.console_log(f"\n[Iteration {iteration}] Situation changed!")
                context.kb_lookups += 1
                has_plan = self.strategy.plan()
                
                if has_plan:
                    config = self.strategy.knowledge.plan_data['config']
                    success = self.strategy.execute(config)
                    if success: context.adaptations += 1
            time.sleep(1)
        
        self.strategy.monitor(verbose=False)
        monitored = self.strategy.knowledge.monitored_data
        car_stats = monitored.get('car_stats', [])
        
        if car_stats:
            recent_stats = car_stats[-30:]
            
            trip_overheads = [s['total_trip_overhead_average'] for s in recent_stats 
                            if isinstance(s, dict)]
            routing_costs = [s['routing_duration'] for s in recent_stats 
                           if isinstance(s, dict)]
            context.avg_trip_overhead = sum(trip_overheads) / len(trip_overheads) if trip_overheads else 0
            context.avg_routing_cost = sum(routing_costs) / len(routing_costs) if routing_costs else 0
        else:
            context.avg_trip_overhead = 0
            context.avg_routing_cost = 0
        
        context.final_situation = self.strategy.current_situation
        
        output.console_log(f"Adaptations made: {context.adaptations}")
        output.console_log(f"KB lookups: {context.kb_lookups}")
        output.console_log(f"Avg trip overhead: {context.avg_trip_overhead:.4f}")
        output.console_log(f"Avg routing cost: {context.avg_routing_cost:.2f}")

    def stop_measurement(self, context: RunnerContext) -> None:
        pass

    def stop_run(self, context: RunnerContext) -> None:
        """Clean up after run"""
        if self.exemplar:
            self.exemplar.stop_container()
        output.console_log("Run stopped")

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, SupportsStr]]:
        """Extract results from the run"""
        return {
            'scenario': getattr(context, 'scenario', ''),
            'adaptations_made': getattr(context, 'adaptations', 0),
            'avg_trip_overhead': getattr(context, 'avg_trip_overhead', 0),
            'avg_routing_cost': getattr(context, 'avg_routing_cost', 0),
            'final_situation': getattr(context, 'final_situation', -1),
            'kb_lookups': getattr(context, 'kb_lookups', 0)
        }

    def after_experiment(self) -> None:
        output.console_log("Runtime with Knowledge Base experiment complete")

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None

