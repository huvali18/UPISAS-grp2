from enum import Enum
from typing import Dict, Optional, List
from UPISAS.strategy import Strategy
from UPISAS.strategies.learner import SituationLearner
import logging
from pathlib import Path
import time
logging.getLogger().setLevel(logging.INFO)
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt

class State(Enum):
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    OPERATIONAL = "operational"


class OptimizationPhase(Enum):
    WARMUP = "warmup"
    COLLECTING = "collecting"
    READY = "ready"


class CrowdNavBaseStrategy(Strategy):
    SITUATIONS = {
        0: {'range': (101, 500), 'representative': 500, 'name': 'low_traffic'},
        1: {'range': (501, 700), 'representative': 700, 'name': 'medium_traffic'},
        2: {'range': (701, 800), 'representative': 800, 'name': 'high_traffic'}
    }

    IGNORE_FIRST_N_RESULTS = 700
    SAMPLE_SIZE = 1200
    N_CALLS = 60

    def __init__(self, exemplar, state: str):
        super().__init__(exemplar)
        self.learner = SituationLearner(
            num_cars_ranges=[(100, 150), (150, 200), (200, 250), (250, 300), 
                           (300, 350), (350, 400), (400, 450), (450, 500), 
                           (500, 550), (550, 600), (600, 650), (650, 700), 
                           (700, 750), (750, 800)],
            candidate_k_values=[2, 3, 4, 5, 6, 7, 8, 9]
        )
        
        self.state: State = State(state)
        logging.info(f"State set to: {self.state}")
        
        self.knowledge_base_file = Path("knowledge_base.json")
        self.config: dict | None = None
        self.current_situation: int | None = None
        self.num_cars: int | None = None
        self.knowledge_base = self._load_knowledge_base()
        
        self.optimizer: Optional[Optimizer] = None
        self.optimization_iteration: int = 0
        self.optimization_phase: OptimizationPhase = OptimizationPhase.WARMUP
        self.current_test_config: Optional[Dict] = None
        self.optimization_history: List = []
        self.warmup_start_trips: int = 0
        self.collection_start_trips: int = 0
        self.all_tested_configs: List = []

    def _load_knowledge_base(self) -> Dict:
        import json
        if not self.knowledge_base_file.exists(): raise FileNotFoundError(f"kb not found: {self.knowledge_base_file}")
        with open(self.knowledge_base_file, 'r') as f: kb = json.load(f)
        logging.info(f"loaded kb from: {kb['metadata']['created_at']}")
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
        logging.info(f"current cars: {current_cars}")
        
        if current_cars == 0:
            return None
        
        for situation_id, info in self.SITUATIONS.items():
            low, high = info['range']
            if low <= current_cars <= high:
                return situation_id
        
        logging.warning(f"car count {current_cars} is broken")
        return None

    def analyze(self):
        data = self.knowledge.monitored_data
        match self.state:
            case State.OPTIMIZATION:
                return self._analyze_optimization(data)
            case State.LEARNING:
                return self._analyze_learning(data)
            case State.OPERATIONAL:
                return self._analyze_operational(data)
            case _:
                raise ValueError(f"Invalid state: {self.state}")

    def _analyze_operational(self, data) -> bool:
        situation_id = self.detect_current_situation()
        
        if situation_id is None:
            logging.warning("No situation detected")
            return False
        
        # changed? -> need to plan
        if situation_id == self.current_situation: 
            return False
            
        self.current_situation = situation_id
        logging.info(f"situation changed to: {situation_id}")
        return True

    def _reset_statistics(self):
        if not self.current_test_config:
            logging.warning("No current config to reset statistics with")
            return
        
        config_with_reset = self.current_test_config.copy()
        config_with_reset.pop('reset_statistics', None)
        config_with_reset['reset_statistics'] = True
        self.execute(adaptation=config_with_reset, with_validation=False)

    def _analyze_optimization(self, data) -> bool:
        if 'car_stats' not in data or len(data['car_stats']) == 0: return False
        last_stats = data['car_stats'][-1]
        logging.debug(f"last stats: {last_stats}")
        total_trips = last_stats.get('total_trips', 0)
        logging.debug(f"total trips: {total_trips}")
        
        # phase 1: warmup
        if self.optimization_phase == OptimizationPhase.WARMUP:
            if total_trips >= self.IGNORE_FIRST_N_RESULTS:
                logging.info(f"warmup complete ({total_trips} trips)")
                self.optimization_phase = OptimizationPhase.COLLECTING
                self._reset_statistics()
                time.sleep(5)
            else:
                if total_trips % 100 == 0 and total_trips > 0: logging.info(f"  Warming up: {total_trips}/{self.IGNORE_FIRST_N_RESULTS}")
            return False
        
        elif self.optimization_phase == OptimizationPhase.COLLECTING:
            if total_trips >= self.SAMPLE_SIZE:
                overhead_avg = last_stats.get('total_trip_overhead_average', 0)
                logging.info(f"collection complete: {total_trips} trips, overhead={overhead_avg:.3f}")
                
                self.all_tested_configs.append({
                    'config': self.current_test_config.copy(),
                    'overhead': overhead_avg,
                    'iteration': self.optimization_iteration
                })
                
                self.optimization_phase = OptimizationPhase.READY
                return True  # ready to plan next config
            else:
                if total_trips % 100 == 0 and total_trips > 0:
                    current_overhead = last_stats.get('total_trip_overhead_average', 0)
                    logging.info(f"collecting: {total_trips}/{self.SAMPLE_SIZE} (overhead: {current_overhead:.3f})")
            return False
        
        return False

    def _analyze_learning(self, data) -> bool:
        return False

    def plan(self):
        match self.state:
            case State.OPTIMIZATION:
                return self._plan_optimization()
            case State.LEARNING:
                return self._plan_learning()
            case State.OPERATIONAL:
                return self._plan_operational()
            case _:
                raise ValueError(f"Invalid state: {self.state}")

    def _plan_operational(self) -> bool:
        logging.info(f"planning situation: {self.current_situation}")
        optimal_config = self.get_optimal_config_for_situation(self.current_situation)
        
        if optimal_config is not None:
            self.set_config(optimal_config)
            optimal_config_copy = optimal_config.copy()
            optimal_config_copy['reset_statistics'] = True
            self.knowledge.plan_data = optimal_config_copy
            logging.info(f"found config in kb: {optimal_config}")
            return True
        else:
            logging.info(f"no optimal config for situation {self.current_situation}, starting opt phase")
            return self._initialize_optimization()

    def _plan_optimization(self) -> bool:
        if self.optimization_phase != OptimizationPhase.READY:
            return False
        
        if len(self.all_tested_configs) > 0:
            last_result = self.all_tested_configs[-1]
            config_values = self._config_to_optimizer_values(last_result['config'])
            self.optimizer.tell(config_values, last_result['overhead'])
            logging.info(f"reported result to optimizer: overhead={last_result['overhead']:.3f}")

        if self.optimization_iteration >= self.N_CALLS:
            self._finalize_optimization()
            return True
        
        self.optimization_iteration += 1
        
        next_config_values = self.optimizer.ask()
        next_config = self._optimizer_values_to_config(next_config_values)
        next_config['total_car_counter'] = self.num_cars
        next_config['reset_statistics'] = True
        
        self.current_test_config = next_config
        self.knowledge.plan_data = next_config
        
        logging.info(f"opt iteration {self.optimization_iteration}/{self.N_CALLS}")
        logging.info(f"testing config: {next_config}")
        self.optimization_phase = OptimizationPhase.WARMUP
        
        return True

    def _plan_learning(self) -> bool:
        pass

    def _initialize_optimization(self) -> bool:
        logging.info(f"initializing opt for situation {self.current_situation}, cars: {self.num_cars}")
        
        self.set_state("optimization")
        
        search_space = [
            Real(0.0, 0.3, name='exploration_percentage'),
            Real(0.0, 0.3, name='route_random_sigma'),
            Real(1.5, 7, name='max_speed_and_length_factor'),
            Real(1.5, 7, name='average_edge_duration_factor'),
            Real(10, 30, name='freshness_update_factor'),
            Integer(100, 700, name='freshness_cut_off_value'),
            Categorical([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], name='re_route_every_ticks'),
            Integer(1, 100, name='edge_average_influence'),
        ]
        
        self.optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="GP",
            n_initial_points=7,
            random_state=self.current_situation + 420
        )
        
        self.optimization_iteration = 0
        self.all_tested_configs = []
        
        logging.info("optimizer initialized")
        
        first_config_values = self.optimizer.ask()
        first_config = self._optimizer_values_to_config(first_config_values)
        first_config['total_car_counter'] = self.num_cars
        first_config['reset_statistics'] = True
        
        self.current_test_config = first_config
        self.knowledge.plan_data = first_config
        
        self.optimization_phase = OptimizationPhase.WARMUP
        self.optimization_iteration = 1
        
        logging.info(f"opt iteration {self.optimization_iteration}/{self.N_CALLS}")
        logging.info(f"first config prepared: {first_config}")
        return True

    def _save_optimization_plots(self, plot_dir: Path):
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        iterations = [r['iteration'] for r in self.all_tested_configs]
        overheads = [r['overhead'] for r in self.all_tested_configs]
        
        running_best = []
        current_best = float('inf')
        for overhead in overheads:
            current_best = min(current_best, overhead)
            running_best.append(current_best)
        
        # Plot 1: Convergence Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(iterations, overheads, 'o-', alpha=0.6, label='Tested configs')
        plt.plot(iterations, running_best, 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Iteration')
        plt.ylabel('Trip Overhead')
        plt.title(f'Optimization Convergence (Situation {self.current_situation})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Overhead Distribution
        plt.subplot(1, 2, 2)
        plt.hist(overheads, bins=min(15, len(overheads)), alpha=0.7, edgecolor='black')
        plt.axvline(min(overheads), color='r', linestyle='--', linewidth=2, label=f'Best: {min(overheads):.3f}')
        plt.xlabel('Trip Overhead')
        plt.ylabel('Frequency')
        plt.title('Overhead Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        convergence_path = plot_dir / f'situation_{self.current_situation}_convergence.png'
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"✓ Saved convergence plot: {convergence_path}")
        
        # Plot 3: Parameter Exploration (show how parameters vary)
        param_names = [
            'exploration_%', 'route_sigma', 'speed_factor', 
            'duration_factor', 'freshness_update', 'freshness_cutoff',
            're_route_ticks', 'edge_influence'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        param_keys = [
            'exploration_percentage', 'route_random_sigma', 
            'max_speed_and_length_factor', 'average_edge_duration_factor',
            'freshness_update_factor', 'freshness_cut_off_value',
            're_route_every_ticks', 'edge_average_influence'
        ]
        
        for idx, (param_key, param_name) in enumerate(zip(param_keys, param_names)):
            ax = axes[idx]
            param_values = [r['config'][param_key] for r in self.all_tested_configs]
            
            # (darker = better)
            scatter = ax.scatter(iterations, param_values, c=overheads, 
                               cmap='RdYlGn_r', alpha=0.7, s=80)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Exploration')
            ax.grid(True, alpha=0.3)
            
            best_idx = overheads.index(min(overheads))
            ax.scatter([iterations[best_idx]], [param_values[best_idx]], 
                      color='red', s=200, marker='*', edgecolors='black', linewidths=2,
                      label='Best', zorder=5)
        
        plt.suptitle(f'Parameter Exploration - Situation {self.current_situation}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(scatter, cax=cbar_ax, label='Trip Overhead')
        
        params_path = plot_dir / f'situation_{self.current_situation}_parameters.png'
        plt.savefig(params_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"saved param exploration plot: {params_path}")

    def _finalize_optimization(self):
        logging.info("optimization complete")
        
        best_result = min(self.all_tested_configs, key=lambda x: x['overhead'])
        best_config = best_result['config']
        best_overhead = best_result['overhead']
        
        logging.info(f"best config: {best_config}")
        logging.info(f"best overhead: {best_overhead:.3f}")
        logging.info(f"out of {len(self.all_tested_configs)} configs")
        
        plot_dir = Path("optimization_plots")
        try:
            self._save_optimization_plots(plot_dir)
        except Exception as e:
            logging.error(f"no plots, gg: {e}")
        
        self.knowledge_base['situations'][str(self.current_situation)]['optimal_config'] = best_config
        
        import json
        with open(self.knowledge_base_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        logging.info(f"saved to kb: {self.knowledge_base_file}")
        
        self.optimization_history = {
            'best_params': best_config,
            'best_overhead': best_overhead,
            'all_results': self.all_tested_configs
        }
        
        self.set_state("operational")
        self.set_config(best_config)
        
        best_config_with_reset = best_config.copy()
        best_config_with_reset['reset_statistics'] = True
        self.knowledge.plan_data = best_config_with_reset

    def _config_to_optimizer_values(self, config: Dict) -> List:
        variables = [
            'exploration_percentage',
            'route_random_sigma',
            'max_speed_and_length_factor',
            'average_edge_duration_factor',
            'freshness_update_factor',
            'freshness_cut_off_value',
            're_route_every_ticks',
            'edge_average_influence'
        ]
        return [config[var] for var in variables]

    def _optimizer_values_to_config(self, values: List) -> Dict:
        return {
            'exploration_percentage': float(values[0]),
            'route_random_sigma': float(values[1]),
            'max_speed_and_length_factor': float(values[2]),
            'average_edge_duration_factor': float(values[3]),
            'freshness_update_factor': float(values[4]),
            'freshness_cut_off_value': float(values[5]),
            're_route_every_ticks': int(values[6]),
            'edge_average_influence': float(values[7])
        }

    def get_optimal_config_for_situation(self, situation_id: int) -> Optional[Dict]:
        try:
            situation_data = self.knowledge_base['situations'][str(situation_id)]
            optimal_config = situation_data.get('optimal_config')
            return optimal_config if optimal_config else None
        except Exception as e:
            logging.error(f"no optimal config for situation {situation_id}, gg: {e}")
            return None

    def execute(self, adaptation=None, endpoint_suffix="execute", with_validation=True):
        logging.info(f"adaptation passed directly: {adaptation}")
        result = super().execute(adaptation, endpoint_suffix, with_validation)
        time.sleep(1)
        return result

