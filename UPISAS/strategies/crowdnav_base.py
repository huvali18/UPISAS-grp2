from UPISAS.strategy import Strategy
from UPISAS.strategies.learner import SituationLearner
import logging

logging.getLogger().setLevel(logging.INFO)

# paper mappings:
# exploration percentage -> exploration_percentage  [0-0.3]
# route randomization -> route_random_sigma  [0-0.3]
# static info weight -> max_speed_and_length_factor  [1-2.5]
# dynamic info weight -> average_edge_duration_factor [1-2.5]
# exploration weight -> exploration_percentage  [5-20]
# data freshness threshold -> freshness_cut_off_value [100-700]
# re-routing frequency -> re_route_every_ticks [10-70]

# unused:
# freshness_update_factor [1, inf]
# edge_average_influence [0, 1]

# edge_average_influence: 140 - Not mentioned in paper (might be implementation-specific)


class CrowdNavBaseStrategy(Strategy):
    def __init__(self, exemplar):
        super().__init__(exemplar)
        self.learner = SituationLearner(num_cars_ranges=[(100, 150), (150, 200), (200, 250), (250, 300), (300, 350), (350, 400), (400, 450), (450, 500), (500, 550), (550, 600), (600, 650), (650, 700), (700, 750), (750, 800)], candidate_k_values=[2, 3, 4, 5, 6, 7, 8, 9])
        self.is_learning = True

    def analyze(self):
        return False

    def _analyze_learning(self):
        data = self.knowledge.monitored_data
        logging.info(f"data={data}")
        return False

    def plan(self):
        return False