from UPISAS.exemplars.crowdnav_d import CrowdNav
from UPISAS.strategies.crowdnav_base import CrowdNavBaseStrategy
import logging
import time
import sys

def run_simulation(crowdnav: CrowdNav, strategy: CrowdNavBaseStrategy, num_cars: int):
    base_config = {
        "exploration_percentage": 0.0,
        "route_random_sigma": 0.2,
        "max_speed_and_length_factor": 1,
        "average_edge_duration_factor": 1,
        "freshness_update_factor": 10,
        "freshness_cut_off_value": 90,
        "re_route_every_ticks": 60,
        "edge_average_influence": 50,
        "total_car_counter": num_cars,
    }
    strategy.execute(base_config)

    samples = {}
    num_samples = 5000
    while len(samples) < num_samples:
        # sys.stdout.write(f"Running simulation... {len(samples)}/{num_samples}\r")
        # sys.stdout.flush()
        logging.warning(f"Running simulation... {len(samples)}/{num_samples}")
        data = crowdnav.get_current_status()
        # if step already in samples, skip
        if data["step"] in samples: continue
        samples[data["step"]] = data["avg_overhead"]
    return samples

def main():
    logging.getLogger().setLevel(logging.WARNING)
    # import json
    crowdnav = CrowdNav(auto_start=True, start_services=True)
    strategy = CrowdNavBaseStrategy(crowdnav)

    if not crowdnav.wait_for_api_ready():
        logging.error("Could not connect to API, skipping status check")
        return

    base_config = {
        "exploration_percentage": 0.0,
        "route_random_sigma": 0.2,
        "max_speed_and_length_factor": 1,
        "average_edge_duration_factor": 1,
        "freshness_update_factor": 10,
        "freshness_cut_off_value": 90,
        "re_route_every_ticks": 60,
        "edge_average_influence": 50,
        "total_car_counter": 150,
    }

    strategy.execute(base_config)
    
    samples = run_simulation(crowdnav, strategy, 150)
    logging.info(f"Samples: {samples}")

    crowdnav.stop_container()


if __name__ == "__main__":
    main()
