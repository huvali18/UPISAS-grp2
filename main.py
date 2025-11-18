from UPISAS.exemplars.crowdnav_d import CrowdNav
from UPISAS.strategies.crowdnav_base import CrowdNavBaseStrategy
import logging
import time
import sys

def run_simulation(crowdnav: CrowdNav, strategy: CrowdNavBaseStrategy, num_cars: int):
    # base_config = {
    #     "exploration_percentage": 0.0,
    #     "route_random_sigma": 0.2,
    #     "max_speed_and_length_factor": 1,
    #     "average_edge_duration_factor": 1,
    #     "freshness_update_factor": 10,
    #     "freshness_cut_off_value": 90,
    #     "re_route_every_ticks": 60,
    #     "edge_average_influence": 50,
    #     "total_car_counter": num_cars,
    # }
    # strategy.execute(base_config)
    default_config = {
        "exploration_percentage": 0,
        "route_random_sigma": 0.2,
        "max_speed_and_length_factor": 1,
        "average_edge_duration_factor": 1,
        "freshness_update_factor": 10,
        "freshness_cut_off_value": 90,
        "re_route_every_ticks": 60,
        "total_car_counter": 500,
        "edge_average_influence": 140,
        "reset_statistics": True
    }
    strategy.execute(default_config, with_validation=False)
    time.sleep(5)
    samples = {}
    num_samples = 3000
    car_counters = []
    while len(samples) < num_samples:
        # sys.stdout.write(f"Running simulation... {len(samples)}/{num_samples}\r")
        # sys.stdout.flush()
        logging.warning(f"Running simulation... {len(samples)}/{num_samples}")
        data = crowdnav.get_current_status()
        car_counters.append(data["cars_driving"])
        # if step aleady in samples, skip
        if data["step"] in samples: continue
        samples[data["step"]] = data["avg_overhead"]

    # get average car counter
    average_car_counter = sum(car_counters) / len(car_counters)
    min_car_counter = min(car_counters)
    max_car_counter = max(car_counters)
    logging.warning(f"Car counters: {car_counters}")
    logging.warning(f"Average car counter: {average_car_counter}")
    logging.warning(f"Min car counter: {min_car_counter}")
    logging.warning(f"Max car counter: {max_car_counter}")
    # average overhead
    average_overhead = sum(samples.values()) / len(samples)
    logging.warning(f"Average overhead: {average_overhead}")
    return samples

def main():
    logging.getLogger().setLevel(logging.WARNING)
    # import json
    crowdnav = CrowdNav(auto_start=True, start_services=True)
    strategy = CrowdNavBaseStrategy(crowdnav)

    if not crowdnav.wait_for_api_ready():
        logging.error("Could not connect to API, skipping status check")
        return

    # base_config = {
    #     "exploration_percentage": 0.0,
    #     "route_random_sigma": 0.2,
    #     "max_speed_and_length_factor": 1,
    #     "average_edge_duration_factor": 1,
    #     "freshness_update_factor": 10,
    #     "freshness_cut_off_value": 90,
    #     "re_route_every_ticks": 60,
    #     "edge_average_influence": 50,
    #     "total_car_counter": 750,
    # }

    # strategy.execute(base_config)
    
    samples = run_simulation(crowdnav, strategy, 750)
    logging.info(f"Samples: {samples}")

    crowdnav.stop_container()


if __name__ == "__main__":
    main()
