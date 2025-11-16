import logging
from UPISAS.exemplar import Exemplar
from UPISAS import get_response_for_get_request

logging.getLogger().setLevel(logging.INFO)


class CrowdNav(Exemplar):
    def __init__(self, auto_start: bool = False, container_name: str = "crowdnav-api", start_services: bool = False, docker_compose_file: str = "/home/hreft/uni/selfada/Crowdnav_Grp2/docker-compose.upisas.yml"):
        self._container_name = container_name
        self.base_endpoint = "http://localhost:8080"
        self.http_server_dockerfile = "/home/hreft/uni/sa/Crowdnav_Grp2/api/Dockerfile.upisas"
        self.http_server_name = "http-server-group-6_4"
        self.docker_compose_file = docker_compose_file

        # self.build_api_image(self.http_server_dockerfile, self.http_server_name)

        crowdnav_docker_kwargs = {
            "name": container_name,
            "image": self.http_server_name,
            "ports": {8080: 8080},
            "network": "fas-net",
            "detach": True
        }
        # if start_services: self.start_services()
        # super().__init__(self.base_endpoint, crowdnav_docker_kwargs, auto_start)
        # if auto_start: self.start_run()
        logging.info("CrowdNav exemplar initialized")
    
    def start_services(self, docker_compose_file: str = "/home/hreft/uni/sa/Crowdnav_Grp2/docker-compose.upisas.yml"):
        """Start docker-compose services."""
        import subprocess
        subprocess.run(["docker", "compose", "-f", docker_compose_file, "up", "-d"], check=True)
        logging.info(f"Docker compose services started from {docker_compose_file}")

    def build_api_image(self, dockerfile: str, name: str):
        import subprocess
        # check if image exists
        try:
            subprocess.run(["docker", "image", "inspect", name], check=True)
            logging.info(f"API image {name} exists")
            return
        except subprocess.CalledProcessError:
            logging.info(f"API image {name} does not exist")
        
        subprocess.run(["docker", "build", "-f", dockerfile, "-t", name, "/home/hreft/uni/sa/Crowdnav_Grp2/api"], check=True)
        logging.info(f"API image built from {dockerfile} and named {name}")

    def wait_for_api_ready(self, max_retries: int = 30, delay: int = 2):
        """Wait for the API to be ready before making requests."""
        import time
        import requests
        
        logging.info("Waiting for API to be ready...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_endpoint}/monitor", timeout=2)
                if response.status_code == 200:
                    logging.info("✓ API is ready!")
                    return True
            except (requests.exceptions.RequestException, Exception) as e:
                if i < max_retries - 1:
                    logging.info(f"Waiting for API... ({i+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logging.error(f"API failed to become ready after {max_retries} retries")
                    return False
        return False
    
    def start_run(self):
        logging.info("Starting FastAPI server inside the container...")
        self.exemplar_container.exec_run(cmd='uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload', detach=True)
        logging.info("FastAPI server started inside container")

    def stop_container(self):
        """
        Stop the container (no-op for external containers).
        """
        logging.info("Stopping CrowdNav services")
        self.stop_services()
        logging.info("Stopping CrowdNav container")
        # super().stop_container()
        logging.info("CrowdNav container stopped")
    
    def stop_services(self):
        import subprocess
        subprocess.run(["docker", "compose", "-f", self.docker_compose_file, "down"], check=True)
        logging.info(f"Docker compose services stopped from {self.docker_compose_file}")
    
    def _perform_get_request(self, endpoint_suffix: str):
        url = f"{self.base_endpoint}/{endpoint_suffix}"
        response = get_response_for_get_request(url)
        return response.json()
    
    def get_current_status(self):
        import json
        try:
            monitor_data = self._perform_get_request("monitor")
            logging.info("================================================")
            logging.info(f"Monitor data: {json.dumps(monitor_data, indent=4)}")
            logging.info("================================================")

            car_stats = monitor_data.get("car_stats", {})
            configs = monitor_data.get("configs", {})
            
            status = {
                "total_trips": car_stats.get("total_trips", 0),
                "avg_trip_time": car_stats.get("total_trip_average", 0),
                "avg_overhead": car_stats.get("total_trip_overhead_average", 0),
                "complaints": configs.get("total_complaints", 0),
                "cars_driving": car_stats.get("driving_car_counter", 0),
                "step": car_stats.get("step", 0)
            }
            
            return status
        except Exception as e:
            logging.error(f"Failed to get current status: {e}")
            return {}

if __name__ == "__main__":
    import json
    crowdnav = CrowdNav(auto_start=True, start_services=True)
    
    # Wait for API to be ready before making requests
    if crowdnav.wait_for_api_ready():
        status = crowdnav.get_current_status()
        logging.info("================================================")
        logging.info(f"Status: {json.dumps(status, indent=4)}")
        logging.info("================================================")
    else:
        logging.error("Could not connect to API, skipping status check")
    
    crowdnav.stop_container()