"""Entry point for running the package as a module."""
import uvicorn
import yaml

CONFIG_PATH = "configs/config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    server_config = config.get("server", {})
    
    uvicorn.run(
        "src.main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False)
    )