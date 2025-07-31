import os
import configparser
import logging
from typing import List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up configuration
config_file_path = os.path.join(os.path.dirname(__file__), "config.ini")
if not os.path.exists(config_file_path):
    logger.error("Config file not found at path: {}".format(config_file_path))
    raise SystemExit

class Config:
    """
    Class to manage configuration settings.

    Reads settings from an INI file and provides methods to access various configuration options.
    """
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)

    def __str__(self):
        return self.config.as_string()

    def get_section(self, section: str) -> Optional[dict]:
        """
        Get parameters for a specific section.

        Args:
            section (str): Section name in the configuration file.

        Returns:
            Optional[dict]: Dictionary of key-value pairs for the section if it exists, otherwise None.
        """
        if self.config.has_section(section):
            config_dict = {key: self.config.get(section, key) for key in self.config.options(section)}
            return config_dict
        else:
            return None

    def get_value(self, section: str, option: str) -> Optional[str]:
        """
        Get a specific option value from the configuration.

        Args:
            section (str): Section name in the configuration file.
            option (str): Option name within the section.

        Returns:
            Optional[str]: Value of the option if it exists, otherwise None.
        """
        if self.config.has_option(section, option):
            return self.config.get(section, option)
        else:
            return None

    def get_int(self, section: str, option: str) -> Optional[int]:
        """
        Get an integer value from the configuration.

        Args:
            section (str): Section name.
            option (str): Option name.

        Returns:
            Optional[int]: Integer value or None if not found/invalid.
        """
        value = self.get_value(section, option)
        if value is not None:
            return int(value)
        return None

    def get_float(self, section: str, option: str) -> Optional[float]:
        """
        Get a float value from the configuration.

        Args:
            section (str): Section name.
            option (str): Option name.

        Returns:
            Optional[float]: Float value or None if not found/invalid.
        """
        value = self.get_value(section, option)
        if value is not None:
            return float(value)
        return None

    def get_bool(self, section: str, option: str) -> Optional[bool]:
        """
        Get a boolean value from the configuration.

        Args:
            section (str): Section name.
            option (str): Option name.

        Returns:
            Optional[bool]: Boolean value or None if not found/invalid.
        """
        value = self.get_value(section, option)
        if value is not None:
            return value.lower() == "true"
        return None

class ModelConfig:
    """Class to manage model-specific configuration settings."""

    def __init__(self):
        self.config = Config()
        self._check_config()

    def _check_config(self):
        """Check if required configuration options are present."""
        required_options = [
            "model_name",
            "input_dim",
            "hidden_dims",
            "output_dim",
            "activation",
            "dropout_rate",
            "batch_size",
            "learning_rate",
            "momentum",
            "weight_decay",
            "nesterov",
            "epochs",
            "train_data_path",
            "val_data_path",
            "test_data_path",
            "checkpoint_path",
            "log_path",
            "device",
        ]
        for option in required_options:
            if self.config.get_value("model", option) is None:
                logger.error(f"Required option '{option}' missing in configuration!")
                raise ValueError(f"Missing configuration option: {option}")

    @property
    def model_name(self) -> str:
        return self.config.get_value("model", "model_name")

    @property
    def input_dim(self) -> int:
        return self.config.get_int("model", "input_dim")

    @property
    def hidden_dims(self) -> List[int]:
        hidden_dims_str = self.config.get_value("model", "hidden_dims")
        return [int(dim) for dim in hidden_dims_str.split()]

    @property
    def output_dim(self) -> int:
        return self.config.get_int("model", "output_dim")

    @property
    def activation(self) -> str:
        return self.config.get_value("model", "activation")

    @property
    def dropout_rate(self) -> float:
        return self.config.get_float("model", "dropout_rate")

    @property
    def batch_size(self) -> int:
        return self.config.get_int("training", "batch_size")

    @property
    def learning_rate(self) -> float:
        return self.config.get_float("training", "learning_rate")

    @property
    def momentum(self) -> float:
        return self.config.get_float("training", "momentum")

    @property
    def weight_decay(self) -> float:
        return self.config.get_float("training", "weight_decay")

    @property
    def nesterov(self) -> bool:
        return self.config.get_bool("training", "nesterov")

    @property
    def epochs(self) -> int:
        return self.config.get_int("training", "epochs")

    @property
    def train_data_path(self) -> str:
        return self.config.get_value("data", "train_data_path")

    @property
    def val_data_path(self) -> Optional[str]:
        return self.config.get_value("data", "val_data_path")

    @property
    def test_data_path(self) -> Optional[str]:
        return self.config.get_value("data", "test_data_path")

    @property
    def checkpoint_path(self) -> str:
        return self.config.get_value("paths", "checkpoint_path")

    @property
    def log_path(self) -> str:
        return self.config.get_value("paths", "log_path")

    @property
    def device(self) -> str:
        return self.config.get_value("device", "device")

class DataConfig:
    """Class to manage data-related configuration settings."""

    def __init__(self):
        self.config = Config()

    @property
    def data_dir(self) -> str:
        return self.config.get_value("data", "data_dir")

    @property
    def dataset_name(self) -> str:
        return self.config.get_value("data", "dataset_name")

    @property
    def data_extension(self) -> str:
        return self.config.get_value("data", "data_extension")

class TrainingConfig:
    """Class to manage training-specific configuration settings."""

    def __init__(self):
        self.config = Config()

    @property
    def enable_training(self) -> bool:
        return self.config.get_bool("training", "enable_training")

    @property
    def random_seed(self) -> int:
        return self.config.get_int("training", "random_seed")

    @property
    def shuffle_data(self) -> bool:
        return self.config.get_bool("training", "shuffle_data")

    @property
    def num_workers(self) -> int:
        return self.config.get_int("training", "num_workers")

class EvaluationConfig:
    """Class to manage evaluation-specific configuration settings."""

    def __init__(self):
        self.config = Config()

    @property
    def enable_evaluation(self) -> bool:
        return self.config.get_bool("evaluation", "enable_evaluation")

    @property
    def metrics(self) -> List[str]:
        metrics_str = self.config.get_value("evaluation", "metrics")
        return metrics_str.split(',') if metrics_str else []

    @property
    def threshold(self) -> float:
        return self.config.get_float("evaluation", "threshold")

class ConfigManager:
    """Centralized class to manage all configuration settings."""

    def __init__(self):
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.training_config = TrainingConfig()
        self.evaluation_config = EvaluationConfig()

    @classmethod
    def get_config_paths(cls) -> List[str]:
        """Get paths to configuration files."""
        config_paths = []
        config_files = ["config.ini", "local_config.ini"]
        for config_file in config_files:
            config_path = os.path.join(os.path.dirname(__file__), config_file)
            if os.path.exists(config_path):
                config_paths.append(config_path)
        return config_paths

    @staticmethod
    def merge_configs(configs: List[Config]) -> Config:
        """Merge multiple Config instances into one."""
        merged_config = Config()
        for config in configs:
            for section in config:
                for option in config[section]:
                    merged_config.config.set(section, option, config[section][option])
        return merged_config

    def load_configs(self):
        """Load configuration from file(s) and merge them."""
        config_paths = self.get_config_paths()
        configs = [Config() for _ in config_paths]
        for i, path in enumerate(config_paths):
            logger.debug(f"Loading configuration from {path}...")
            configs[i].config.read(path)

        self.config = self.merge_configs(configs)

    def save_config(self, path: Optional[str]=None) -> None:
        """Save current configuration to a file."""
        if not path:
            path = config_file_path
        with open(path, 'w') as config_file:
            self.config.config.write(config_file)
        logger.info(f"Configuration saved to {path}")

config_manager = ConfigManager()
config_manager.load_configs()

# Make config object accessible globally
config = config_manager.config