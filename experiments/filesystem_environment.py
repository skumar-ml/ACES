import os
from pathlib import Path
from typing import Optional, Union

from agent.src.environment import BaseShoppingEnvironment
from experiments.config import ExperimentData
from experiments.utils.dataset_ops import get_experiment_screenshot_png_path


class FilesystemShoppingEnvironment(BaseShoppingEnvironment):
    """
    A shopping environment that reads screenshots from the filesystem instead of capturing them.
    
    This is useful for replaying experiments or testing with pre-captured screenshots.
    """
    
    def __init__(
        self,
        screenshots_dir: Path,
        query: str,
        experiment_label: str,
        experiment_number: int,
        dataset_name: str,
        remote: bool,
        dataset_csv_path: Optional[str] = None,
    ):
        """
        Initialize the filesystem environment.
        
        Args:
            screenshots_dir: Base directory containing screenshots (used when ``dataset_csv_path`` is None)
            query: The product query (e.g., "mousepad")
            experiment_label: The experiment label (e.g., "control")
            experiment_number: The experiment number
            remote: If True, return GCS URLs instead of bytes
            dataset_csv_path: When set, local PNG path follows :func:`get_experiment_screenshot_png_path`
        """
        self.screenshots_dir = screenshots_dir
        self.remote = remote
        self.dataset_name = dataset_name
        
        # Create ExperimentData object to use standardized path methods
        import pandas as pd
        self.experiment_data = ExperimentData(
            experiment_label=experiment_label,
            experiment_number=experiment_number,
            experiment_df=pd.DataFrame(),  # Empty DataFrame, not needed for path construction
            query=query,
            dataset_name=dataset_name
        )
        
        if dataset_csv_path:
            self.screenshot_path = get_experiment_screenshot_png_path(
                dataset_csv_path, query, experiment_label, experiment_number
            )
        else:
            self.screenshot_path = self.experiment_data.get_local_screenshot_path(screenshots_dir)
    
    def capture_screenshot(self) -> Union[bytes, str]:
        """
        Read a screenshot from the filesystem or return a GCS URL.
        
        Returns:
            Union[bytes, str]: Screenshot data as bytes if remote=False, 
                              or public GCS URL as string if remote=True
            
        Raises:
            FileNotFoundError: If the screenshot file doesn't exist (local mode)
            ValueError: If GCS_BUCKET_NAME environment variable is not set (remote mode)
        """
        if self.remote:
            bucket_name = os.getenv('GCS_BUCKET_NAME')
            if not bucket_name:
                raise ValueError("GCS_BUCKET_NAME environment variable must be set for remote mode")
            
            # Use ExperimentData method to get standardized HTTPS URL
            return self.experiment_data.get_https_screenshot_url(bucket_name)
        else:
            if not self.screenshot_path.exists():
                raise FileNotFoundError(f"Screenshot not found: {self.screenshot_path}")
            
            return self.screenshot_path.read_bytes()