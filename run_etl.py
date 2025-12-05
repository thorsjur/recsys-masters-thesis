import argparse
import sys
import logging
from dataset_registry import DATASET_REGISTRY
from util.logging_config import setup_logging

def main():
    """Run the ETL pipeline for the specified dataset configuration."""
    parser = argparse.ArgumentParser(description="Execute the data preprocessing pipeline.")
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        choices=DATASET_REGISTRY.keys(),
        help=f"Specify the dataset configuration key. Available options: {list(DATASET_REGISTRY.keys())}"
    )
    
    parser.add_argument('--debug', action='store_true', help="Enable debug-level logging.")

    args = parser.parse_args()
    setup_logging(
        debug_mode=args.debug,
        log_dir='output/logs/run_etl',
        log_prefix='etl'
    )

    logging.info(f"Starting preprocessing pipeline for '{args.config}'")

    try:
        config_factory = DATASET_REGISTRY[args.config]
        config, loader_class = config_factory()
    except KeyError:
        logging.error(f"Configuration key '{args.config}' was not found in the registry.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)

    logging.info(f"Output directory: {config.output_dir}")
    
    try:
        loader = loader_class(config)
        loader.execute_pipeline()
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    logging.info(f"Pipeline completed for '{args.config}'")

if __name__ == "__main__":
    main()