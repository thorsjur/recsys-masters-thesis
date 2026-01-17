import argparse
import sys
import logging
from dataset_registry import DATASET_REGISTRY, get_available_datasets
from util.logging_config import setup_logging

def main():
    """Run the ETL pipeline for the specified dataset configuration."""
    parser = argparse.ArgumentParser(description="Execute the data preprocessing pipeline.")
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        choices=get_available_datasets(),
        help=f"Specify the dataset configuration key. Available options: {get_available_datasets()}"
    )
    
    parser.add_argument('--debug', action='store_true', help="Enable debug-level logging.")
    
    parser.add_argument(
        '--temporal-days',
        type=int,
        default=None,
        help="Generate day-wise splits for temporal experiments (e.g., --temporal-days 30 creates day_1.inter through day_30.inter)"
    )
    
    parser.add_argument(
        '--temporal-hours',
        type=int,
        default=None,
        help="Generate hour-wise splits for temporal experiments (e.g., --temporal-hours 168 creates hour_1.inter through hour_168.inter)"
    )

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
        
        # Set temporal splitting if specified
        if args.temporal_hours and args.temporal_days:
            logging.error("Cannot specify both --temporal-hours and --temporal-days")
            sys.exit(1)
        
        if args.temporal_hours:
            config.temporal_days = (args.temporal_hours, 'hour')
            logging.info(f"Temporal hour-wise splitting enabled: {args.temporal_hours} hours")
        elif args.temporal_days:
            config.temporal_days = args.temporal_days
            logging.info(f"Temporal day-wise splitting enabled: {args.temporal_days} days")
    except KeyError:
        logging.error(f"Configuration key '{args.config}' was not found in the registry.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to initialize configuration: {e}")
        sys.exit(1)

    logging.info(f"Output directory: {config.output_dir}")
    
    try:
        loader = loader_class(config)
        loader.execute_etl_pipeline()
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    
    logging.info(f"Pipeline completed for '{args.config}'")

if __name__ == "__main__":
    main()