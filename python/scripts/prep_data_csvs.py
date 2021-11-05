if __name__ == "__main__":
    import argparse
    from g2net.utils.config_reader import load_config
    from g2net.io.prep_data import create_train_and_test_sub_csvs

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path",
                        type=str,
                        required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    config = load_config(args.yml_path)
    create_train_and_test_sub_csvs(config)