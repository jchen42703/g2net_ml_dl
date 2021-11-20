if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from g2net.utils.config_reader import load_config
    from g2net.inference import Inferrer, create_test_transforms, create_test_dataloader

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path",
                        type=str,
                        required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    cfg = load_config(args.yml_path)["infer_config"]
    print("CONFIG: \n", cfg)

    # Creates the subset of the train.csv to load data for testing
    # Works since we only used a subset of the actual training set for
    # training the model.
    test_path = os.path.join(cfg["dset_dir"], "train.csv")
    start = cfg["idx_start"]
    end = cfg["idx_end"]
    test_df = pd.read_csv(test_path).iloc[start:end]

    transforms = create_test_transforms()
    test_loader = create_test_dataloader(test_df, batch_size=cfg["batch_size"])
    inferrer = Inferrer(test_loader,
                        cfg["base_model_paths"],
                        cfg["filter_model_paths"],
                        threshold=cfg["threshold"])
    inferrer.infer_all()
    csv_path = os.path.join(cfg["export_dir"], "inference_metrics.csv")
    inferrer.metrics.to_csv(csv_path)