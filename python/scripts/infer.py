if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from g2net.utils.config_reader import load_config
    from g2net.inference import GlobalEvaluator, Inferrer, create_test_transforms, create_test_dataloader
    from glob import glob
    import pprint

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path",
                        type=str,
                        required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    cfg = load_config(args.yml_path)["infer_config"]
    pp = pprint.PrettyPrinter(depth=4)
    print("CONFIG: \n")
    pp.pprint(cfg)

    # Creates the subset of the train.csv to load data for testing
    # Works since we only used a subset of the actual training set for
    # training the model.
    test_path = os.path.join(cfg["dset_dir"], "train.csv")
    start = cfg["idx_start"]
    end = cfg["idx_end"]
    test_df = pd.read_csv(test_path).iloc[start:end]

    transforms = create_test_transforms()
    test_loader = create_test_dataloader(test_df,
                                         batch_size=cfg["batch_size"],
                                         test_transforms=transforms)
    inferrer = Inferrer(test_loader,
                        cfg["base_model_paths"],
                        cfg["filter_model_paths"],
                        cfg["export_dir"],
                        threshold=cfg["threshold"],
                        cpu_only=cfg["cpu_only"])
    inferrer.infer_all()
    csv_path = os.path.join(cfg["export_dir"], "inference_times.csv")
    inferrer.metrics.to_csv(csv_path)

    base_model_paths = glob(
        os.path.join(cfg["export_dir"], "base_preds", "*.npy"))
    filter_preds_paths = glob(
        os.path.join(cfg["export_dir"], "filter_preds", "*.npy"))
    both_preds_paths = glob(
        os.path.join(cfg["export_dir"], "both_preds", "*.npy"))

    evaluator = GlobalEvaluator(test_loader, base_model_paths,
                                filter_preds_paths, both_preds_paths)
    evaluator.evaluate_all()
    csv_path = os.path.join(cfg["export_dir"], "inference_metrics.csv")
    evaluator.metrics.to_csv(csv_path)