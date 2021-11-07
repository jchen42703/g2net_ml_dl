import yaml


def load_config(yml_path):
    """Loads a .yml file.
    Args:
        yml_path (str): Path to a .yaml or .yml file.
    Returns:
        config (dict): parsed .yml config
    """
    with open(yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config