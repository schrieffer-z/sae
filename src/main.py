import os
from utils import parse_args, SAE_pipeline


if __name__ == '__main__':
    cfg = parse_args()
    cfg.model_path = os.path.normpath(cfg.model_path)
    cfg.train_data_path = os.path.normpath(cfg.train_data_path)
    cfg.eval_data_path = os.path.normpath(cfg.eval_data_path)
    cfg.apply_data_path = os.path.normpath(cfg.apply_data_path)

    pipeline = SAE_pipeline(cfg)
    pipeline.run()