import os
from utils import parse_args, SAE_pipeline


if __name__ == '__main__':
    cfg = parse_args()
    cfg.model_path = os.path.normpath(cfg.model_path)
    cfg.train_data_path = os.path.normpath(cfg.pipe_data_path[0])
    cfg.eval_data_path = os.path.normpath(cfg.pipe_data_path[1])
    cfg.apply_data_path = os.path.normpath(cfg.pipe_data_path[2])

    pipeline = SAE_pipeline(cfg)
    pipeline.run()