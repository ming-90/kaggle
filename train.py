import torch
import random
import numpy as np
import os

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True   # 확정적 연산 사용
torch.backends.cudnn.benchmark = False      # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False        # cudnn 사용 해제