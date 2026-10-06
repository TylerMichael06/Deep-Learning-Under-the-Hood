"""OBD-II virtual-sensor fault detection (Hoda's model). See README.md and docs/model_build_guide.md."""
import numpy as np, torch

torch.manual_seed(0); np.random.seed(0)   # the original single script did this on import; same seeds -> same results
