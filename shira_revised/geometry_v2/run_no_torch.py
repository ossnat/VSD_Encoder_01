"""Run a repo script in an environment without torch (the Linux VM used here).
Scripts 11 / 20 / 27 only import torch through src/DL_features/__init__ (backbone); they never call it.
Usage: python shira_revised/geometry_v2/run_no_torch.py scripts/11_....py [args]
On the Mac with scripts/py (torch installed) just run the scripts directly."""
import sys, runpy, types
from unittest import mock
for name in ("torch", "torch.nn", "torch.nn.functional", "torchvision", "torchvision.models", "torchvision.transforms"):
    sys.modules.setdefault(name, mock.MagicMock(name=name))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
