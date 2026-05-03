import logging
import torch.utils.cpp_extension


logger = logging.getLogger(__name__)

print("Torch include paths:")
for p in torch.utils.cpp_extension.include_paths():
    print(p)
