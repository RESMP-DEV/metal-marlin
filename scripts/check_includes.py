import torch.utils.cpp_extension
print("Torch include paths:")
for p in torch.utils.cpp_extension.include_paths():
    print(p)
