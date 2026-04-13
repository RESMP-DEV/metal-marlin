
def split_kernels():
    with open('contrib/metal_marlin/metal_marlin/kernels.py') as f:
        content = f.read()
    
    # We will create the following modules:
    # - constants.py: TILE_M, TILE_N, etc. and constants block
    # - shaders.py: The raw Metal source strings
    # - dispatcher.py: MarlinGemmDispatcher, etc.
    # - __init__.py: Re-export all
    
    pass

split_kernels()
