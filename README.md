# sac_method
N-point spherical configuration generation with spherical area coordinate method

See preprint.pdf for details.

## Requirements
- python >= 3.9.0
- numpy >= 1.20.1
- matplotlib >= 3.4.1
- numba >= 0.53.0
- scipy >= 1.7.0

## Running (examples):
    python sac_method.py --mn 1,1 2,0 2,0 2,0 2,0 2,0 --plot
    python sac_method.py --mn 10,0 --plot
    python sac_method.py --mn 4,0 4,0 --plot

Example image generated with:

    python sac_method.py --mn 1,1 4,0 4,0 --plot
![SAC method example](https://github.com/bsxfun/sac_method/blob/main/sac_example.png?raw=true)
