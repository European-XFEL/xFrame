# Data Selection 
The `DataSelection` class takes the follwing arguments:

- `run` : (int) Run number
- `cells` : (slice|np.array) 
    - `cells_mode` : ('exact' | 'relative') Whether to interpret the specified values as exact cell ids or as indices of the array of all present cell ids.
- `pulses` : (slice|np.array) 
    - `pulses_mode` : ('exact' | 'relative') Whether to interpret the specified values as exact pulse ids or as indices of the array of all present pulse ids.
- `trains` : (slice|np.array) 
    - `trains_mode` : ('exact' | 'relative') Whether to interpret the specified values as exact train ids or as indices of the array of all present train ids.
- `module` : (sub array of np.arange(16)) AGIPD modules to load data from
- `n_frames` : (int) The maximu amount of patterns to load
