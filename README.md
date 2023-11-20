# parallel-boruvka

# Test with Gaussian Random Noise

> Graph test data available at [here](https://algs4.cs.princeton.edu/43mst/).

Sequential - Without optimization - 1,000,000 vertex - 5.73 seconds
Sequential - Path compression as iterative func - 1,000,000 vertex - 5.35 seconds
Sequential - Reduce findParentIdx calls - 1,000,000 vertex - 4.37 seconds
Sequential - Reduce nearest reassignment - 1,000,000 vertex - 4.12 seconds

# TODO

- Convert all to 1D array in core dist wrapper.
- Use GPU for core dist calc
- Free mem