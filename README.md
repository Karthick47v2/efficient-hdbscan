<h1 align="center">Welcome to efficient-hdbscan 👋</h1>

HDBSCAN, which stands for Hierarchical Density-Based Spatial Clustering of Applications with Noise, is a clustering algorithm that extends the capabilities of DBSCAN by allowing it to find clusters of varying densities. This enables HDBSCAN to be more robust to parameter selection and to return meaningful clusters with little or no parameter tuning. It is particularly useful for exploratory data analysis, as it can efficiently handle large datasets and provides fast and reliable clustering results.

This repository contains a parallel implementation of HDBSCAN using OpenMP. By leveraging parallel processing, it can effectively utilize multiple CPU cores, making it suitable for high-performance computing environments and modern multicore systems. Additionally, this implementation is faster than the one provided by scikit-learn.

## Prerequisite

- Python (>=3.8)
- CMake
- OpenMP
- GCC/Clang

## Install

1. Create shared libraries using make.
```
make
```

2. Install required python libraries
```
pip install -r requirements.txt
```

## Usage

Once the code is compiled, you can use the provided Python wrapper to utilize the parallel HDBSCAN implementation. A sample usage is given in `main.py`.


## Author

👤 **Karthick T. Sharma**

- Github: [@Karthick47v2](https://github.com/Karthick47v2)
- LinkedIn: [@Karthick47](https://linkedin.com/in/Karthick47)

## Citation

```
@inproceedings{inproceedings,
author = {Campello, Ricardo and Moulavi, Davoud and Sander, Joerg},
year = {2013},
month = {04},
pages = {160-172},
title = {Density-Based Clustering Based on Hierarchical Density Estimates},
volume = {7819},
isbn = {978-3-642-37455-5},
doi = {10.1007/978-3-642-37456-2_14}
}
```

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/Karthick47v2/efficient-hdbscan/issues).

## Show your support

Give a ⭐️ if this project helped you!