## GPU based DBSCAN algorithm in CUDA
Over the last decade, there has been an ever￾increasing interest in parallel computation. The primary goal of adopting parallel computing is to improve the speed of intensive computation. Density-based clustering algorithms are widely used data mining algorithm used to find the clusters in data without any prior knowledge of its belongings. It is one of the algorithms which require intensive computing and there have been many attempts to improve the performance of it. In this project, The DBSCAN algorithm is implemented in the Compute Unified Device Architecture (CUDA) programming model to achieve the parallelism and high-performance benefits offered by GPU. The result of the DBSCAN algorithm executed in parallel GPU is expected to be similar to the sequential CPU implementation. The demonstration of comparison in the performance between the sequential and parallel implementation of this algorithm is the main motive for the project.

## How to run the algorithm
### CPU based algorithm
```bash
g++ ./dbscan_cpu.cpp -o ./dbscan_cpu.exe

./dbscan_cpu.exe ./dataset/dataset.txt
```

### GPU based algorithm
```bash
nvcc -O3 ./dbscan_gpu.cu -o ./dbscan_gpu.exe

./dbscan_gpu.exe ./dataset/dataset.txt
```
It expect location of dataset file in it's first argument.

## DBSCAN result comparision
Using the full dataset, the result from two algorithm is given below.

### CPU algorithm result
* Number of clusters: 764
* Number of Noises: 123

### GPU algorithm result
* Number of clusters: 756
* Number of Noises: 103

*The deviation in GPU based algorithm is because of parallel execution of cluster expansion and extra merging of clusters*