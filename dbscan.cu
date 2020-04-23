#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#define DATASET_COUNT 1000
#define DIMENSION 2

#define MAX_SEEDS 1024

#define THREAD_BLOCKS 128
#define THREAD_COUNT 256

#define UNPROCESSED -1
#define NOISE -2

using namespace std;

__device__ __constant__ int minPts = 4;
__device__ __constant__ double eps = 1.5;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

int importDataset(char *fname, double *dataset);

bool monitorPoints(vector<int> &uprocessedPoints, int *clusterCount,
                   int *d_cluster, int *d_seedList, int *currentSeedLength,
                   bool *d_collisionMatrix);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *currentSeedLength, bool *collisionMatrix);

__device__ void markAsCandidate(int neighborId, int chainID, int *cluster,
                                int *seedList, int *currentSeedLength,
                                bool *collisionMatrix);
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Provide dataset file in the argument");
    return 0;
  }
  char inputFname[500];
  strcpy(inputFname, argv[1]);

  printf("Dataset file: %s\n", inputFname);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);
  int ret = importDataset(inputFname, importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // CPU Variable declarations
  vector<int> unprocessedPoints;

    // Initializing data in dataset
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }

  printf("Dataset has %d data. \n", unprocessedPoints.size());

  double *d_dataset;
  int *d_cluster;
  int *d_seedList;
  int *d_currentSeedLength;
  bool *d_collisionMatrix;

  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));
  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(
      cudaMalloc((void **)&d_currentSeedLength, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));
  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(cudaMemset(d_currentSeedLength, 0, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_collisionMatrix, false,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  // Start the DBSCAN process between CPU AND GPU
  int clusterCount = 0;
  bool exit = false;

  while (!exit) {
    bool completed =
        monitorPoints(unprocessedPoints, &clusterCount, d_cluster, d_seedList,
                      d_currentSeedLength, d_collisionMatrix);

    if (completed) {
      exit = true;
    }

    printf("Number of cluster %d, unprocessed points: %d\n", clusterCount,
           unprocessedPoints.size());

    if (exit) break;

    /******************************************************
     *  Run DBSCAN Kernel
     ******************************************************
     */

    cudaDeviceSynchronize();

    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_dataset, d_cluster, d_seedList,
                            d_currentSeedLength, d_collisionMatrix);

    cudaDeviceSynchronize();

    /******************************************************
     *  /end Run DBSCAN Kernel
     ******************************************************
     */
  }

  printf("Final cluster: %d\n", clusterCount);
}

bool monitorPoints(vector<int> &uprocessedPoints, int *clusterCount,
                   int *d_cluster, int *d_seedList, int *d_currentSeedLength,
                   bool *d_collisionMatrix) {
  int *localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localSeedLength, d_currentSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  bool completeSeedListFirst = false;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localSeedLength[i] > 0) {
      completeSeedListFirst = true;
      break;
    }
  }
  if (completeSeedListFirst) return false;

  int *localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost));

  bool *localCollisionMatrix =
      (bool *)malloc(sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost));

  map<int, int> clusterMap;
  set<int> blockSet;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    clusterMap[i] = i;
    blockSet.insert(i);
  }

  set<int>::iterator it;
  while (blockSet.empty() == 0) {
    it = blockSet.begin();
    int currentBlock = *it;
    set<int> expansionQueue;
    set<int> finalQueue;
    expansionQueue.insert(currentBlock);
    while (expansionQueue.empty() == 0) {
      it = expansionQueue.begin();
      int expandBlock = *it;
      expansionQueue.erase(it);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if ((localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] ||
             localCollisionMatrix[x * THREAD_BLOCKS + expandBlock]) &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      clusterMap[*it] = currentBlock;
    }
  }

  vector<vector<int>> clusters(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (localCluster[i] >= 0 && localCluster[i] < THREAD_BLOCKS) {
      clusters[clusterMap[localCluster[i]]].push_back(i);
    }
  }

  for (int i = 0; i < clusters.size(); i++) {
    if (clusters[i].size() != 0) (*clusterCount)++;
    ;
    for (int x = 0; x < clusters[i].size(); x++) {
      localCluster[clusters[i][x]] = *clusterCount + THREAD_BLOCKS + 1;
    }
  }

  gpuErrchk(cudaMemset(d_collisionMatrix, false,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  int *localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    while (!uprocessedPoints.empty()) {
      int lastPoint = uprocessedPoints.back();
      uprocessedPoints.pop_back();
      if (localCluster[lastPoint] == UNPROCESSED) {
        localSeedLength[i] = 1;
        localSeedList[i * MAX_SEEDS] = lastPoint;
        break;
      }
    }
  }

  gpuErrchk(cudaMemcpy(d_currentSeedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_cluster, localCluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  if (uprocessedPoints.empty()) return true;

  return false;
}

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *currentSeedLength, bool *collisionMatrix) {
  __shared__ int pointID;

  __shared__ double point[DIMENSION];

  __shared__ int neighbors[MAX_SEEDS];

  __shared__ int neighborsCount;

  neighborsCount = 0;

  int chainID = blockIdx.x;

  int seedLength = currentSeedLength[chainID];

  // If seedlength is 0 then exit
  if (seedLength == 0) return;

  pointID = seedList[chainID * MAX_SEEDS + seedLength - 1];

  for (int x = 0; x < DIMENSION; x++) {
    point[x] = dataset[pointID * DIMENSION + x];
  }

  // If the point is already processed then exit
  __shared__ bool processed;
  processed = false;

  if (threadIdx.x == 0) {
    currentSeedLength[chainID] = currentSeedLength[chainID] - 1;
    if (cluster[pointID] != UNPROCESSED) processed = true;
  }
  __syncthreads();

  if (processed) return;

  __syncthreads();

  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    register double comparingPoint[DIMENSION];
    for (int x = 0; x < DIMENSION; x++) {
      comparingPoint[x] = dataset[i * DIMENSION + x];
    }

    register double distance = 0.0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }
    distance = sqrtf(distance);

    if (distance <= eps) {
      register int currentNeighbor = atomicAdd(&neighborsCount, 1);
      if (neighborsCount >= minPts) {
        markAsCandidate(i, chainID, cluster, seedList, currentSeedLength,
                        collisionMatrix);
      } else {
        neighbors[currentNeighbor] = i;
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0 && neighborsCount > minPts) {
    neighborsCount = minPts;
  }

  __syncthreads();

  if (neighborsCount >= minPts) {
    cluster[pointID] = chainID;
    for (int i = threadIdx.x; i < neighborsCount; i = i + THREAD_COUNT) {
      markAsCandidate(neighbors[i], chainID, cluster, seedList,
                      currentSeedLength, collisionMatrix);
    }
  } else {
    cluster[pointID] = NOISE;
  }

  __syncthreads();

  if (currentSeedLength[chainID] >= MAX_SEEDS) {
    currentSeedLength[chainID] = MAX_SEEDS - 1;
  }
}

__device__ void markAsCandidate(int neighborId, int chainID, int *cluster,
                                int *seedList, int *currentSeedLength,
                                bool *collisionMatrix) {
  register int oldState =
      atomicCAS(&(cluster[neighborId]), UNPROCESSED, chainID);

  if (oldState == UNPROCESSED) {

    register int h = atomicAdd(&(currentSeedLength[chainID]), 1);
    if (h < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + h] = neighborId;
    }
  } else if (oldState != NOISE && oldState != chainID &&
             oldState < THREAD_BLOCKS) {
    if (oldState < chainID) {
      collisionMatrix[oldState * THREAD_BLOCKS + chainID] = true;
    } else {
      collisionMatrix[chainID * THREAD_BLOCKS + oldState] = true;
    }
  } else if (oldState == NOISE) {
    atomicCAS(&(cluster[neighborId]), NOISE, chainID);
  }
}

int importDataset(char *fname, double *dataset) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }
  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
    char *field = strtok(buf, ",");
    double tmp;
    sscanf(field, "%lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;
    while (field) {
      field = strtok(NULL, ",");
      if (field != NULL) {
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}