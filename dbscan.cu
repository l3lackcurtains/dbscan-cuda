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

#define DATASET_COUNT 5000
#define DIMENSION 2

#define MAX_SEEDS 1024

#define THREAD_BLOCKS 128
#define THREAD_COUNT 256

#define UNPROCESSED -1
#define NOISE -2

using namespace std;

__device__ __constant__ int minpts = 4;
__device__ __constant__ double eps = 1.5;

int importDataset(char *fname, double *dataset);

bool monitorSeedPoints(int *d_clusters, int *d_seedList,
                       int *d_currentSeedLength, bool *collisionMatrix,
                       vector<int> &pointsRemaining, int *clusterCount);

__global__ void DBSCAN(double *dataset, int *clusters, int *seedList,
                       int *currentSeedLength, bool *collisionMatrix);

__device__ void processObject(int pointID, int comparingPointID,
                              int *neighborsCount, int *neighbors,
                              int *seedList, double *dataset);

__device__ void markAsCandidate(int neighborID, int chainID, int *clusters,
                                bool *collisionMatrix, int *currentSeedLength,
                                int *seedList);

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Provide dataset file in the argument");
    return 0;
  }
  char inputFname[500];
  strcpy(inputFname, argv[1]);

  printf("Dataset file: %s\n", inputFname);

  double *h_dataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);
  int ret = importDataset(inputFname, h_dataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // CPU Variable declarations
  vector<int> pointsRemaining;

  // GPU Variable declarations
  double *d_dataset;
  int *d_clusters;
  int *d_seedList;
  int *d_currentSeedLength;
  bool *d_collisionMatrix;

  // GPU Memory allocation
  cudaMalloc((void **)&d_dataset, sizeof(double) * DATASET_COUNT * DIMENSION);
  cudaMalloc((void **)&d_clusters, sizeof(int) * DATASET_COUNT);
  cudaMalloc((void **)&d_seedList, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  cudaMalloc((void **)&d_currentSeedLength, sizeof(int) * THREAD_BLOCKS);
  cudaMalloc((void **)&d_collisionMatrix,
             sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  // Initializing data in dataset
  for (int x = 0; x < DATASET_COUNT; x++) {
    pointsRemaining.push_back(x);
  }

  printf("Dataset has %d data. \n", pointsRemaining.size());

  // Initialize GPU variables with data
  cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  cudaMemset(d_currentSeedLength, 0, sizeof(int) * THREAD_BLOCKS);
  cudaMemset(d_clusters, UNPROCESSED, sizeof(int) * DATASET_COUNT);
  cudaMemset(d_collisionMatrix, false,
             sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);
  cudaMemcpy(d_dataset, h_dataset, sizeof(double) * DATASET_COUNT * DIMENSION,
             cudaMemcpyHostToDevice);

  // Start the DBSCAN process between CPU AND GPU
  int clusterCount = 0;
  bool exit = false;

  while (!exit) {
    bool completed =
        monitorSeedPoints(d_clusters, d_seedList, d_currentSeedLength,
                          d_collisionMatrix, pointsRemaining, &clusterCount);

    if (completed) {
      exit = true;
    }

    printf("Number of clusters %d, points remaining: %d\n", clusterCount,
           pointsRemaining.size());

    if (exit) break;

    /******************************************************
     *  Run DBSCAN Kernel
     ******************************************************
     */

    cudaDeviceSynchronize();

    dim3 GRID(THREAD_BLOCKS, 1);
    dim3 BLOCK(THREAD_COUNT, 1);

    DBSCAN<<<GRID, BLOCK>>>(d_dataset, d_clusters, d_seedList,
                            d_currentSeedLength, d_collisionMatrix);

    cudaDeviceSynchronize();

    /******************************************************
     *  /end Run DBSCAN Kernel
     ******************************************************
     */
  }

  printf("Final clusters: %d\n", clusterCount);
}

bool monitorSeedPoints(int *d_clusters, int *d_seedList,
                       int *d_currentSeedLength, bool *d_collisionMatrix,
                       vector<int> &pointsRemaining, int *clusterCount) {
  int *localSeedCount = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  cudaMemcpy(localSeedCount, d_currentSeedLength, sizeof(int) * THREAD_BLOCKS,
             cudaMemcpyDeviceToHost);

  bool completeSeedListFirst = false;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localSeedCount[i] > 0) {
      completeSeedListFirst = true;
      break;
    }
  }

  if (completeSeedListFirst) {
    // printf("Processing from seed list\n");
    return false;
  }

  // printf("Processing from new seed list\n");

  int *localClusters = (int *)malloc(sizeof(int) * DATASET_COUNT);
  cudaMemcpy(localClusters, d_clusters, sizeof(int) * DATASET_COUNT,
             cudaMemcpyDeviceToHost);

  bool *localCollisionMatrix =
      (bool *)malloc(sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
             sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS,
             cudaMemcpyDeviceToHost);

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
    set<int> expansionSet;
    set<int> finalBlockSet;
    expansionSet.insert(currentBlock);
    while (expansionSet.empty() == 0) {
      it = expansionSet.begin();
      int expandBlock = *it;
      expansionSet.erase(it);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if ((localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] ||
             localCollisionMatrix[x * THREAD_BLOCKS + expandBlock]) &&
            blockSet.find(x) != blockSet.end()) {
          finalBlockSet.insert(x);
          expansionSet.insert(x);
        }
      }
    }
    for (it = finalBlockSet.begin(); it != finalBlockSet.end(); ++it) {
      clusterMap[*it] = currentBlock;
    }
  }

  vector<vector<int>> clustersList(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (localClusters[i] >= 0 && localClusters[i] < THREAD_BLOCKS) {
      clustersList[clusterMap[localClusters[i]]].push_back(i);
    }
  }

  for (int i = 0; i < clustersList.size(); i++) {
    if (clustersList[i].size() != 0) (*clusterCount)++;
    for (int j = 0; j < clustersList[i].size(); j++) {
      localClusters[clustersList[i][j]] = *clusterCount + THREAD_BLOCKS;
    }
  }

  int *localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);

  cudaMemcpy(localSeedList, d_seedList, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
             cudaMemcpyDeviceToHost);

  // Initialize with first seed point
  int processedCount = 0;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    bool processed = true;
    while (!pointsRemaining.empty()) {
      int point = pointsRemaining.back();
      pointsRemaining.pop_back();
      if (localClusters[point] == UNPROCESSED) {
        localSeedList[i * MAX_SEEDS] = point;
        localSeedCount[i] = 1;
        processed = false;
        break;
      }
    }
    if (processed) {
      processedCount++;
    }
  }

  if (processedCount == THREAD_BLOCKS) {
    return true;
  }

  cudaMemset(d_collisionMatrix, false,
             sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  cudaMemcpy(d_clusters, localClusters, sizeof(int) * DATASET_COUNT,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_seedList, localSeedList, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
             cudaMemcpyHostToDevice);

  cudaMemcpy(d_currentSeedLength, localSeedCount, sizeof(int) * THREAD_BLOCKS,
             cudaMemcpyHostToDevice);

  return false;
}

__global__ void DBSCAN(double *dataset, int *clusters, int *seedList,
                       int *currentSeedLength, bool *collisionMatrix) {
  __shared__ int pointID;

  __shared__ int neighbors[MAX_SEEDS];

  __shared__ int neighborsCount;

  int chainID = blockIdx.x;

  int seedLength = currentSeedLength[chainID];

  // If seedlength is 0 then exit
  if (seedLength == 0) return;

  pointID = seedList[chainID * MAX_SEEDS + seedLength - 1];

  // If the point is already processed then exit
  __shared__ bool processed;
  processed = false;

  if (threadIdx.x == 0) {
    currentSeedLength[chainID] = currentSeedLength[chainID] - 1;
    if (clusters[pointID] != UNPROCESSED) processed = true;
  }
  __syncthreads();
  if (processed) return;

  neighborsCount = 0;
  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    processObject(pointID, i, &neighborsCount, neighbors, seedList, dataset);
  }

  __syncthreads();

  if (neighborsCount >= minpts) {
    clusters[pointID] = chainID;
    for (int i = threadIdx.x; i < neighborsCount; i = i + THREAD_COUNT) {
      markAsCandidate(neighbors[i], chainID, clusters, collisionMatrix,
                      currentSeedLength, seedList);
    }
  } else {
    clusters[pointID] = NOISE;
  }

  __syncthreads();
}

__device__ void processObject(int pointID, int comparingPointID,
                              int *neighborsCount, int *neighbors,
                              int *seedList, double *dataset) {
  register double comparingPoint[DIMENSION], point[DIMENSION];

  for (int i = 0; i < DIMENSION; i++) {
    point[i] = dataset[pointID * DIMENSION + i];
    comparingPoint[i] = dataset[comparingPointID * DIMENSION + i];
  }

  register double distance = 0.0;
  for (int x = 0; x < DIMENSION; x++) {
    distance += (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
  }
  distance = sqrtf(distance);

  if (distance <= eps) {
    register int currentNeighbor = atomicAdd(neighborsCount, 1);
    neighbors[currentNeighbor] = comparingPointID;
  }
}

__device__ void markAsCandidate(int neighborID, int chainID, int *clusters,
                                bool *collisionMatrix, int *currentSeedLength,
                                int *seedList) {
  register int oldState =
      atomicCAS(&clusters[neighborID], UNPROCESSED, chainID);

  if (oldState == UNPROCESSED) {
    register int newSeedLength = atomicAdd(&currentSeedLength[chainID], 1);
    if (newSeedLength < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + newSeedLength] = neighborID;
    }
  } else if (oldState != NOISE && oldState != chainID &&
             oldState < THREAD_BLOCKS) {
    if (chainID > oldState) {
      collisionMatrix[chainID * THREAD_BLOCKS + oldState] = true;
    } else {
      collisionMatrix[oldState * THREAD_BLOCKS + chainID] = true;
    }
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