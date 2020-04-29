#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

using namespace std;

#define DATASET_COUNT 10000
#define DIMENSION 2
#define MAX_SEEDS 1024
#define REFILL_MAX_SEEDS 1024
#define THREAD_BLOCKS 32
#define THREAD_COUNT 128
#define UNPROCESSED -1
#define NOISE -2

__device__ __constant__ int minPts = 4;
__device__ __constant__ float eps = 1.5;

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

int ImportDataset(char const *fname, float *dataset);
bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *clusterCount,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       bool *d_collisionMatrix);
void FinalizeClusters(int *cluster, int *clusterCount, bool *d_collisionMatrix);
__global__ void DBSCAN(float *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, bool *collisionMatrix);
__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *refillSeedList, int *refillSeedLength,
                                bool *collisionMatrix);
__global__ void showClusters(int *cluster);


int step;
int main() {
  float *importedDataset =
      (float *)malloc(sizeof(float) * DATASET_COUNT * DIMENSION);

  int ret = ImportDataset("./dataset/dataset.txt", importedDataset);

  for (int i = 0; i < 2; i++) {
    printf("Sample Data %f\n", importedDataset[i]);
  }

  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }
  printf("Imported %d data in dataset\n", unprocessedPoints.size());

  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  float *d_dataset;
  d_dataset = (float *)malloc(sizeof(float) * DATASET_COUNT * DIMENSION);

  int *d_cluster;
  d_cluster = (int *)malloc(sizeof(int) * DATASET_COUNT);

  int *d_seedList;
  d_seedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);

  int *d_seedLength;
  d_seedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);

  int *d_refillSeedList;
  d_refillSeedList =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS);

  int *d_refillSeedLength;
  d_refillSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);

  bool *d_collisionMatrix;
  d_collisionMatrix =
      (bool *)malloc(sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  /**
   **************************************************************************
   * Memory allocation
   **************************************************************************
   */

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(float) * DATASET_COUNT * DIMENSION));
  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_refillSeedList,
                       sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS));
  gpuErrchk(
      cudaMalloc((void **)&d_refillSeedLength, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  /**
   **************************************************************************
   * Assignment with default values
   **************************************************************************
   */
  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(float) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));
  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_refillSeedList, -1,
                       sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS));
  gpuErrchk(cudaMemset(d_refillSeedLength, 0, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_collisionMatrix, false,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  int clusterCount = 0;
  bool exit = false;
  step = 0;
  while (!exit) {
    // step++;
    // printf("Step %d\n", step);


    bool completed = MonitorSeedPoints(
        unprocessedPoints, &clusterCount, d_cluster, d_seedList, d_seedLength,
        d_refillSeedList, d_refillSeedLength, d_collisionMatrix);

    if (completed) {
      exit = true;
    }

    // printf("Number of cluster %d, unprocessed points: %d\n", clusterCount,
    //        unprocessedPoints.size());

    if (exit) break;

    gpuErrchk(cudaDeviceSynchronize());

    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_refillSeedList,
        d_refillSeedLength, d_collisionMatrix);

    gpuErrchk(cudaDeviceSynchronize());
  }

  showClusters<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_cluster);
  gpuErrchk(cudaDeviceSynchronize());

  printf("Final number of cluster %d\n", clusterCount);
}

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *clusterCount,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       bool *d_collisionMatrix) {
  int *localSeedLength;
  localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);

  gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localRefillSeedLength;
  localRefillSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);

  gpuErrchk(cudaMemcpy(localRefillSeedLength, d_refillSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localSeedList;
  localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);

  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  int *localRefillSeedList;
  localRefillSeedList =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS);

  gpuErrchk(cudaMemcpy(localRefillSeedList, d_refillSeedList,
                       sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS,
                       cudaMemcpyDeviceToHost));
  

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localSeedLength[i] >= MAX_SEEDS) {
      localSeedLength[i] = MAX_SEEDS - 1;
    }
    if (localRefillSeedLength[i] >= REFILL_MAX_SEEDS) {
      localRefillSeedLength[i] = REFILL_MAX_SEEDS - 1;
    }
  }

  bool completeSeedListFirst = false;
  bool refilled = false;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localSeedLength[i] > 0) {
      completeSeedListFirst = true;
      break;
    } else {
      if (localRefillSeedLength[i] > 0) {
        // printf("%d Refill, seedLength %d, refill Length %d \n", i,
        // localSeedLength[i], localRefillSeedLength[i]);

        while (localSeedLength[i] < MAX_SEEDS && localRefillSeedLength[i] > 0) {
          localRefillSeedLength[i] = localRefillSeedLength[i] - 1;

          localSeedList[i * MAX_SEEDS + localSeedLength[i]] =
              localRefillSeedList[i * REFILL_MAX_SEEDS +
                                  localRefillSeedLength[i]];

          localSeedLength[i] = localSeedLength[i] + 1;
        }
        // printf("%d Refill, seedLength %d, refill Length %d \n", i,
        // localSeedLength[i], localRefillSeedLength[i]);
        // printf("==========================\n");

        refilled = true;
        break;
      }
    }
  }

  if (refilled) {
    gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                         sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_refillSeedLength, localRefillSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    return false;
  }

  if (completeSeedListFirst) {
    return false;
  }

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);

  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  FinalizeClusters(localCluster, clusterCount, d_collisionMatrix);

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    while (!unprocessedPoints.empty()) {
      int lastPoint = unprocessedPoints.back();
      unprocessedPoints.pop_back();
      if (localCluster[lastPoint] == UNPROCESSED) {
        localSeedLength[i] = 1;
        localSeedList[i * MAX_SEEDS] = lastPoint;
        break;
      }
    }
  }

  // printf("Remaining %d\n", unprocessedPoints.size());
  // for (int i = 0; i < THREAD_BLOCKS; i++) {
  //   printf("Seed list %d\n", localSeedList[i * MAX_SEEDS]);
  //   printf("Seed Length %d\n", localSeedLength[i]);
  // }

  gpuErrchk(cudaMemcpy(d_cluster, localCluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  if (unprocessedPoints.empty()) return true;

  return false;
}

void FinalizeClusters(int *cluster, int *clusterCount,
                      bool *d_collisionMatrix) {
  bool *localCollisionMatrix;
  localCollisionMatrix =
      (bool *)malloc(sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS);

  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost)); 

map<int, int> clusterMap;
set<int> blockSet;
for (int i = 0; i < THREAD_BLOCKS; i++) {
  blockSet.insert(i);
}

set<int>::iterator it;
while (blockSet.empty() == 0) {
  it = blockSet.begin();
  int curBlock = *it;
  set<int> expansionQueue;
  set<int> finalQueue;
  expansionQueue.insert(curBlock);
  finalQueue.insert(curBlock);
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
    clusterMap[*it] = curBlock;
  }
}

vector<vector<int>> localClustersList(THREAD_BLOCKS, vector<int>());
for (int i = 0; i < DATASET_COUNT; i++) {
  if (cluster[i] >= 0 && cluster[i] < THREAD_BLOCKS) {
    localClustersList[clusterMap[cluster[i]]].push_back(i);
  }
}

for (int i = 0; i < localClustersList.size(); i++) {
  if (localClustersList[i].size() < 5) continue;
  for (int x = 0; x < localClustersList[i].size(); x++) {
    cluster[localClustersList[i][x]] = *clusterCount + THREAD_BLOCKS + 1;
  }
  (*clusterCount)++;
}
// for(int i = 0; i < DATASET_COUNT; i++) {
//   printf("Point %d, cluster %d\n", i, cluster[i]);
// }

gpuErrchk(cudaMemset(d_collisionMatrix, false,
                     sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));
}

__global__ void DBSCAN(float *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, bool *collisionMatrix) {
  __shared__ int pointID;

  __shared__ int neighbors[MAX_SEEDS];

  __shared__ int neighborsCount;

  __shared__ int chainID;

  chainID = blockIdx.x;

  int currentSeedLength = seedLength[chainID];

  if (currentSeedLength == 0) return;

  pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];

  __shared__ bool processed;
  processed = false;

  if (threadIdx.x == 0) {
    seedLength[chainID] = currentSeedLength - 1;
    if (cluster[pointID] != UNPROCESSED) processed = true;
  }

  if (processed) return;

  __syncthreads();

  neighborsCount = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    register float comparingPoint[DIMENSION], point[DIMENSION];
    for (int x = 0; x < DIMENSION; x++) {
      comparingPoint[x] = dataset[i * DIMENSION + x];
    }

    for (int x = 0; x < DIMENSION; x++) {
      point[x] = dataset[pointID * DIMENSION + x];
    }

    register float distance = 0.0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }
    distance = sqrtf(distance);

    if (distance <= eps) {
      register int currentNeighborCount = atomicAdd(&neighborsCount, 1);
      if (neighborsCount >= minPts) {
        if(i != pointID) {
          MarkAsCandidate(i, chainID, cluster, seedList, seedLength,
                        refillSeedList, refillSeedLength, collisionMatrix);
        }
        
      } else {
        neighbors[currentNeighborCount] = i;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // printf("%d has neighbors count: %d %d %d\n", chainID, neighborsCount,
    // neighborsCount >= minPts, pointID);
  }

  if (neighborsCount >= minPts) {
    cluster[pointID] = chainID;
    for (int i = threadIdx.x; i < minPts; i = i + THREAD_COUNT) {
      MarkAsCandidate(neighbors[i], chainID, cluster, seedList, seedLength,
                      refillSeedList, refillSeedLength, collisionMatrix);
    }
  } else {
    cluster[pointID] = NOISE;
  }

  __syncthreads();
}

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *refillSeedList, int *refillSeedLength,
                                bool *collisionMatrix) {
  
  register int oldState = atomicCAS(&cluster[neighborID], UNPROCESSED, chainID);

  if (oldState == UNPROCESSED) {
    register int h = atomicAdd(&(seedLength[chainID]), 1);
    if (h < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + h] = neighborID;
    } else {
      register int k = atomicAdd(&(refillSeedLength[chainID]), 1);
      if (k < REFILL_MAX_SEEDS) {
        refillSeedList[chainID * REFILL_MAX_SEEDS + k] = neighborID;
      }
    }
  } else if (oldState != NOISE && oldState != chainID &&
             oldState < THREAD_BLOCKS) {
    if (oldState < chainID) {
      collisionMatrix[oldState * THREAD_BLOCKS + chainID] = true;
    } else {
      collisionMatrix[chainID * THREAD_BLOCKS + oldState] = true;
    }
  } else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  }

  
}

__global__ void showClusters(int *cluster) {
  int pointId = blockIdx.x * blockDim.x + threadIdx.x;
  if(pointId < DATASET_COUNT && cluster[pointId] < 1) {
    printf("Point %d: %d\n", pointId, cluster[pointId]);
  }
}

int ImportDataset(char const *fname, float *dataset) {
  FILE *fp = fopen(fname, "r");

  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }
  char buf[4096];
  int rowCnt = 0;
  int colCnt = 0;
  while (fgets(buf, 4096, fp) && rowCnt < DATASET_COUNT) {
    colCnt = 0;
    char *field = strtok(buf, ",");
    float tmp;
    sscanf(field, "%f", &tmp);
    dataset[rowCnt * DIMENSION + colCnt] = tmp;
    while (field) {
      colCnt++;
      field = strtok(NULL, ",");
      if (field != NULL) {
        float tmp;
        sscanf(field, "%f", &tmp);
        dataset[rowCnt * DIMENSION + colCnt] = tmp;
      }
    }
    rowCnt++;
  }
  fclose(fp);
  return 0;
}