#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#define DATASET_COUNT 20000
#define DIMENSION 2
#define MAX_SEEDS 1024
#define REFILL_MAX_SEEDS 3000

#define THREAD_BLOCKS 32
#define THREAD_COUNT 128

#define UNPROCESSED -1
#define NOISE -2

using namespace std;

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

__device__ __constant__ int minPts = 4;
__device__ __constant__ double eps = 1.5;

__device__ double dataset[DATASET_COUNT][DIMENSION];
__device__ int cluster[DATASET_COUNT];
__device__ int seedList[THREAD_BLOCKS][MAX_SEEDS];
__device__ int currentSeedLength[THREAD_BLOCKS];
__device__ long long int refillSeedList[THREAD_BLOCKS][REFILL_MAX_SEEDS];
__device__ int refillCurrentSeedLength[THREAD_BLOCKS];
__device__ bool collisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS];

bool **d_collisionMatrix;
int **d_seedList;
int *d_currentSeedLength;
long long int **d_refillSeedList;
int *d_refillCurrentSeedLength;
double **d_dataset;
int *d_cluster;

int importDataset(char const *fname, double **dataset);
bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *clusterCount);
void FinalizeClusters(int *cluster, int *clusterCount);
__global__ void DBSCAN();
__global__ void showClusters();
__device__ void markAsCandidate(int neighborID, int chainID);

int main() {
  double **importedDataset =
      (double **)malloc(sizeof(double *) * DATASET_COUNT);
  for (int i = 0; i < DATASET_COUNT; i++) {
    importedDataset[i] = (double *)malloc(sizeof(double) * DIMENSION);
  }
  int ret = importDataset("./dataset/dataset.txt", importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }
  printf("Imported %d data in dataset\n", unprocessedPoints.size());

  gpuErrchk(cudaFree(0));

  gpuErrchk(cudaGetSymbolAddress((void **)&d_collisionMatrix, collisionMatrix));
  gpuErrchk(cudaGetSymbolAddress((void **)&d_seedList, seedList));
  gpuErrchk(
      cudaGetSymbolAddress((void **)&d_currentSeedLength, currentSeedLength));
  gpuErrchk(cudaGetSymbolAddress((void **)&d_refillSeedList, refillSeedList));
  gpuErrchk(cudaGetSymbolAddress((void **)&d_refillCurrentSeedLength,
                                 refillCurrentSeedLength));
  gpuErrchk(cudaGetSymbolAddress((void **)&d_dataset, dataset));
  gpuErrchk(cudaGetSymbolAddress((void **)&d_cluster, cluster));

  gpuErrchk(cudaMemset(d_collisionMatrix, false,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMemset(d_currentSeedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(
      cudaMemset(d_refillSeedList, -1,
                 sizeof(long long int) * THREAD_BLOCKS * REFILL_MAX_SEEDS));

  gpuErrchk(
      cudaMemset(d_refillCurrentSeedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));

  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * 2,
                       cudaMemcpyHostToDevice));

  int clusterCount = 0;

  bool exit = false;
  while (!exit) {
    bool completed = MonitorSeedPoints(unprocessedPoints, &clusterCount);

    if (completed) {
      exit = true;
    }

    printf("Number of cluster %d, unprocessed points: %d\n", clusterCount,
           unprocessedPoints.size());

    if (exit) break;

    gpuErrchk(cudaDeviceSynchronize());

    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>();

    gpuErrchk(cudaDeviceSynchronize());
  }

  gpuErrchk(cudaDeviceSynchronize());
  showClusters<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>();

  printf("Final number of cluster %d\n", clusterCount);
}

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *clusterCount) {
  int localSeedLength[THREAD_BLOCKS];
  gpuErrchk(cudaMemcpy(localSeedLength, d_currentSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int localRefillSeedLength[THREAD_BLOCKS];
  gpuErrchk(cudaMemcpy(localRefillSeedLength, d_refillCurrentSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int localSeedList[THREAD_BLOCKS][MAX_SEEDS];
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  long long int localRefillSeedList[THREAD_BLOCKS][REFILL_MAX_SEEDS];
  gpuErrchk(cudaMemcpy(localRefillSeedList, d_refillSeedList,
                       sizeof(long long int) * THREAD_BLOCKS * REFILL_MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

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

          localSeedList[i][localSeedLength[i]] =
              localRefillSeedList[i][localRefillSeedLength[i]];

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
    gpuErrchk(cudaMemcpy(d_currentSeedLength, localSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                         sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_refillCurrentSeedLength, localRefillSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    return false;
  }

  if (completeSeedListFirst){
    return false;
  }
  int localCluster[DATASET_COUNT];
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  FinalizeClusters(localCluster, clusterCount);

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    while (!unprocessedPoints.empty()) {
      int lastPoint = unprocessedPoints.back();
      unprocessedPoints.pop_back();
      if (localCluster[lastPoint] == UNPROCESSED) {
        localSeedLength[i] = 1;
        localSeedList[i][0] = lastPoint;
        break;
      }
    }
  }

  gpuErrchk(cudaMemcpy(d_cluster, localCluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_currentSeedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  if (unprocessedPoints.empty()) {
    // for(int i = 0; i < DATASET_COUNT; i++) {
    //   printf("Point %d, cluster %d\n", i, localCluster[i]);
    // }
  }
  if (unprocessedPoints.empty()) return true;

  return false;
}

void FinalizeClusters(int *cluster, int *clusterCount) {
  bool localCollisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS];

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
        if ((localCollisionMatrix[expandBlock][x] ||
             localCollisionMatrix[x][expandBlock]) &&
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

  vector<vector<int>> clusters(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (cluster[i] >= 0 && cluster[i] < THREAD_BLOCKS) {
      clusters[clusterMap[cluster[i]]].push_back(i);
    }
  }

  for (int i = 0; i < clusters.size(); i++) {
    if (clusters[i].size() == 0) continue;
    for (int x = 0; x < clusters[i].size(); x++) {
      cluster[clusters[i][x]] = *clusterCount + THREAD_BLOCKS + 1;
    }
    (*clusterCount)++;
  }

  gpuErrchk(cudaMemset(d_collisionMatrix, false,
                       sizeof(bool) * THREAD_BLOCKS * THREAD_BLOCKS));
}

__global__ void DBSCAN() {
  __shared__ double point[DIMENSION];

  __shared__ int pointID;

  __shared__ int neighbors[MAX_SEEDS];

  __shared__ int neighborsCount;

  __shared__ int chainID;

  chainID = blockIdx.x;

  int seedLength = currentSeedLength[chainID];

  if (seedLength == 0) return;

  pointID = seedList[chainID][seedLength - 1];

  __shared__ bool processed;
  processed = false;

  if (threadIdx.x == 0) {
    currentSeedLength[chainID] = seedLength - 1;
    if (cluster[pointID] != UNPROCESSED) processed = true;
  }

  if (processed) return;

  __syncthreads();

  neighborsCount = 0;

  for (int x = 0; x < DIMENSION; x++) {
    point[x] = dataset[pointID][x];
  }

  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    register double comparingPoint[DIMENSION];
    for (int x = 0; x < DIMENSION; x++) {
      comparingPoint[x] = dataset[i][x];
    }

    register double distance = 0.0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }
    distance = sqrtf(distance);

    if (distance <= eps) {
      register int currentNeighborCount = atomicAdd(&neighborsCount, 1);
      if (neighborsCount >= minPts) {
        markAsCandidate(i, chainID);
      } else {
        neighbors[currentNeighborCount] = i;
      }
    }
  }

  __syncthreads();

  if (neighborsCount >= minPts) {
    cluster[pointID] = chainID;
    for (int i = threadIdx.x; i < minPts; i = i + THREAD_COUNT) {
      markAsCandidate(neighbors[i], chainID);
    }

  } else {
    cluster[pointID] = NOISE;
  }

  __syncthreads();

  if (threadIdx.x == 0 && currentSeedLength[chainID] >= MAX_SEEDS) {
    currentSeedLength[chainID] = MAX_SEEDS - 1;
  }

  if (threadIdx.x == 0 &&
      refillCurrentSeedLength[chainID] >= REFILL_MAX_SEEDS) {
    currentSeedLength[chainID] = REFILL_MAX_SEEDS - 1;
  }
}

__device__ void markAsCandidate(int neighborID, int chainID) {
  register int oldState = atomicCAS(&cluster[neighborID], UNPROCESSED, chainID);
  if (oldState == UNPROCESSED) {
    register int h = atomicAdd(&(currentSeedLength[chainID]), 1);
    if (h < MAX_SEEDS) {
      seedList[chainID][h] = neighborID;
    } else {
      register int k = atomicAdd(&(refillCurrentSeedLength[chainID]), 1);
      if (k < REFILL_MAX_SEEDS) {
        refillSeedList[chainID][k] = neighborID;
      }
    }
  } else if (oldState != NOISE && oldState != chainID &&
             oldState < THREAD_BLOCKS) {
    if (oldState < chainID) {
      collisionMatrix[oldState][chainID] = true;
    } else {
      collisionMatrix[chainID][oldState] = true;
    }
  } else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  }
}

__global__ void showClusters() {
  int pointId = blockIdx.x * blockDim.x + threadIdx.x;
  // if(pointId < DATASET_COUNT) {
  //   printf("Point %d: %d\n", pointId, cluster[pointId]);
  // }
}

int importDataset(char const *fname, double **dataset) {
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
    double tmp;
    sscanf(field, "%f", &tmp);
    dataset[rowCnt][colCnt] = tmp;
    while (field) {
      colCnt++;
      field = strtok(NULL, ",");
      if (field != NULL) {
        double tmp;
        sscanf(field, "%f", &tmp);
        dataset[rowCnt][colCnt] = tmp;
      }
    }
    rowCnt++;
  }
  fclose(fp);
  return 0;
}