#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

using namespace std;

// #define DATASET_COUNT 100000
#define DATASET_COUNT 1864620
#define DIMENSION 2
#define MAX_SEEDS 1024
#define REFILL_MAX_SEEDS 2048
#define EXTRA_COLLISION_SIZE 256
#define THREAD_BLOCKS 16
#define THREAD_COUNT 1024
#define UNPROCESSED -1
#define NOISE -2
#define MINPTS 4
#define EPS 1.5

__device__ __constant__ int minPts = MINPTS;
__device__ __constant__ float eps = EPS;

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

int ImportDataset(char const *fname, double *dataset);
bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *clusterCount, int *noiseCount, int *d_cluster,
                       int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_extraCollisionLength);
__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, int *collisionMatrix,
                       int *extraCollision, int *extraCollisionLength);
__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *refillSeedList, int *refillSeedLength,
                                int *collisionMatrix, int *extraCollision,
                                int *extraCollisionLength);
int main(int argc, char **argv) {
  /**
   **************************************************************************
   * Get the dataset file from argument and import data
   **************************************************************************
   */

  char inputFname[500];
  if (argc != 2) {
    fprintf(stderr, "Please provide the dataset file path in the arguments\n");
    exit(0);
  }

  strcpy(inputFname, argv[1]);

  printf("Using dataset file %s\n", inputFname);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  int ret = ImportDataset(inputFname, importedDataset);

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
  printf("Imported %lu data in dataset\n", unprocessedPoints.size());

  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  /**
   **************************************************************************
   * Memory allocation
   **************************************************************************
   */
  double *d_dataset;
  int *d_cluster;
  int *d_seedList;
  int *d_seedLength;
  int *d_refillSeedList;
  int *d_refillSeedLength;
  int *d_collisionMatrix;
  int *d_extraCollision;
  int *d_extraCollisionLength;

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));
  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_refillSeedList,
                       sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS));
  gpuErrchk(
      cudaMalloc((void **)&d_refillSeedLength, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));
  gpuErrchk(cudaMalloc((void **)&d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));
  gpuErrchk(cudaMalloc((void **)&d_extraCollisionLength,
                       sizeof(int) * THREAD_BLOCKS));

  /**
   **************************************************************************
   * Assignment with default values
   **************************************************************************
   */
  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));
  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));
  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_refillSeedList, -1,
                       sizeof(int) * THREAD_BLOCKS * REFILL_MAX_SEEDS));
  gpuErrchk(cudaMemset(d_refillSeedLength, 0, sizeof(int) * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));
  gpuErrchk(cudaMemset(d_extraCollisionLength, 0, sizeof(int) * THREAD_BLOCKS));

  /**
   **************************************************************************
   * Start the DBSCAN algorithm
   **************************************************************************
   */

  int runningCluster = 0;
  int clusterCount = 0;
  int noiseCount = 0;
  bool exit = false;
  while (!exit) {
    int completed = MonitorSeedPoints(
        unprocessedPoints, &runningCluster, &clusterCount, &noiseCount,
        d_cluster, d_seedList, d_seedLength, d_refillSeedList,
        d_refillSeedLength, d_collisionMatrix, d_extraCollision,
        d_extraCollisionLength);

    if (completed) {
      exit = true;
    }

    printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
           unprocessedPoints.size());

    if (exit) break;

    gpuErrchk(cudaDeviceSynchronize());

    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_refillSeedList,
        d_refillSeedLength, d_collisionMatrix, d_extraCollision,
        d_extraCollisionLength);

    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   **************************************************************************
   * End DBSCAN and show the results
   **************************************************************************
   */

  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);

  /**
   **************************************************************************
   * Free memory allocations
   **************************************************************************
   */
  cudaFree(d_dataset);
  cudaFree(d_cluster);
  cudaFree(d_seedList);
  cudaFree(d_seedLength);
  cudaFree(d_refillSeedList);
  cudaFree(d_refillSeedLength);
  cudaFree(d_collisionMatrix);
  cudaFree(d_extraCollision);
  cudaFree(d_extraCollisionLength);
}

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *clusterCount, int *noiseCount, int *d_cluster,
                       int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_extraCollisionLength) {
  /**
   **************************************************************************
   * Define and copy GPU variables to CPU variables
   **************************************************************************
   */
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

  /**
   **************************************************************************
   * Check if the seedlist is not empty, If so continue with DBSCAN process
   * if seedlist is empty, check refill seed list
   * if there are points in refill list, transfer to seedlist
   **************************************************************************
   */

  int completeSeedListFirst = false;
  int refilled = false;
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

  /**
   **************************************************************************
   * If seedlist still have points, go to DBSCAN process
   **************************************************************************
   */

  if (completeSeedListFirst) {
    free(localSeedList);
    free(localSeedLength);
    free(localRefillSeedList);
    free(localRefillSeedLength);
    return false;
  }

  /**
   **************************************************************************
   * If refill has been done, reload the seedlist and seedlist to GPU
   * and return to DBSCAN process
   **************************************************************************
   */
  if (refilled) {
    gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                         sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_refillSeedLength, localRefillSeedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
    free(localSeedList);
    free(localSeedLength);
    free(localRefillSeedList);
    free(localRefillSeedLength);
    return false;
  }

  /**
   **************************************************************************
   * Define and copy GPU variables to CPU variables
   **************************************************************************
   */

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);

  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  int *localCollisionMatrix;
  localCollisionMatrix =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS);

  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost));

  int *localExtraCollision;
  localExtraCollision =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE);

  gpuErrchk(cudaMemcpy(localExtraCollision, d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE,
                       cudaMemcpyDeviceToHost));
  int *localExtraCollisionLength;
  localExtraCollisionLength =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE);

  gpuErrchk(cudaMemcpy(localExtraCollisionLength, d_extraCollisionLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  /**
   **************************************************************************
   * If seedlist is empty and refill is also empty
   * Then check the collision between chains
   * and finalize the clusters
   **************************************************************************
   */

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
        if ((localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 ||
             localCollisionMatrix[x * THREAD_BLOCKS + expandBlock] == 1) &&
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

  vector<vector<int>> clustersList(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (localCluster[i] >= 0 && localCluster[i] < THREAD_BLOCKS) {
      clustersList[clusterMap[localCluster[i]]].push_back(i);
    }
  }

  // Check extra collision
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localExtraCollisionLength[i] == 0) continue;
    if (localExtraCollisionLength[i] > 1) {
      for (int j = 1; j < localExtraCollisionLength[i]; j++) {
        for (int k = 0; k < DATASET_COUNT; k++) {
          if (localCluster[k] ==
              localExtraCollision[i * EXTRA_COLLISION_SIZE + j]) {
            localCluster[k] = localExtraCollision[i * EXTRA_COLLISION_SIZE];
          }
        }
      }
    }
    for (int x = 0; x < clustersList[clusterMap[i]].size(); x++) {
      localCluster[clustersList[clusterMap[i]][x]] =
          localExtraCollision[i * EXTRA_COLLISION_SIZE];
    }
    clustersList[clusterMap[i]].clear();
  }

  for (int i = 0; i < clustersList.size(); i++) {
    if (clustersList[i].size() == 0) continue;
    for (int x = 0; x < clustersList[i].size(); x++) {
      localCluster[clustersList[i][x]] = *runningCluster + THREAD_BLOCKS + 1;
    }
    (*runningCluster)++;
  }

  /**
   **************************************************************************
   * After finilazing the cluster, check the remaining points and
   * insert one point to each of the seedlist
   **************************************************************************
   */

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

  /**
  **************************************************************************
  * FInally, transfer back the CPU memory to GPU and run DBSCAN process
  **************************************************************************
  */

  gpuErrchk(cudaMemcpy(d_cluster, localCluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));
  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));
  gpuErrchk(cudaMemset(d_extraCollisionLength, 0, sizeof(int) * THREAD_BLOCKS));

  if (unprocessedPoints.empty()) {
    int localClusterCount = 0;
    int localNoiseCount = 0;
    for (int i = THREAD_BLOCKS + 1; i < (*runningCluster) + THREAD_BLOCKS + 1;
         i++) {
      bool found = false;
      for (int j = 0; j < DATASET_COUNT; j++) {
        if (localCluster[j] == i) {
          found = true;
          break;
        }
      }
      if (found) {
        localClusterCount++;
      }
    }
    for (int j = 0; j < DATASET_COUNT; j++) {
      if (localCluster[j] == NOISE) {
        localNoiseCount++;
      }
    }
    *clusterCount = localClusterCount;
    *noiseCount = localNoiseCount;

  }

  free(localCluster);
  free(localSeedList);
  free(localRefillSeedList);
  free(localSeedLength);
  free(localRefillSeedLength);
  free(localCollisionMatrix);
  free(localExtraCollision);
  free(localExtraCollisionLength);

  // IF all points has been processed exit DBSCAN
  if (unprocessedPoints.empty()) return true;

  return false;
}

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, int *collisionMatrix,
                       int *extraCollision, int *extraCollisionLength) {
  /**
   **************************************************************************
   * Define shared variables
   **************************************************************************
   */

  __shared__ int pointID;
  __shared__ int neighborBuffer[MINPTS];
  __shared__ int neighborCount;
  __shared__ int chainID;
  __shared__ int point[DIMENSION];
  __shared__ int currentSeedLength;

  /**
   **************************************************************************
   * Get current chain length, and If its zero, exit
   **************************************************************************
   */

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
    currentSeedLength = seedLength[chainID];
    pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
  }
  __syncthreads();
  if (currentSeedLength == 0) return;

  __shared__ int processed;
  if (threadIdx.x == 0) {
    processed = false;
    seedLength[chainID] = currentSeedLength - 1;
    if (cluster[pointID] != UNPROCESSED) processed = true;
  }

  __syncthreads();

  if (processed) return;

  if (threadIdx.x == 0) {
    neighborCount = 0;
    for (int x = 0; x < DIMENSION; x++) {
      point[x] = dataset[pointID * DIMENSION + x];
    }
  }
  __syncthreads();

  /**
   **************************************************************************
   * Find the neighbors of the pointID
   * Mark point as candidate if points are more than min points
   * Keep record of left over neighbors in neighborBuffer
   **************************************************************************
   */
  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    register double comparingPoint[DIMENSION];
    for (int x = 0; x < DIMENSION; x++) {
      comparingPoint[x] = dataset[i * DIMENSION + x];
    }

    register double distance = 0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }

    if (distance <= eps * eps) {
      register int currentNeighborCount = atomicAdd(&neighborCount, 1);
      if (currentNeighborCount >= minPts) {
        MarkAsCandidate(i, chainID, cluster, seedList, seedLength,
                        refillSeedList, refillSeedLength, collisionMatrix,
                        extraCollision, extraCollisionLength);
      } else {
        neighborBuffer[currentNeighborCount] = i;
      }
    }
  }
  __syncthreads();

  /**
   **************************************************************************
   * Mark the left over neighbors in neighborBuffer as cluster member
   * If neighbors are less than minPts, assign pointID with noise
   **************************************************************************
   */

  if (neighborCount >= minPts) {
    cluster[pointID] = chainID;
    for (int i = threadIdx.x; i < minPts; i = i + THREAD_COUNT) {
      MarkAsCandidate(neighborBuffer[i], chainID, cluster, seedList, seedLength,
                      refillSeedList, refillSeedLength, collisionMatrix,
                      extraCollision, extraCollisionLength);
    }
  } else {
    cluster[pointID] = NOISE;
  }

  __syncthreads();

  /**
   **************************************************************************
   * Check Thread length, If it exceeds MAX limit the length
   * As seedlist wont have data beyond its max length
   **************************************************************************
   */

  if (threadIdx.x == 0 && seedLength[chainID] >= MAX_SEEDS) {
    seedLength[chainID] = MAX_SEEDS - 1;
  }

  __syncthreads();

  if (threadIdx.x == 0 && refillSeedLength[chainID] >= REFILL_MAX_SEEDS) {
    refillSeedLength[chainID] = REFILL_MAX_SEEDS - 1;
  }
  __syncthreads();
}

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *refillSeedList, int *refillSeedLength,
                                int *collisionMatrix, int *extraCollision,
                                int *extraCollisionLength) {
  /**
  **************************************************************************
  * Get the old cluster state of the neighbor
  * If the state is unprocessed, assign it with chainID
  **************************************************************************
  */
  register int oldState = atomicCAS(&cluster[neighborID], UNPROCESSED, chainID);

  /**
   **************************************************************************
   * For unprocessed old state of neighbors, add them to seedlist and
   * refill seedlist
   **************************************************************************
   */
  if (oldState == UNPROCESSED) {
    register int sl = atomicAdd(&(seedLength[chainID]), 1);
    if (sl < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + sl] = neighborID;
    } else {
      register int rsl = atomicAdd(&(refillSeedLength[chainID]), 1);
      if (rsl < REFILL_MAX_SEEDS) {
        refillSeedList[chainID * REFILL_MAX_SEEDS + rsl] = neighborID;
      }
    }
  }

  /**
   **************************************************************************
   * If the old state of neighbor is not noise, not member of chain and cluster
   * is within THREADBLOCK, maek the collision between old and new state
   **************************************************************************
   */
  else if (oldState != NOISE && oldState != chainID &&
           oldState < THREAD_BLOCKS &&
           collisionMatrix[oldState * THREAD_BLOCKS + chainID] == -1 &&
           collisionMatrix[oldState * THREAD_BLOCKS + chainID] == -1) {
    if (chainID > oldState) {
      collisionMatrix[oldState * THREAD_BLOCKS + chainID] = 1;
    } else {
      collisionMatrix[chainID * THREAD_BLOCKS + oldState] = 1;
    }
  }

  /**
   **************************************************************************
   * If the old state is noise, assign it to chainID cluster
   **************************************************************************
   */
  else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  } else if (oldState >= THREAD_BLOCKS) {
    register int cl = atomicAdd(&(extraCollisionLength[chainID]), 1);
    extraCollision[chainID * EXTRA_COLLISION_SIZE + cl] = oldState;

    bool alreadyThere = false;
    for (int k = 0; k < cl; k++) {
      if (oldState == extraCollision[chainID * EXTRA_COLLISION_SIZE + k]) {
        alreadyThere = true;
        break;
      }
    }
    if (alreadyThere) {
      cl = atomicSub(&(extraCollisionLength[chainID]), 1);
    }
  }
}

/**
 **************************************************************************
 * Import dataset from file.
 **************************************************************************
 */
int ImportDataset(char const *fname, double *dataset) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
    char *field = strtok(buf, ",");
    long double tmp;
    sscanf(field, "%Lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        long double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}