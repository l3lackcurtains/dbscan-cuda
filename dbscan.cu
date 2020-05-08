#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <vector>

using namespace std;

// Number of data in dataset to use

#define DATASET_COUNT 100000
// #define DATASET_COUNT 1864620

// Dimension of the dataset
#define DIMENSION 2

// Maximum size of seed list
#define MAX_SEEDS 1024

// Maximum size of refill seed list
#define REFILL_MAX_SEEDS 2048

// Extra collission size to detect final clusters collision
#define EXTRA_COLLISION_SIZE 256

// Number of blocks
#define THREAD_BLOCKS 16

// Number of threads per block
#define THREAD_COUNT 1024

// Status of points that are not clusterized
#define UNPROCESSED -1

// Status for noise point
#define NOISE -2

// Minimum number of points in DBSCAN
#define MINPTS 4

// Epslion value in DBSCAN
#define EPS 1.5

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* GPU ERROR function checks for potential erros in cuda function execution
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
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

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Declare CPU and GPU Functions
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int ImportDataset(char const *fname, double *dataset);

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_extraCollisionLength);

void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, int *collisionMatrix,
                       int *extraCollision, int *extraCollisionLength);

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *refillSeedList, int *refillSeedLength,
                                int *collisionMatrix, int *extraCollision,
                                int *extraCollisionLength);

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Main CPU function
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
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

  // Get the dataset file name from argument
  strcpy(inputFname, argv[1]);
  printf("Using dataset file %s\n", inputFname);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  // Import data from dataset
  int ret = ImportDataset(inputFname, importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // Check if the data parsed is correct
  for (int i = 0; i < 2; i++) {
    printf("Sample Data %f\n", importedDataset[i]);
  }

  // Get the total count of dataset
  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }

  printf("Imported %lu data in dataset\n", unprocessedPoints.size());

  // Reset the GPU device for potential memory issues
  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  /**
   **************************************************************************
   * CUDA Memory allocation
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
  * Cuda and CPU performance metrics
  **************************************************************************
  */

  float gpuTime = 0, cpuTime = 0;

  // cuda time event
  cudaEvent_t gpuStart, gpuStop;
  gpuErrchk(cudaEventCreate(&gpuStart));
  gpuErrchk(cudaEventCreate(&gpuStop));

  clock_t totalTimeStart, totalTimeStop;
  float totalTime = 0.0;

  /**
   **************************************************************************
   * Start the DBSCAN algorithm
   **************************************************************************
   */

  // Keep track of number of cluster formed without global merge
  int runningCluster = 0;

  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  totalTimeStart = clock();

  while (!exit) {
    // Monitor the seed list and return the comptetion status of points
    int completed = MonitorSeedPoints(
        unprocessedPoints, &runningCluster, d_cluster, d_seedList, d_seedLength,
        d_refillSeedList, d_refillSeedLength, d_collisionMatrix,
        d_extraCollision, d_extraCollisionLength);

    printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
           unprocessedPoints.size());

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    gpuErrchk(cudaEventRecord(gpuStart, 0));

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_refillSeedList,
        d_refillSeedLength, d_collisionMatrix, d_extraCollision,
        d_extraCollisionLength);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(gpuStop, 0));
    cudaEventSynchronize(gpuStop);

    gpuErrchk(cudaEventSynchronize(gpuStop));

    float time = 0.0;
    gpuErrchk(cudaEventElapsedTime(&time, gpuStart, gpuStop));
    gpuTime += time / 1000;
  }

  /**
   **************************************************************************
   * End DBSCAN and show the results
   **************************************************************************
   */

  // Time measurement
  totalTimeStop = clock();
  totalTime = (totalTimeStop - totalTimeStart) / (float)1000;
  cpuTime = totalTime - gpuTime;

  printf("==============================================\n");
  printf("Overall Time: %3.2f seconds\n", totalTime);
  printf("GPU Only Time: %3.2f seconds\n", gpuTime);
  printf("CPU Only Time: %3.2f seconds\n", cpuTime);
  printf("==============================================\n");

  // Get the DBSCAN result
  GetDbscanResult(d_dataset, d_cluster, &runningCluster, &clusterCount,
                  &noiseCount);

  printf("==============================================\n");
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
  printf("==============================================\n");

  /**
   **************************************************************************
   * Free CUDA memory allocations
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
  cudaFree(gpuStart);
  cudaFree(gpuStop);
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Monitor Seed Points performs the following operations.
* 1) Check if the seed list is empty. If it is empty check the refill seed list
* else, return false to process next seed point by DBSCAN.
* 2) If seed list is empty, It will check refill seed list and fill the points
* from refill seed list to seed list
* 3) If seed list and refill seed list both are empty, then check for the
* collision matrix and form a cluster by merging chains.
* 4) After clusters are merged, new points are assigned to seed list
* 5) Lastly, It checks if all the points are processed. If so it will return
* true and DBSCAN algorithm will exit.
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_refillSeedList, int *d_refillSeedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_extraCollisionLength) {
  /**
   **************************************************************************
   * Copy GPU variables content to CPU variables for seed list management
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

  // Check if the seed list is empty
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    // If seed list is not empty set completeSeedListFirst as true
    if (localSeedLength[i] > 0) {
      completeSeedListFirst = true;
      break;
    }
    // If seed list is empty then, the seed list is loaded with data from
    // refill seed list, If refilling is done refill is set true
    else {
      if (localRefillSeedLength[i] > 0) {
        while (localSeedLength[i] < MAX_SEEDS && localRefillSeedLength[i] > 0) {
          localRefillSeedLength[i] = localRefillSeedLength[i] - 1;

          localSeedList[i * MAX_SEEDS + localSeedLength[i]] =
              localRefillSeedList[i * REFILL_MAX_SEEDS +
                                  localRefillSeedLength[i]];

          localSeedLength[i] = localSeedLength[i] + 1;
        }
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
   * Copy GPU variables to CPU variables for collision detection
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
   * If seedlist is empty and refill is also empty Then check the collision
   * between chains and finalize the clusters
   **************************************************************************
   */

  // Define cluster to map the collisions
  map<int, int> clusterMap;
  set<int> blockSet;

  // Insert chains in blockset
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    blockSet.insert(i);
  }

  set<int>::iterator it;

  // Iterate through the block set until it's empty
  while (blockSet.empty() == 0) {
    // Get a chain from blockset
    it = blockSet.begin();
    int curBlock = *it;

    // Expansion Queue is use to see expansion of collision
    set<int> expansionQueue;

    // Final Queue stores mapped chains for blockset chain
    set<int> finalQueue;

    // Insert current chain from blockset to expansion and final queue
    expansionQueue.insert(curBlock);
    finalQueue.insert(curBlock);

    // Iterate through expansion queue until it's empty
    while (expansionQueue.empty() == 0) {
      // Get first element from expansion queue
      it = expansionQueue.begin();
      int expandBlock = *it;

      // Remove the element because we are about to expand
      expansionQueue.erase(it);

      // Also erase from blockset, because we checked this chain
      blockSet.erase(expandBlock);

      // Loop through chains to see more collisions
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;

        // If there is collision, insert the chain in finalqueue
        // Also, insert in expansion queue for further checking
        // of collision with this chain
        if ((localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 ||
             localCollisionMatrix[x * THREAD_BLOCKS + expandBlock] == 1) &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    // Iterate through final queue, and map collided chains with blockset chain
    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      clusterMap[*it] = curBlock;
    }
  }

  // Loop through dataset and get points for mapped chain
  vector<vector<int>> clustersList(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (localCluster[i] >= 0 && localCluster[i] < THREAD_BLOCKS) {
      clustersList[clusterMap[localCluster[i]]].push_back(i);
    }
  }

  // Check extra collision with cluster ID greater than thread block
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    // If no extra collision found, continue
    if (localExtraCollisionLength[i] == 0) continue;

    // If the extra collision length is more than one then, assign all the
    // collided cluster points to the first cluster in the list
    // During this process, actual cluster are decreased which is not
    // maintained during the process.
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

    // Also, Assign the mapped chains to the first cluster in extra collision
    for (int x = 0; x < clustersList[clusterMap[i]].size(); x++) {
      localCluster[clustersList[clusterMap[i]][x]] =
          localExtraCollision[i * EXTRA_COLLISION_SIZE];
    }

    // Clear the mapped chains, as we assigned to clsuter already
    clustersList[clusterMap[i]].clear();
  }

  // From all the mapped chains, form a new cluster
  for (int i = 0; i < clustersList.size(); i++) {
    if (clustersList[i].size() == 0) continue;
    for (int x = 0; x < clustersList[i].size(); x++) {
      localCluster[clustersList[i][x]] = *runningCluster + THREAD_BLOCKS;
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

  /**
   **************************************************************************
   * Free CPU memory allocations
   **************************************************************************
   */
  free(localCluster);
  free(localSeedList);
  free(localRefillSeedList);
  free(localSeedLength);
  free(localRefillSeedLength);
  free(localCollisionMatrix);
  free(localExtraCollision);
  free(localExtraCollisionLength);

  // If all points has been processed exit DBSCAN
  if (unprocessedPoints.empty()) return true;

  return false;
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Get DBSCAN result
* Get the final cluster and print the overall result
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount) {
  /**
  **************************************************************************
  * Print the cluster and noise results
  **************************************************************************
  */

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  double *dataset;
  dataset = (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);
  gpuErrchk(cudaMemcpy(dataset, d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyDeviceToHost));

  map<int, int> finalClusterMap;
  int localClusterCount = 0;
  int localNoiseCount = 0;
  for (int i = THREAD_BLOCKS; i <= (*runningCluster) + THREAD_BLOCKS; i++) {
    bool found = false;
    for (int j = 0; j < DATASET_COUNT; j++) {
      if (localCluster[j] == i) {
        found = true;
        break;
      }
    }
    if (found) {
      ++localClusterCount;
      finalClusterMap[i] = localClusterCount;
    }
  }
  for (int j = 0; j < DATASET_COUNT; j++) {
    if (localCluster[j] == NOISE) {
      localNoiseCount++;
    }
  }
  *clusterCount = localClusterCount;
  *noiseCount = localNoiseCount;

  // Output to file
  ofstream outputFile;
  outputFile.open("gpu_dbscan_output.txt");

  for (int j = 0; j < DATASET_COUNT; j++) {
    if (finalClusterMap[localCluster[j]] > 0) {
      localCluster[j] = finalClusterMap[localCluster[j]];
    } else {
      localCluster[j] = -1;
    }
  }

  for (int j = 0; j < DATASET_COUNT; j++) {
    outputFile << localCluster[j] << ", " << dataset[j * DIMENSION] << ", "
               << dataset[j * DIMENSION + 1] << endl;
  }

  outputFile.close();

  free(localCluster);
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* DBSCAN: Main kernel function of the algorithm
* It does the following functions.
* 1) Every block gets a point from seedlist to expand. If these points are
* processed already, it returns
* 2) It expands the points by finding neighbors points
* 3) Checks for the collision and mark the collision in collision matrix
//////////////////////////////////////////////////////////////////////////
*/
__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *refillSeedList,
                       int *refillSeedLength, int *collisionMatrix,
                       int *extraCollision, int *extraCollisionLength) {
  /**
   **************************************************************************
   * Define shared variables
   **************************************************************************
   */

  // Point ID to expand by a block
  __shared__ int pointID;

  // Neighbors to store of neighbors points exceeds minpoints
  __shared__ int neighborBuffer[MINPTS];

  // It counts the total neighbors
  __shared__ int neighborCount;

  // ChainID is basically blockID
  __shared__ int chainID;

  // Store the point from pointID
  __shared__ int point[DIMENSION];

  // Length of the seedlist to check its size
  __shared__ int currentSeedLength;

  /**
   **************************************************************************
   * Get current chain length, and If its zero, exit
   **************************************************************************
   */

  // Assign chainID, current seed length and pointID
  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
    currentSeedLength = seedLength[chainID];
    pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
  }
  __syncthreads();

  // If seed length is 0, return
  if (currentSeedLength == 0) return;

  // Check if the point is already processed
  __shared__ int processed;
  if (threadIdx.x == 0) {
    processed = false;
    seedLength[chainID] = currentSeedLength - 1;
    if (cluster[pointID] != UNPROCESSED) processed = true;
  }
  __syncthreads();

  // If the point is already processed, return
  if (processed) return;

  // Assign neighorCount to 0 and assign point data
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

    // find the distance between the points
    register double distance = 0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }

    // If distance is less than elipson, mark point as candidate
    if (distance <= EPS * EPS) {
      register int currentNeighborCount = atomicAdd(&neighborCount, 1);
      if (currentNeighborCount >= MINPTS) {
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
   * If neighbors are less than MINPTS, assign pointID with noise
   **************************************************************************
   */

  if (neighborCount >= MINPTS) {
    cluster[pointID] = chainID;
    for (int i = threadIdx.x; i < MINPTS; i = i + THREAD_COUNT) {
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

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Mark as candidate
* It does the following functions:
* 1) Mark the neighbor's cluster with chainID if its old state is unprocessed
* 2) If the oldstate is unprocessed, insert the neighnor point to seed list
* 3) if the seed list exceeds max value, insert into refill seed list
* 4) If the old state is less than THREAD BLOCK, record the collision in
* collision matrix
* 5) If the old state is greater than THREAD BLOCK, record the collision
* in extra collision
//////////////////////////////////////////////////////////////////////////
*/

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
           oldState < THREAD_BLOCKS) {
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
  }
  /**
   **************************************************************************
   * If the old state is greater than thread block, record the extra collisions
   **************************************************************************
   */

  else if (oldState >= THREAD_BLOCKS) {
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
//////////////////////////////////////////////////////////////////////////
* Import Dataset
* It imports the data from the file and store in dataset variable
//////////////////////////////////////////////////////////////////////////
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