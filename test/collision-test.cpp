#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>
using namespace std;

#define THREAD_BLOCKS 8

int main() {
  int *localCollisionMatrix;
  localCollisionMatrix =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS);

  int *localCollisionMatrixLength;
  localCollisionMatrixLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);

  for (int i = 0; i < THREAD_BLOCKS * THREAD_BLOCKS; i++) {
    localCollisionMatrix[i] = -1;
  }

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    localCollisionMatrixLength[i] = 0;
  }

  localCollisionMatrixLength[0] = 2;
  localCollisionMatrix[0 * THREAD_BLOCKS + 0] = 1;
  localCollisionMatrix[0 * THREAD_BLOCKS + 1] = 2;

  localCollisionMatrixLength[1] = 2;
  localCollisionMatrix[1 * THREAD_BLOCKS + 0] = 3;
  localCollisionMatrix[1 * THREAD_BLOCKS + 1] = 2;

  localCollisionMatrixLength[2] = 1;
  localCollisionMatrix[2 * THREAD_BLOCKS + 0] = 2;

  localCollisionMatrixLength[4] = 2;
  localCollisionMatrix[4 * THREAD_BLOCKS + 0] = 5;
  localCollisionMatrix[4 * THREAD_BLOCKS + 1] = 6;

  localCollisionMatrixLength[5] = 1;
  localCollisionMatrix[5 * THREAD_BLOCKS + 0] = 7;

  localCollisionMatrixLength[6] = 1;
  localCollisionMatrix[6 * THREAD_BLOCKS + 0] = 7;

  localCollisionMatrixLength[7] = 1;
  localCollisionMatrix[7 * THREAD_BLOCKS + 0] = 7;

  map<int, int> clusterMap;
  set<int> blockSet;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    clusterMap[i] = i;
    blockSet.insert(i);
  }
  set<int>::iterator it;
  while (!blockSet.empty()) {
    it = blockSet.begin();
    int curBlock = *it;

    set<int> expansionQueue;
    set<int> finalQueue;

    expansionQueue.insert(curBlock);
    finalQueue.insert(curBlock);

    while (!expansionQueue.empty()) {
      it = expansionQueue.begin();
      int expandBlock = *it;

      expansionQueue.erase(it);
      blockSet.erase(expandBlock);

      for (int y = 0; y < localCollisionMatrixLength[expandBlock]; y++) {
        int x = localCollisionMatrix[expandBlock * THREAD_BLOCKS + y];

        if(x == expandBlock) continue;

        if (blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      clusterMap[*it] = curBlock;
    }
  }

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    printf("%d maps to %d \n", i, clusterMap[i]);
  }

  return 0;
}