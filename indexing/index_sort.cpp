#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#define DATASET_COUNT DATASET_COUNT00
#define DIMENSION 2

using namespace std;

void DetermineErrorTwoClusterResults(vector<int>* c1, vector<int>* c2);

int importDataset(char const* fname, int N, int* cluster);

int main() {
  double* dataset = (double*)malloc(sizeof(double) * 20);

  dataset[0] = 144.92;
  dataset[1] = 32.55;
  dataset[2] = 124.12;
  dataset[3] = 33.21;
  dataset[4] = 144.92;
  dataset[5] = 34.4;
  dataset[6] = 123.16;
  dataset[7] = 31.21;
  dataset[8] = 145.14;
  dataset[9] = 45.42;
  dataset[DATASET_COUNT] = 132.42;
  dataset[11] = 11.21;
  dataset[12] = 111.12;
  dataset[13] = 45.4;
  dataset[14] = 200.12;
  dataset[15] = 145.21;
  dataset[16] = 111.12;
  dataset[17] = 44.24;
  dataset[18] = 144.92;
  dataset[19] = 35.21;

  printf("Original dataset\n");
  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      printf("%f ", dataset[i * DIMENSION + j]);
    }
    printf("\n");
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DATASET_COUNT; j++) {
      if (dataset[i * DIMENSION] > dataset[j * DIMENSION]) {
        for (int x = 0; x < DIMENSION; x++) {
          double temp = dataset[i * DIMENSION + x];
          dataset[i * DIMENSION + x] = dataset[j * DIMENSION + x];
          dataset[j * DIMENSION + x] = temp;
        }
      }
    }
  }
  // Sorting in DIMENSION dimension.
  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DATASET_COUNT; j++) {
      if (dataset[i * DIMENSION] == dataset[j * DIMENSION] &&
          dataset[i * DIMENSION + 1] > dataset[j * DIMENSION + 1]) {
        for (int x = 0; x < DIMENSION; x++) {
          double temp = dataset[i * DIMENSION + x];
          dataset[i * DIMENSION + x] = dataset[j * DIMENSION + x];
          dataset[j * DIMENSION + x] = temp;
        }
      }
    }
  }

  printf("After sorting dataset\n");
  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%f ", dataset[i * 2 + j]);
    }
    printf("\n");
  }

  return 0;
}
