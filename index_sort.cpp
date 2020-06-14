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

#define DATASET_SIZE 10000

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
  dataset[10] = 132.42;
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
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%f ", dataset[i * 2 + j]);
    }
    printf("\n");
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (dataset[i * 2] > dataset[j * 2]) {
        for (int x = 0; x < 2; x++) {
          double temp = dataset[i * 2 + x];
          dataset[i * 2 + x] = dataset[j * 2 + x];
          dataset[j * 2 + x] = temp;
        }
      }
    }
  }
  // Sorting in 2 dimension.
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      if (dataset[i * 2] == dataset[j * 2] &&
          dataset[i * 2 + 1] > dataset[j * 2 + 1]) {
        for (int x = 0; x < 2; x++) {
          double temp = dataset[i * 2 + x];
          dataset[i * 2 + x] = dataset[j * 2 + x];
          dataset[j * 2 + x] = temp;
        }
      }
    }
  }

  printf("After sorting dataset\n");
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%f ", dataset[i * 2 + j]);
    }
    printf("\n");
  }

  return 0;
}
