#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define DATASET_COUNT 10
#define DIMENSION 2
#define THREAD_BLOCKS 1
#define THREAD_COUNT 1

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

struct TupleComp
{
	__host__ __device__ bool operator()(const thrust::tuple<double, double>& t1, const thrust::tuple<double, double>& t2)
	{
		if (t1.get<0>() < t2.get<0>())
			return true;
		if (t1.get<0>() > t2.get<0>())
			return false;
		return t1.get<1>() < t2.get<1>();
	}
};

int main(int argc, char **argv) {
  thrust::host_vector<double> h_dataset(DATASET_COUNT * DIMENSION);

  h_dataset[0] = 144.92;
  h_dataset[1] = 32.55;
  h_dataset[2] = 124.12;
  h_dataset[3] = 33.21;
  h_dataset[4] = 144.92;
  h_dataset[5] = 34.4;
  h_dataset[6] = 123.16;
  h_dataset[7] = 31.21;
  h_dataset[8] = 145.14;
  h_dataset[9] = 45.42;
  h_dataset[10] = 132.42;
  h_dataset[11] = 11.21;
  h_dataset[12] = 111.12;
  h_dataset[13] = 45.4;
  h_dataset[14] = 200.12;
  h_dataset[15] = 145.21;
  h_dataset[16] = 111.12;
  h_dataset[17] = 44.24;
  h_dataset[18] = 144.92;
  h_dataset[19] = 35.21;

  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  double dataset_tuple_x[DATASET_COUNT];
  double dataset_tuple_y[DATASET_COUNT];
  for(int i = 0; i < DATASET_COUNT; i++) {
    dataset_tuple_x[i] = h_dataset[i*DIMENSION];
    dataset_tuple_y[i] = h_dataset[i*DIMENSION + 1];
  }

  double *d_dataset_tuple1;
  gpuErrchk(cudaMalloc(&d_dataset_tuple1, DATASET_COUNT * sizeof(double)));
  
  double *d_dataset_tuple2;
  gpuErrchk(cudaMalloc(&d_dataset_tuple2, DATASET_COUNT * sizeof(double)));

	gpuErrchk(cudaMemcpy(d_dataset_tuple1, dataset_tuple_x, DATASET_COUNT * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_dataset_tuple2, dataset_tuple_y, DATASET_COUNT * sizeof(double), cudaMemcpyHostToDevice));

  
	thrust::device_ptr<double> dev_ptr_vector1 = thrust::device_pointer_cast(d_dataset_tuple1);
  thrust::device_ptr<double> dev_ptr_vector2 = thrust::device_pointer_cast(d_dataset_tuple2);
  
  auto begin = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_vector1, dev_ptr_vector2));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(dev_ptr_vector1 + DATASET_COUNT, dev_ptr_vector2 + DATASET_COUNT));

  thrust::sort(begin, end, TupleComp());
  
  double *h_vector1_output = (double *)malloc(DATASET_COUNT * sizeof(double));
	double *h_vector2_output = (double *)malloc(DATASET_COUNT * sizeof(double));

	gpuErrchk(cudaMemcpy(h_vector1_output, d_dataset_tuple1, DATASET_COUNT * sizeof(double), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_vector2_output, d_dataset_tuple2, DATASET_COUNT * sizeof(double), cudaMemcpyDeviceToHost));

  for(int i = 0; i < DATASET_COUNT; i++) {
    h_dataset[i*DIMENSION] = h_vector1_output[i];
    h_dataset[i*DIMENSION + 1] = h_vector2_output[i];
  }
 
  printf("Sorted dataset\n");
  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      printf("%f ", h_dataset[i * DIMENSION + j]);
    }
    printf("\n");
  }


  return 0;
}
