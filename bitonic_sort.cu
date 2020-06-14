#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T>
struct ShouldSwap {
  __host__ __device__ virtual bool operator()(const T left,
                                              const T right) const;
};

template <typename T>
__host__ __device__ __inline__ void swap(T* a, T* r);

template <typename T>
__global__ void bubbleSort(T* v, const unsigned int n,
                           ShouldSwap<T> shouldSwap);

int main(int argc, char** argv) {
  // vector size
  const unsigned int size = 10;

  // host vector
  int h_v[size] = {3, 7, 1, 10, 6, 9, 5, 2, 8, 4};

  // device vector
  int* d_v = 0;

  cudaMalloc((void**)&d_v, size * sizeof(int));

  cudaMemcpy(d_v, h_v, size * sizeof(int), cudaMemcpyHostToDevice);

  dim3 grdDim(1, 1, 1);
  dim3 blkDim(size / 2, 1, 1);

  ShouldSwap<int> shouldSwap;

  bubbleSort<int><<<grdDim, blkDim>>>(d_v, size, shouldSwap);
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    // system("pause"); // when using VisStudio
    exit(-1);
  }

  cudaMemcpy(h_v, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_v);

  for (int i = 0; i < size; i++) {
    std::cout << (i == 0 ? "{" : "") << h_v[i] << (i < size - 1 ? " ," : "}");
  }
  std::cout << std::endl;

  // system("pause"); // when using VisStudio
  return 0;
}

template <typename T>
__host__ __device__ bool ShouldSwap<T>::operator()(const T left,
                                                   const T right) const {
  return left > right;
}

template <typename T>
__host__ __device__ __inline__ void swap(T* a, T* b) {
  T tmp = *a;
  *a = *b;
  *b = tmp;
}

template <typename T>
__global__ void bubbleSort(T* v, const unsigned int n,
                           ShouldSwap<T> shouldSwap) {
  const unsigned int tIdx = threadIdx.x;

  for (unsigned int i = 0; i < n; i++) {
    unsigned int offset = i % 2;
    unsigned int indiceGauche = 2 * tIdx + offset;
    unsigned int indiceDroite = indiceGauche + 1;

    if (indiceDroite < n) {
      if (shouldSwap(v[indiceGauche], v[indiceDroite])) {
        swap<T>(&v[indiceGauche], &v[indiceDroite]);
      }
    }
    __syncthreads();
  }
}