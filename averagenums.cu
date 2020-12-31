#include<iostream>
#include<vector>

__global__ void averageCal(float *a, float *b, int n){
// int index = blockIdx.x*blockDim.x + threadIdx.x;
for(int i = 0; i < n; i++){
*b += a[i];
}

__syncthreads();
*b /= n;

}


int main(){

int N = 10;	// Number of elements in the array
std::vector<float> num_array(N);
// std::vector<float> average_array(N);
float averageVal = 0;
for(auto i = 0; i < N; i++){
num_array[i] = i;
}

// Allocate device memory
size_t size = N*sizeof(float);
float *d_num;
cudaMalloc(&d_num, size);
float *d_average;
cudaMalloc(&d_average, sizeof(float));

cudaMemcpy(d_num, num_array.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_average, &averageVal, sizeof(float), cudaMemcpyHostToDevice);

//Invoke kernel
int threadperblock = 256;
int blockdim = (N + threadperblock - 1)/threadperblock;
averageCal<<<blockdim, threadperblock>>>(d_num, d_average, N);

// Copy the results
cudaMemcpy(&averageVal, d_average, size, cudaMemcpyDeviceToHost);

std::cout << averageVal << std::endl;

cudaFree(d_num);
cudaFree(d_average);

}
