#include<iostream>
#include<vector>

__global__ void averageCal(float *a, float *b, int n){
int index = blockIdx.x*blockDim.x + threadIdx.x;
//for(int i = 0; i < n; i++){
//b[i] += a[i];
//}
b[index] += a[index];

__syncthreads();
//for(int i = 0; i < n; i++){
//b[i] /= n;
//}
b[index] /= n;

}


int main(){

int N = 100;	// Number of elements in the array
std::vector<float> num_array(N);
std::vector<float> averageVal(N, 0.0);
// float averageVal = 0;
for(auto i = 0; i < N; i++){
num_array[i] = i;
}

// Allocate device memory
size_t size = N*sizeof(float);
float *d_num;
cudaMalloc(&d_num, size);
float *d_average;
cudaMalloc(&d_average, size);

cudaMemcpy(d_num, num_array.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_average, averageVal.data(), size, cudaMemcpyHostToDevice);

//Invoke kernel
int threadperblock = 256;
int blockdim = (N + threadperblock - 1)/threadperblock;
averageCal<<<blockdim, threadperblock>>>(d_num, d_average, N);

// Copy the results
cudaMemcpy(averageVal.data(), d_average, size, cudaMemcpyDeviceToHost);

// std::cout << averageVal << std::endl;
for(int i = 0; i < N; i++){
std::cout << averageVal[i] << std::endl;
}

cudaFree(d_num);
cudaFree(d_average);

}
