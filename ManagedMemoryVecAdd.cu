// sudo nvprof --unified-memory-profiling off ./ManagedMemoryVecAdd
// Use this command for profiling without errors for unified memory profiling

#include<iostream>

__global__ void vecAdd(int *a, int *b, int *c, int N){
int i = blockDim.x * blockIdx.x + threadIdx.x;

if(i < N){
c[i] = a[i] + b[i];
}
}

__global__ void squareVec(int *a, int *b, int N){
int i = blockDim.x * blockIdx.x + threadIdx.x;

if(i < N){
b[i] = a[i]*a[i];
}
}


int main(){

int N = 20;
size_t size = N * sizeof(int);

int *a, *b, *c;
cudaMallocManaged(&a, size);							// Unified memory; ALWAYS use cudaMemPrefetchAync() with Unified memory to reduce overhead time
cudaMallocManaged(&b, size);
cudaMallocManaged(&c, size);

for(auto i = 0; i < N; i ++){
a[i] = i;
b[i] = 2*i;
}

int id = cudaGetDevice(&id);							// Get the device ID
cudaMemPrefetchAsync(a, size, id);						// Use the device ID to prefetch 'a' to the GPU memory
cudaMemPrefetchAsync(b, size, id);
cudaMemPrefetchAsync(c, size, id);

int NumThreadsPerBlock = 256;
int BlockSize = (N + NumThreadsPerBlock -1)/NumThreadsPerBlock;
vecAdd<<<BlockSize, NumThreadsPerBlock>>>(a, b, c, N);
cudaDeviceSynchronize();							// Sunchronize all the threads before moving forward

cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);				// Prefetch 'a' to the CPU memory; directly use built-in function cudaCpuDeviceId
cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);
cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

std::cout << "Printing the vector" << std::endl;
for(auto i = 0; i < N; i++){
std::cout << c[i] << std::endl;
}


cudaFree(a);
cudaFree(b);

int *c_squared;
cudaMallocManaged(&c_squared, size);

int id2 = cudaGetDevice(&id);
cudaMemPrefetchAsync(c, size, id);
cudaMemPrefetchAsync(c_squared, size, id2);

squareVec<<<BlockSize, NumThreadsPerBlock>>>(c,c_squared, N);
cudaDeviceSynchronize();

cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
cudaMemPrefetchAsync(c_squared, size, cudaCpuDeviceId);

std::cout << "Printing the vector squared" << std::endl;
for(auto i = 0; i < N; i++){
std::cout << c_squared[i] << std::endl;
}

cudaFree(c_squared);
cudaFree(c);

return 0;
}
