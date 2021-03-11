#include<iostream>

const int SHARED_MEM_SIZE = 128*sizeof(int);

__global__ void ReverseFunc(int *a, int *r, int N){
	__shared__ int sh[SHARED_MEM_SIZE];
	int id = threadIdx.x + blockDim.x*blockIdx.x;

	sh[threadIdx.x] = a[id];
	__syncthreads();
	r[id] = sh[blockDim.x-threadIdx.x-1];
}

int main(){
	int *a, *r;
	int *d_a, *d_r;

	int N = 1024;
	int size = N*sizeof(int);


	a = (int*)malloc(size);
	r = (int*)malloc(size);
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_r, size);

	for(int i = 0; i < N; i++){a[i] = i;}

	cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);

	int threadsPerBlock = 64;
	int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	ReverseFunc<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_r, N);
	// cudaThreadSynchronize();

	cudaMemcpy(r,d_r,size,cudaMemcpyDeviceToHost);

	// for(int i = 0; i< N; i++){std::cout << r[i] << std::endl;}


	free(a);
	free(r);
	cudaFree(d_a);
	cudaFree(d_r);

	return 0;
}
