#include<iostream>
#include<vector>

const int SHARED_MEM = 256;

__global__ void absoluteKernel(int *a, int *abs_a, int N){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(index<N){
		if(a[index] < 0){
			abs_a[index] = -1*a[index];
		}
		else{
			abs_a[index] = a[index];
		}
	}
}

__global__ void findmaxnorm(int *a, int *res, int N){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	// extern __shared__ int sh[];
	__shared__ int sh[SHARED_MEM*sizeof(int)];
	sh[threadIdx.x] = a[index];
	__syncthreads();
	
	int dist = blockDim.x/2;
	while (dist > 0){
		if(threadIdx.x+dist < blockDim.x){
			if(sh[threadIdx.x] < sh[threadIdx.x+dist]){
				sh[threadIdx.x] = sh[threadIdx.x+dist];
				// printf("%d\n", sh[threadIdx.x]);
			}
		}
		dist /= 2;
	}
	__syncthreads();
	
	if(threadIdx.x == 0){*res = sh[0];}
	
}

int main(){
	const int N = 1024;
	size_t size = N*sizeof(int);
	
	std::vector<int> arr(N);
	std::vector<int> absarr(N,0);
	int norm = 0;
	
	for(auto& i:arr){i = (-1)^(rand()%3)*rand()%10;}
	
	int *d_arr, *d_absarr, *d_norm;
	cudaMalloc((void **)&d_arr, size);
	cudaMalloc((void **)&d_absarr, size);
	cudaMalloc((void **)&d_norm, sizeof(int));
	
	cudaMemcpy(d_arr, arr.data(), size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 32;
	int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	
	absoluteKernel<<<blocksPerGrid,threadsPerBlock>>>(d_arr, d_absarr, N);
	cudaMemcpy(absarr.data(), d_absarr, size, cudaMemcpyDeviceToHost);
	
	// for(const auto& i:absarr){std::cout << i << std::endl;}
	
	findmaxnorm<<<blocksPerGrid,threadsPerBlock>>>(d_absarr, d_norm, N);
	cudaMemcpy(&norm, d_norm, sizeof(int), cudaMemcpyDeviceToHost);
	
	std::cout << "norm is: " << norm << std::endl;
	
	cudaFree(d_arr);
	cudaFree(d_absarr);
	cudaFree(d_norm);

	return 0;
}
