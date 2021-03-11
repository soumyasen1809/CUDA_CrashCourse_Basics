#include<iostream>
#include<vector>

const int SHARED_MEM = 256*sizeof(int);

__global__ void convKernel(int *a, int *m, int *r, int N, int M){
	
	int index = threadIdx.x+ blockIdx.x*blockDim.x;
	__shared__ int sh[SHARED_MEM];
	
	sh[threadIdx.x] = a[index];
	__syncthreads();
	
	// int dist = M/2;
	int temp = 0;
	
	for(int j = 0; j < M; j++){
		if(threadIdx.x+j<blockDim.x){
			temp += sh[threadIdx.x+j]*m[j];
		}
		if(threadIdx.x+j>=blockDim.x){
			temp += a[index+j]*m[j];
		}
	}
	r[index] = temp;

}


int main(){
	int N = 1024;
	int M = 5;
	int array_size = N*sizeof(int);
	int filter_size = M*sizeof(int);
	
	std::vector<int> feature(N);
	std::vector<int> filter(M);
	std::vector<int> result(N);
	
	int *d_feature, *d_filter, *d_result;
	cudaMalloc((void **)&d_feature, array_size);	
	cudaMalloc((void **)&d_filter, filter_size);	
	cudaMalloc((void **)&d_result, array_size);
	
	for(auto& i:feature){i = rand()%10;}
	for(auto& i:filter){i = rand()%10;}
	
	cudaMemcpy(d_feature, feature.data(), array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter.data(), filter_size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 256;
	int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	convKernel<<<blocksPerGrid, threadsPerBlock>>>(d_feature, d_filter, d_result, N, M);
	
	cudaMemcpy(result.data(), d_result, array_size, cudaMemcpyDeviceToHost);
	
	for(const auto& i:result){std::cout << i << std::endl;}
	
	cudaFree(d_feature);
	cudaFree(d_filter);
	cudaFree(d_result);

	return 0;
}
