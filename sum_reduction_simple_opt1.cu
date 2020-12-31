#include<iostream>
#include<vector>

const int sharedMem = 256*sizeof(double);

__global__ void redSum(double *a, double *out){

	__shared__ double red_mat[sharedMem];
	auto i = blockDim.x*blockIdx.x + threadIdx.x;
	
	red_mat[threadIdx.x] = a[i];
	__syncthreads();
	
	for(auto k = 1; k < blockDim.x; k*=2){
		auto index = 2*k*threadIdx.x;		// Leads to shared memory bank conflicts
		if(index < blockDim.x){
			red_mat[index] += red_mat[index+k];
		}
	}
	__syncthreads();
	
	if(threadIdx.x == 0){
		out[blockIdx.x] = red_mat[threadIdx.x];
	}
}


int main(){

int N = 32768;
size_t size = N *sizeof(double);

std::vector<double> h_a(N);
std::vector<double> h_out(N, 0.0);

for(auto i = 0; i < N; i++){
	h_a[i] = 1;
}

double *d_a, *d_out;
cudaMalloc(&d_a, size);
cudaMalloc(&d_out, size);

cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);

int threadsPerBlock = 256;
int blocksPerGrid = N/threadsPerBlock;
redSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_out);

cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);

std::cout << h_out[0] << std::endl;

cudaFree(d_a);
cudaFree(d_out);

return 0;
}
