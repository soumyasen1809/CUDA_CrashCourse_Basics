#include<iostream>
#include<vector>

const int sharedMem = 256*sizeof(double);

__global__ void redSum(double *a, double *out){

	__shared__ double red_mat[sharedMem];
	auto i = blockDim.x*blockIdx.x + threadIdx.x;
	
	red_mat[threadIdx.x] = a[i];
	__syncthreads();
	
	for(auto k = 1; k < blockDim.x; k*=2){
		if(threadIdx.x % (2*k) == 0){		// modular is an expensive operation, avoid it for GPU computing
			red_mat[threadIdx.x] += red_mat[threadIdx.x + k];
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
	h_a[i] = 2*i;
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
