#include<iostream>
#include<vector>
#include<cstdlib>

__global__ void convolution_kernel(double *arr, double *mask, double *output, int N, int M){
	auto i = blockDim.x*blockIdx.x+threadIdx.x;
	
	auto start = i - (M/2);
	
	auto temp = 0.0;
	for(auto k = 0; k < M; k++){
		if((start+k >=0) && (start+k <N)){
			temp += arr[start+k]*mask[k];
		}
	}
	output[i] = temp;
}

int main(){

int N = 1048576;	// size of the array = 2^20
size_t size_N = N*sizeof(double);
int M = 7;	// size of the mask
size_t size_M = M*sizeof(double);
std::vector<double> h_array(N);
std::vector<double> h_mask(M);
std::vector<double> h_output(N);

for(auto& i:h_array){i = rand()%100;}
for(auto& j:h_mask){j = rand()%10;}

double *d_array, *d_mask, *d_output;
cudaMalloc(&d_array, size_N);
cudaMalloc(&d_output, size_N);
cudaMalloc(&d_mask, size_M);

cudaMemcpy(d_array, h_array.data(), size_N, cudaMemcpyHostToDevice);
cudaMemcpy(d_mask, h_mask.data(), size_M, cudaMemcpyHostToDevice);

int threadsPerBlock = 32;
int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
convolution_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_mask, d_output, N, M);

cudaMemcpy(h_output.data(), d_output, size_N, cudaMemcpyDeviceToHost);

// Uncomment to print the output
// for(auto& i:h_output){std::cout << i << std::endl;}

cudaFree(d_array);
cudaFree(d_output);
cudaFree(d_mask);

return 0;
}
