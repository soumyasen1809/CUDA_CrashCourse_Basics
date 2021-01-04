// CUDA code to find the maximum in an array

#include<iostream>
#include<vector>
#include<cstdlib>
#include<algorithm>

const int SHARED_MEM = 256;

__global__ void maxFinder(double *arr, double *m, int N){
auto index = blockDim.x*blockIdx.x+threadIdx.x;

__shared__ double cache[SHARED_MEM];

float temp = 0;
int stride = blockDim.x;
int offset = 0;

while(index+offset < N){
temp = fmaxf(temp, arr[index+offset]);
offset += stride;
}
cache[threadIdx.x] = temp;
__syncthreads();

int i = blockDim.x/2;
while(i > 0){
cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x+i]);
i /= 2;
}
__syncthreads();

if(threadIdx.x == 0){
*m = cache[0];
}

}


int main(){
int N = 10240;
size_t size = N*sizeof(double);

std::vector<double> array(N);
for(auto& i:array){i = rand()%100;}
double h_max;
double h_max_check;
h_max_check = *std::max_element(array.begin(), array.end());

double *d_arr, *d_max;
cudaMalloc(&d_arr, size);
cudaMalloc(&d_max, sizeof(double));

cudaMemcpy(d_arr, array.data(), size, cudaMemcpyHostToDevice);

int threadsPerBlock = 64;
int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
maxFinder<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_max, N);

cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);

std::cout << "The max value in the array is: " << h_max << std::endl;
std::cout << "The max value with std library is: " << h_max_check << std::endl;

cudaFree(d_arr);
cudaFree(d_max);

return 0;
}
