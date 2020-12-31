// Inner product of 2 vectors

#include<iostream>
#include<vector>

__global__ void vecProd(float *a, float *b, float *c, int N){
int i = blockDim.x*blockIdx.x + threadIdx.x;

if (i < N){
c[i] = a[i]*b[i];
}
}

int main(){

std::vector<float> v1;
std::vector<float> v2;
std::vector<float> v3;

for(auto i = 0; i < 10; i++){
v1.emplace_back(i);
v2.emplace_back(2*i);
v3.emplace_back(0);
}

size_t size = v1.size()*sizeof(float);

float *d_v1, *d_v2, *d_v3;
cudaMalloc(&d_v1, size);
cudaMalloc(&d_v2, size);
cudaMalloc(&d_v3, size);

cudaMemcpy(d_v1, v1.data(), size, cudaMemcpyHostToDevice);
cudaMemcpy(d_v2, v2.data(), size, cudaMemcpyHostToDevice);

int threadsPerBlock = 256;
int numBlocks = (v1.size() + threadsPerBlock - 1)/threadsPerBlock;
vecProd<<<numBlocks, threadsPerBlock>>>(d_v1, d_v2, d_v3, v1.size());

cudaMemcpy(v3.data(), d_v3, size, cudaMemcpyDeviceToHost);

for(auto& i:v3){
std::cout << i << std::endl;
}

cudaFree(d_v1);
cudaFree(d_v2);
cudaFree(d_v3);

return 0;
}
