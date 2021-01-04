#include<iostream>
#include<vector>
#include<random>

const int SHARED_MEM = 256;

__global__ void dotProduct(int *x, int *y, int *dot, int N){
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    __shared__ int cache[SHARED_MEM];

    int offset = 0;
    int stride = blockDim.x;

    while(index+offset<N){
        cache[threadIdx.x] += x[index+offset]*y[index+offset];
        offset += stride;
    }
    __syncthreads();
    
    int i = blockDim.x/2;
    while(i > 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] += cache[threadIdx.x+i];
        }
        i/=2;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(dot, cache[0]);
        // *dot = cache[0];
    }

}


int main(){
    int N = 10240;
    size_t size = N*sizeof(int);

    std::vector<int> h_x(N);
    std::vector<int> h_y(N);
    for(auto i = 0; i < N; i++){
        h_x[i] = rand()%10;
        h_y[i] = rand()%10;
    }

    int *h_dot;
    h_dot = (int*)malloc(sizeof(int));

    int *d_x, *d_y, *d_dot;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_dot, sizeof(int));
    cudaMemset(d_dot, 0, sizeof(int));

    cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice);

    auto threadsPerBlock = 32;
    auto blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
    dotProduct<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,d_dot,N);

    cudaMemcpy(h_dot, d_dot, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *h_dot << std::endl;



    free(h_dot);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dot);

    return 0;
}
