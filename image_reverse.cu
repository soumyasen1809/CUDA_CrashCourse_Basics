#include<iostream>

struct colours{
	int red;
	int green;
	int blue;
};

__global__ void imageReverse(colours* c_arr, colours* rev_c_arr, int N){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	rev_c_arr[index].red = 255 - c_arr[index].red;
	rev_c_arr[index].green = 255 - c_arr[index].green;
	rev_c_arr[index].blue = 255 - c_arr[index].blue;
}

int main(){
	int N = 1024;
	int size = N*sizeof(colours);
	
	colours colour_array[N], reverse_colour_array[N];
	colours* d_colour_array, *d_reverse_colour_array;
	
	for(auto& i:colour_array){
		i.red = rand()%255;
		i.green = rand()%255;
		i.blue = rand()%255;
	}
	
	cudaMalloc(&d_colour_array, size);
	cudaMalloc(&d_reverse_colour_array, size);
	
	cudaMemcpy(d_colour_array, colour_array, size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 64;
	int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	imageReverse<<<blocksPerGrid,threadsPerBlock>>>(d_colour_array, d_reverse_colour_array, N);
	
	cudaMemcpy(reverse_colour_array, d_reverse_colour_array, size, cudaMemcpyDeviceToHost);
	
	// for(const auto& i:reverse_colour_array){
	// 	std::cout << i.red << "," << i.green << "," << i.blue << std::endl;
	// }
	
	cudaFree(d_colour_array);
	cudaFree(d_reverse_colour_array);

	return 0;
}
