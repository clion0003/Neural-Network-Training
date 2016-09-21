#ifdef GPU
#include "output_layer.h"

namespace mlp{

	__global__ void dsigmoidKernel(float *g_, float *exp, float *output, int max_index)
	{
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < max_index)g_[i] = (exp[i]-output[i])*(output[i] * (1 - output[i]));
	}


	void OutputLayer::forward_gpu(){
		this->err = 0;
		exp_y_vec.clear();
		/*XOR 使用下面一行代码*/
		//exp_y_vec.push_back(this -> exp_y);

		/*MNIST 使用下面两行代码 */
		exp_y_vec.resize(in_depth_);
		for (size_t i = 0; i < exp_y_vec.size(); i++)exp_y_vec[i] = 0;
		exp_y_vec[this->exp_y] = 1;
		
		checkerror(cudaMemcpy(exp_y_gpu, exp_y_vec.data(), sizeof(float)*in_depth_, cudaMemcpyHostToDevice),"exp_y h2d");

		vec_t output_tmp;
		output_tmp.resize(in_depth_);
		checkerror(cudaMemcpy(output_tmp.data(), input_gpu, sizeof(float)*in_depth_, cudaMemcpyDeviceToHost), "output d2h");

		for (size_t i = 0; i < in_depth_; i++){
			err += 0.5 * (exp_y_vec[i] - output_tmp[i]) *
				(exp_y_vec[i] - output_tmp[i]);
		}
		output_gpu = input_gpu;
	}

	void OutputLayer::fix_backprop_gpu(){
		dim3 thread_num(32);
		dim3 block_num((in_depth_+31) / 32);
		dsigmoidKernel << <block_num, thread_num >> >(g_gpu, exp_y_gpu, output_gpu, in_depth_);
		checkerror(cudaDeviceSynchronize(), "dx sigmoid kernel");
	}

}

#endif