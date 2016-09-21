#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#pragma once
#include "layer.h"
#include "util.h"
#include <math.h>
#include <vector>

#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
#endif


namespace mlp{

	class FullyConnectedLayer :public Layer
	{
	public:
		FullyConnectedLayer(size_t in_depth, size_t out_depth, activation* a) :
			Layer(in_depth, out_depth, a)
		{
			output_.resize(out_depth_);
			W_.resize(in_depth_ * out_depth_);
			W_fix.resize(in_depth_ * out_depth_);
			for (int i = 0; i < in_depth_ * out_depth_; i++)
			{
				W_fix[i] = 0;
			}
			deltaW_.resize(in_depth_ * out_depth_);
			b_.resize(out_depth_);
			b_fix.resize(out_depth_);
			for (int i = 0; i < out_depth_; i++)
			{
				b_fix[i] = 0;
			}
			g_.resize(in_depth_);
			Fault_.resizeFault(out_depth_, in_depth_ + 1);
			this->init_weight();

			n = in_depth + 1;

#ifdef GPU
			if (cublasCreate(&cublas_handle)!=CUBLAS_STATUS_SUCCESS){
				std::cout << "cublas handle create failed!" << std::endl;
			}

			checkerror(cudaMalloc(&W_gpu, in_depth_*out_depth_*sizeof(float)), "malloc W_gpu");
			checkerror(cudaMalloc(&deltaW_gpu, in_depth_*out_depth_*sizeof(float)), "malloc deltaW_gpu");
			checkerror(cudaMalloc(&b_gpu, out_depth_*sizeof(float)), "malloc b_gpu");
			
			checkerror(cudaMalloc(&fault_gpu, (in_depth_+1)*out_depth_*sizeof(float)), "malloc fault_gpu");
			checkerror(cudaMalloc(&W_fix_gpu, in_depth_*out_depth_*sizeof(char)), "malloc W_fix_gpu");
			checkerror(cudaMalloc(&b_fix_gpu, out_depth_*sizeof(char)), "malloc b_fix_gpu");

			checkerror(cudaMalloc(&output_gpu, out_depth_*sizeof(float)), "malloc output_gpu");
			checkerror(cudaMalloc(&g_gpu, in_depth_*sizeof(float)), "malloc g_gpu");
#endif
		}

		void find_fixed(int number);

		void generateFault_varition(float sigma);

		void generateFault_sa();

		void init(int size);

		bool path(int u);

		double bestmatch(bool maxsum);

		void remap_best();

		void fault_forward();

		void forward();

		void fix_backprop();

		void back_prop();

		void init_weight();

#ifdef GPU
		void transfer_weight_h2d();
		void transfer_weight_d2h();
		void transfer_deltaW_h2d();
		void transfer_input_h2d();
		void transfer_output_d2h();
		void init_cuda_memory(){
			transfer_weight_h2d();
		};
		void forward_gpu();
		void forward_fault_gpu();
		void fix_backprop_gpu();
		void transfer_fault_h2d();
		void transfer_fixed_h2d();
#endif


	private:

		int n;
		int match[1024];
		double weight[1024][1024];
		double lx[1024], ly[1024];
		bool sx[1024], sy[1024];
		

		vec_t get_W(size_t index);

		vec_t get_W_step(size_t in);


		vec_t get_Fault(size_t index);

		vec_char get_Fault_Type(size_t index);

		vec_t get_Fault_in(size_t index);

		vec_char get_Fault_in_type(size_t index);

		~FullyConnectedLayer();

	};
}

#endif 