#ifndef LAYER_H_
#define LAYER_H_
#pragma once
#include "activation.h"
#include "fault.h"


#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
#endif

namespace mlp{
	class Layer
	{
	public:
		Layer(size_t in_depth,
			size_t out_depth, activation* a) :
			in_depth_(in_depth), out_depth_(out_depth), a_(a)
		{}

		virtual void init_weight() = 0;
		virtual void forward() = 0;
		virtual void back_prop() = 0;
		virtual void fix_backprop() = 0;
		virtual void fault_forward() = 0;
		virtual void generateFault_varition(float sigma) = 0;
		virtual void generateFault_sa() = 0;
		virtual void find_fixed(int number) = 0;
		virtual void remap_best() = 0;

#ifdef GPU
		virtual void transfer_weight_h2d() = 0;
		virtual void transfer_weight_d2h() = 0;
		virtual void transfer_input_h2d() = 0;
		virtual void transfer_output_d2h() = 0;
		virtual void init_cuda_memory() = 0;
		virtual void forward_gpu() = 0;
		virtual void forward_fault_gpu() = 0;
		virtual void fix_backprop_gpu() = 0;
		virtual void transfer_fault_h2d() = 0;
		virtual void transfer_fixed_h2d() = 0;
		virtual void transfer_deltaW_h2d() = 0;
		cublasHandle_t cublas_handle;
#endif

		size_t in_depth_;
		size_t out_depth_;

		vec_t W_;
		vec_t deltaW_; //last iter weight change for momentum;

#ifdef GPU
		float *W_gpu;
		float *deltaW_gpu;
#endif

		vec_t b_;

#ifdef GPU
		float *b_gpu;
#endif

		Fault Fault_;

#ifdef GPU
		float *fault_gpu,*fault_gpu_mid;
#endif

		vec_char W_fix;
		vec_char b_fix;
		size_t fixed_number;

#ifdef GPU
		char *W_fix_gpu;
		char *b_fix_gpu;
#endif

		vec_t F_;

		activation* a_;

		vec_t input_;
		vec_t output_;

#ifdef GPU
		float *input_gpu;
		float *output_gpu;
#endif

		Layer* next;

		float_t alpha_; // learning rate
		float_t lambda_; // momentum
		vec_t g_; // err terms

#ifdef GPU
		float *g_gpu;
#endif

		/*output*/
		float_t err;
		int exp_y;
		vec_t exp_y_vec;

#ifdef GPU
		float *exp_y_gpu;
#endif

		
		

	};
} //namspace mlp

#endif