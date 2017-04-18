#ifndef NETWORK_H_
#define NETWORK_H_

#pragma once

#include "util.h"
#include "mnist_parser.h"
#include "output_layer.h"
#include "mnist_parser.h"
#include "fullyconnected_layer.h"
#include "convolutional_layer.h"
#include "maxpooling_layer.h"
#include "fstream"
#include "iostream"

#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
#endif


namespace mlp{
#define MAX_ITER 500
#define FIX_NUMBER 25
#define M 50
#define END_CONDITION 1e-2
//#define RETRAIN_ALL

	class Mlp
	{
	public:
		Mlp(float_t alpha, float_t lambda):
			alpha_(alpha), lambda_(lambda)
		{}

		//////GPU part
#ifdef GPU
		void initialize_gpu_memory(	const vec2d_t& train_x, const vec_t& train_y, size_t train_size,
									const vec2d_t& test_x, const vec_t& test_y, size_t test_size)
		{
			train_x_ = train_x;
			train_y_ = train_y;
			test_x_ = test_x;
			test_y_ = test_y;
			int size_pic, num_pic;
			num_pic = train_x_.size();
			size_pic = train_x_[0].size();
			checkerror(cudaMalloc(&train_x_gpu, num_pic*size_pic*sizeof(float)), "cudamalloc train_x");
			checkerror(cudaMalloc(&train_y_gpu, num_pic*sizeof(float)), "cudamalloc train_y");
			for (int i = 0; i < num_pic; i++)
			{
				checkerror(cudaMemcpy(train_x_gpu + i*size_pic, train_x_[i].data(), size_pic*sizeof(float), cudaMemcpyHostToDevice), "cuda memcpy train_x -> gpu");
			}
			checkerror(cudaMemcpy(train_y_gpu, train_y_.data(), num_pic*sizeof(float), cudaMemcpyHostToDevice), "cuda memecpy train_y -> gpu");
			num_pic = test_x_.size();
			size_pic = test_x_[0].size();
			checkerror(cudaMalloc(&test_x_gpu, num_pic*size_pic*sizeof(float)), "cudamalloc test_x");
			checkerror(cudaMalloc(&test_y_gpu, num_pic*sizeof(float)), "cudamalloc test_y");
			for (int i = 0; i < num_pic; i++)
			{
				checkerror(cudaMemcpy(test_x_gpu + i*size_pic, test_x_[i].data(), size_pic*sizeof(float), cudaMemcpyHostToDevice), "cuda memcpy test_x -> gpu");
			}
			checkerror(cudaMemcpy(test_y_gpu, test_y_.data(), num_pic*sizeof(float), cudaMemcpyHostToDevice), "cuda memecpy test_y -> gpu");

			for (auto layer : layers){
				layer->init_cuda_memory();
			}

			checkerror(cudaMalloc(&network_input, size_pic*sizeof(float)), "cudamalloc network_input");
			std::cout << "size_pic:" << size_pic << std::endl;

		}

		void realease_gpu_memory(){
			if (train_x_gpu)checkerror(cudaFree(train_x_gpu), "cudafree train_x_gpu");
			if (train_y_gpu)checkerror(cudaFree(train_y_gpu), "cudafree train_y_gpu");
			if (test_x_gpu)checkerror(cudaFree(test_x_gpu), "cudafree test_x_gpu");
			if (test_y_gpu)checkerror(cudaFree(test_y_gpu), "cudafree test_y_gpu");
			if (network_input)checkerror(cudaFree(network_input), "cudafree network_input");
		}

		float test_gpu(const vec2d_t& test_x, const vec_t& test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size_){
				if (test_gpu_once(iter)){
					bang++;
				}
				iter++;
			}
			std::cout << (float)bang / test_size_ << std::endl;
			return (float)bang / test_size_;
		}

		bool test_gpu_once(int index){
			//auto test_x_index = uniform_rand(0, test_size_ - 1);
			int test_x_index = index;
			checkerror(cudaMemcpy(network_input, test_x_[test_x_index].data(), sizeof(float)*test_x_[test_x_index].size(), cudaMemcpyHostToDevice), "cudamemcpy networkinput");
			layers[0]->input_gpu = network_input;
			for (auto layer : layers){
				layer->forward_gpu();
				if (layer->next != nullptr){
					layer->next->input_gpu = layer->output_gpu;
				}
			}
			//std::cout << "exp:" << test_y_[test_x_index];
			//std::cout << "result:";
			//disp_vec_t(layers.back()->output_);
			//return true;
			layers.back()->transfer_output_d2h();
			//disp_vec_t(layers.back()->output_);
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		void gpu_update_weight()
		{
			for (auto layer:layers)
			{
				layer->transfer_weight_d2h();
			}
		}

		void retrain_gpu(const vec2d_t& train_x, const vec_t& train_y, size_t train_size, double end_condition){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;

			for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
				layer->find_fixed(FIX_NUMBER);
				layer->transfer_fixed_h2d();
				layer->transfer_weight_h2d();
			}

			auto stop = false;
			int iter = 0;
			vec_index train_array;
			std::srand(unsigned(time(NULL)));

#ifdef RETRAIN_ALL
			for (int i = 0; i < train_size; i++)train_array.push_back(i);
#endif
			while (iter < MAX_ITER && !stop){
				iter++;
#ifdef RETRAIN_ALL
				std::random_shuffle(train_array.begin(), train_array.end());
#else
				train_array.clear();
				for (int i = 0; i < 1000; i++)train_array.push_back(uniform_rand(0, train_size_ - 1));
#endif
				auto err = retrain_once_gpu(train_array);
				//std::cout << "err: " << err << std::endl;
				if (err < end_condition) stop = true;
			}
		}

		float_t retrain_once_gpu(vec_index vec){
			float_t err = 0;
			int size = vec.size();
			int iter = 0;
			while (iter < size){
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
				assert(vec[iter] < train_size_);
				int train_x_index = vec[iter];
				//std::cout << "train_x:" <<train_x_[train_x_index].size() << std::endl;
				checkerror(cudaMemcpy(network_input, train_x_[train_x_index].data(), sizeof(float)*train_x_[train_x_index].size(), cudaMemcpyHostToDevice), "cudamemcpy networkinput");
				
				layers[0]->input_gpu = network_input;

				layers.back()->exp_y = (int)train_y_[train_x_index];

				/*期待结果*/
				//std::cout << "layer exp y: " << layers.back()->exp_y << std::endl;
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward_gpu();
					if (layer->next != nullptr){
						layer->next->input_gpu = layer->output_gpu;
					}
				}

				//layers.back()->

				/*MNIST 每一轮拟合后的结果*/
				//std::cout << (int)max_iter(layers.back()->input_) << std::endl;

				/*输出XOR每一轮拟合后的结果*/
				//disp_vec_t(layers.back()->input_);

				err += layers.back()->err;
				/*
				back propgation
				*/

				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->fix_backprop_gpu();
				}
				iter++;
			}
			return err / size;
		}

		float_t retrain_gpu_all(int iters, const vec2d_t& train_x, const vec_t& train_y, size_t train_size,
			const vec2d_t& test_x, const vec_t& test_y, size_t test_size)
		{

			this->add_layer(new OutputLayer(layers.back()->out_depth_, layers.back()->a_));

			initialize_gpu_memory(train_x,train_y,train_size,test_x,test_y,test_size);
			for (auto layer : layers)
			{
				layer->transfer_deltaW_h2d();
			}

			double data[5]={0,0,0,0,0};
			for (int i = 0; i < iters; i++)
			{
				//retrain_gpu(train_x, train_y, train_size, 0.05);
				//std::cout << "iter" << i << " ";
				//gpu_update_weight();
				//fault_test(test_x, test_y, test_size);
				retrain_gpu(train_x, train_y, train_size, 0.05);
				
				gpu_update_weight();
                bool flag = true;
				if(i%5==0)
				{
					std::cout << "iter" << i << " ";
					data[4] = data[3];
					data[3] = data[2];
					data[2] = data[1];
					data[1] = data[0];
					data[0] = fault_test(test_x, test_y, test_size);
                    float aver = (data[4]+data[3]+data[2]+data[1]+data[0])/5;
                    for(int i=0;i<5;i++)
                    {
                        if(data[i]-aver>0.005 || data[i]-aver<-0.005)flag = false;
                    }
				}
                
				if(flag && data[4] != 0)
				{
					std::cout<<"end at iter "<<i<<std::endl;
					break;
				}
                //else std::cout<<std::endl;
			}

			this->delete_layer();

			realease_gpu_memory();

			return fault_test(test_x, test_y, test_size);
		}

#endif


		/////CPU part

		void retrain(const vec2d_t& train_x, const vec_t& train_y, size_t train_size,double end_condition){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;

			this->add_layer(new OutputLayer(layers.back()->out_depth_, layers.back()->a_));

			for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
				layer->find_fixed(FIX_NUMBER);
			}

			auto stop = false;
			int iter = 0;
			vec_index train_array;
			std::srand(unsigned(time(NULL)));

#ifdef RETRAIN_ALL
			for (int i = 0; i < train_size; i++)train_array.push_back(i);
#endif
			while (iter < MAX_ITER && !stop){
				iter++;
#ifdef RETRAIN_ALL
				std::random_shuffle(train_array.begin(), train_array.end());
#else
				train_array.clear();
				for (int i = 0; i < 1000; i++)train_array.push_back(uniform_rand(0, train_size_ - 1));
#endif
				auto err = retrain_once(train_array);
				std::cout << "err: " << err << std::endl;
				if (err < end_condition) stop = true;
			}
			this->delete_layer();
		}

		void generateFault_varition(float sigma){
			for (auto layer : layers){
				layer->generateFault_varition(sigma);
			}
		}

		void generateFault_sa(){
			for (auto layer : layers){
				layer->generateFault_sa();
			}
		}

		void remap_best(){
			for (auto layer : layers){
				layer->remap_best();
			}
		}


		void train(const vec2d_t& train_x, const vec_t& train_y, size_t train_size){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;
			/*
			auto add OutputLayer as the last layer.
			*/
			this->add_layer(new OutputLayer(layers.back()->out_depth_, layers.back()->a_));

			for (auto layer : layers){
				layer->alpha_ = alpha_, layer->lambda_ = lambda_;
			}

			/*
			start training...
			*/
			auto stop = false;
			int iter = 0;
			vec_index train_array;
			std::srand(unsigned(time(NULL)));

			for (int i = 0; i < train_size; i++)train_array.push_back(i);
			
			while (iter < MAX_ITER && !stop){
				iter++;
				std::random_shuffle(train_array.begin(), train_array.end());
				auto err = train_once(train_array);
				std::cout << "err: " <<  err << std::endl;
				if (err < END_CONDITION) stop = true;
			}
			this->delete_layer();
		}

		float test(const vec2d_t& test_x, const vec_t& test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size){
				if (test_once(iter)) bang++;
				iter++;
			}
			std::cout << (float)bang / test_size_ << std::endl;
			return (float)bang / test_size_;
		}

		float fault_test(const vec2d_t& test_x, const vec_t& test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size_){
				if (fault_test_once(iter)) bang++;
				iter++;
			}
			std::cout << (float)bang / test_size_ << std::endl;
			return (float)bang / test_size_;
		}

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

		void delete_layer(){
			if (!layers.empty()){
				auto layer = layers[0];
				if (layer->next == NULL)return;
				else while (layer->next->next != NULL)layer = layer->next;
				free(layer->next);
				layer->next = NULL;
				this->layers.pop_back();
			}
		}

		//void fout_weight(std::ofstream &ofile)
		//{
		//	for (auto layer : layers){
		//		ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
		//		for (int i = 0; i < layer->out_depth_; i++){
		//			for(int j = 0; j < layer->in_depth_; j++)
		//			{
		//				ofile << layer->W_[i*layer->in_depth_ + j] << " ";
		//			}
		//			ofile << layer->b_[i] << std::endl;
		//		}
		//	}
		//}

		//void fin_weight(std::ifstream &infile)
		//{
		//	for (auto layer : layers)
		//	{
		//		int in_depth, out_depth;
		//		infile >> in_depth >> out_depth;
		//		for (int i = 0; i < out_depth; i++){
		//			for (int j = 0; j < in_depth;j++)
		//			{
		//				infile >> layer->W_[i*in_depth + j];
		//			}
		//			infile >> layer->b_[i];
		//		}
		//	}
		//	
		//}

		void fout_weight(std::ofstream &ofile)
		{
			for (auto layer : layers){
				for (int i = 0; i < layer->W_.size(); i++){
					ofile << layer->W_[i] << " ";
				}
				ofile << std::endl;
				for (int i = 0; i < layer->b_.size(); i++){
					ofile << layer->b_[i] << " ";
				}
				ofile << std::endl;
			}
		}

		void fin_weight(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				for (int i = 0; i < layer->W_.size(); i++){
					infile >> layer->W_[i];
				}

				for (int i = 0; i < layer->b_.size(); i++){
					infile >> layer->b_[i];
				}
			}
			
		}

		void fout_fixed(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_ + 1; j++)
					{
						ofile << (int)(layer->W_fix[i*layer->in_depth_+j]) << " ";
					}
					ofile << (int)(layer->b_fix[i]) << std::endl;
				}
			}
		}

		void fout_fault(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_ + 1; j++)
					{
						ofile << layer->Fault_.getFaultValue(i, j) << " ";
					}
				}
			}
		}

		void fin_fault(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				int in_depth, out_depth;
				float tmp;
				infile >> in_depth >> out_depth;
				for (int i = 0; i < out_depth; i++){
					for (int j = 0; j < in_depth + 1; j++)
					{
						infile >> tmp;
						layer->Fault_.setFaultValue(i, j, tmp);
						layer->Fault_.setFaultType(i, j, 2);
					}
				}
			}
		}

		void fout_fault_type(std::ofstream &ofile)
		{
			for (auto layer : layers){
				ofile << layer->in_depth_ << " " << layer->out_depth_ << std::endl;
				for (int i = 0; i < layer->out_depth_; i++){
					for (int j = 0; j < layer->in_depth_ + 1; j++)
					{
						ofile << layer->Fault_.getFaultType(i, j) << " ";
					}
				}
			}
		}

		void fin_fault_type(std::ifstream &infile)
		{
			for (auto layer : layers)
			{
				int in_depth, out_depth;
				int tmp;
				infile >> in_depth >> out_depth;
				for (int i = 0; i < out_depth; i++){
					for (int j = 0; j < in_depth + 1; j++)
					{
						infile >> tmp;
						layer->Fault_.setFaultType(i, j, tmp);
					}
				}
			}
		}

	private:

#ifdef GPU
		float *train_x_gpu,*train_y_gpu,*test_x_gpu,*test_y_gpu;
		float *network_input;
		float *gpu_workspace;
#endif

		size_t max_iter(const vec_t& v){
			size_t i = 0;
			float_t max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

		bool test_once(int index){
			//auto test_x_index = uniform_rand(0, test_size_ - 1);
			int test_x_index = index;
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
				layer->forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			//std::cout << "exp:" << test_y_[test_x_index];
			//std::cout << "result:";
			//disp_vec_t(layers.back()->output_);
			//return true;
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		bool fault_test_once(int index){
			//auto test_x_index = uniform_rand(0, test_size_ - 1);
			int test_x_index = index;
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
				layer->fault_forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			//std::cout << "exp:" << test_y_[test_x_index];
			//std::cout << "result:";
			//disp_vec_t(layers.back()->output_);
			//return true;
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		float_t retrain_once(vec_index vec){
			float_t err = 0;
			int size = vec.size();
			int iter = 0;
			while (iter < size){
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
				assert(vec[iter] < train_size_);
				int train_x_index = vec[iter];
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];

				/*期待结果*/
				//std::cout << "layer exp y: " << layers.back()->exp_y << std::endl;
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward();
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}

				/*MNIST 每一轮拟合后的结果*/
				//std::cout << (int)max_iter(layers.back()->input_) << std::endl;

				/*输出XOR每一轮拟合后的结果*/
				//disp_vec_t(layers.back()->input_);

				err += layers.back()->err;
				/*
				back propgation
				*/

				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->fix_backprop();
				}
				iter++;
			}
			return err / size;
		}

		float_t train_once(vec_index vec){
			float_t err = 0;
			int size = vec.size();
			int iter = 0;
			while (iter < size){
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
				assert(vec[iter] < train_size_);
				int train_x_index = vec[iter];
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];
				
				/*期待结果*/
				//std::cout << "layer exp y: " << layers.back()->exp_y << std::endl;
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward();
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}

				/*MNIST 每一轮拟合后的结果*/
				//std::cout << (int)max_iter(layers.back()->input_) << std::endl;
				
				/*输出XOR每一轮拟合后的结果*/
				//disp_vec_t(layers.back()->input_);

				err += layers.back()->err;
				/*
				back propgation
				*/

				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->back_prop();
				}
				iter++;
			}
			return err / size;
		}

		std::vector < Layer* > layers;

		size_t train_size_;
		vec2d_t train_x_;
		vec_t train_y_;

		size_t test_size_;
		vec2d_t test_x_;
		vec_t test_y_;

		float_t alpha_;
		float_t lambda_;
	};
#undef MAX_ITER
#undef M
} //namespace mlp

#endif
