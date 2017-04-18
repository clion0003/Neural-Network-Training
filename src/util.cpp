#include "util.h"

#define MAX_WEIGHT 19.062
#define MIN_WEIGHT 0

namespace mlp{
	void disp_vec_t(const vec_t& v){
		for (auto i : v)
			std::cout << i << "\t";
		std::cout << "\n";
	}

	void disp_vec2d_t(const vec2d_t& v){
		for (auto i : v){
			for (auto i_ : i)
				std::cout << i_ << "\t";
			std::cout << "\n";
		}
	}

	float_t fault_dot(const vec_t& x, const vec_t& w, const vec_t& f, const vec_char& ft)
	{
		assert(x.size() == w.size());
		assert(x.size() == f.size());
		assert(f.size() == ft.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			if (ft[i] == 2)//2-variation
			{
				sum += x[i] * w[i] * f[i];
			}
			else if (ft[i] == 0)  //0-sa0
			{
				sum += x[i] * MAX_WEIGHT;
			}
			else if (ft[i] == 3)
			{
				sum += x[i] * (-MAX_WEIGHT);
			}
			else if (ft[i] == 1)//1-sa1
			{
				sum += x[i] * MIN_WEIGHT;
			}
		}
		return sum;
	}

	float_t fault_b(const float t, const float fault, const int fault_type)
	{
		if (fault_type == 2)return t*(1 + fault);
		if (fault_type == 1)return MIN_WEIGHT;
		if (fault_type == 0)return MAX_WEIGHT;
		if (fault_type == 3)return -MAX_WEIGHT;
		return 0;
	}

	float_t dot(const vec_t& x, const vec_t& w){
		assert(x.size() == w.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			sum += x[i] * w[i];
		}
		return sum;
	}

	float_t abs_dot(const vec_t& w, const vec_t& f, const vec_char& ft){
		assert(w.size() == f.size());
		assert(f.size() == ft.size());
		float_t sum = 0;
		for (size_t i = 0; i < w.size(); i++){
			if (ft[i] == 2)//2-variation
			{
				sum += fabs(w[i] * f[i]);
			}
			else if (ft[i] == 0)//0-sa0
			{
				sum += fabs(MAX_WEIGHT - w[i]);
			}
			else if (ft[i] == 1)//1-sa1
			{
				sum += fabs(MIN_WEIGHT - w[i]);
			}
			else if (ft[i] == 3)
			{
				sum += fabs(-MAX_WEIGHT - w[i]);
			}
		}
		return sum;
	}
	vec_t f_muti_vec(float_t x, const vec_t& v){
		vec_t r;
		for_each(v.begin(), v.end(), [&](float_t i){
			r.push_back(x * i);
		});
		return r;
	}

	vec_t get_W(size_t index, size_t in_size_, const vec_t& W_){
		vec_t v;
		for (int i = 0; i < in_size_; i++){
			v.push_back(W_[index * in_size_ + i]);
		}
		return v;
	}

#ifdef GPU
	int checkerror(cudaError_t cudaStatus, std::string error_string)
	{
		if (cudaStatus != cudaSuccess) {
			std::cout << error_string << std::endl;
			return 1;
		}
		return 0;
	}
#endif
}