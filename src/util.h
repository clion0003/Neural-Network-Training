#ifndef UTIL_H_
#define UTIL_H_

#pragma once

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>
#include <time.h>
#include <stdlib.h>

#include "boost/random.hpp"

#ifdef GPU
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <string>
#endif

namespace mlp {
	typedef std::vector<float_t> vec_t;
	typedef std::vector<std::vector<float_t>> vec2d_t;
	typedef std::vector<char> vec_char;
	typedef std::vector<std::vector<char>> vec2d_char;
	typedef std::vector<int> vec_index;

	struct fault
	{
		int type;  //0-sa0;  1-sa1;   2-variation
		double value; //value of variation
	};

	typedef std::vector<fault> vec_fault;
	typedef std::vector<std::vector<fault>> vec2d_fault;

	inline int uniform_rand(int min, int max) {
		static boost::mt19937 gen(0);
		boost::uniform_smallint<> dst(min, max);
		return dst(gen);
	}

	template<typename T>
	inline T uniform_rand(T min, T max) {
		static boost::mt19937 gen(0);
		boost::uniform_real<T> dst(min, max);
		return dst(gen);
	}

	template<typename Iter>
	void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
		for (Iter it = begin; it != end; ++it)
			*it = uniform_rand(min, max);
	}

	void disp_vec_t(const vec_t& v);

	void disp_vec2d_t(const vec2d_t& v);

	float_t fault_b(const float t, const float fault, const int fault_type);

	float_t fault_dot(const vec_t& x, const vec_t& w, const vec_t& f, const vec_char& ft);

	float_t dot(const vec_t& x, const vec_t& w);

	float_t abs_dot(const vec_t& w, const vec_t& f, const vec_char& ft);

	vec_t f_muti_vec(float_t x, const vec_t& v);

	vec_t get_W(size_t index, size_t in_size_, const vec_t& W_);

#ifdef GPU
	int checkerror(cudaError_t cudaStatus, std::string error_string);
#endif
} // namespace mlp

#endif //UTIL_H_