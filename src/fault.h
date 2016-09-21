#ifndef FAULT_H_
#define FAULT_H_
#pragma once
#include "util.h"
#include <cassert>


namespace mlp{
	class Fault
	{
	public:


		void generateVariation(double sigma);

		void generateLogVariation(double sigma);

		void generateSA0(double sa0_fault_ratio, int sa0_faulty_columns);

		void generateSA1(double sa1_fault_ratio, int sa1_faulty_rows, int sa1_bad_block);

		void resizeFault(int row_size, int column_size);

		float getFaultValue(int row_index, int col_index);

		int	getFaultType(int row_index, int col_index);

		void setFaultValue(int row_index, int col_index, double v);

		void setFaultType(int row_index, int col_index, int v);

		Fault(){}

		Fault(int row_size, int column_size);


	private:
		int column = 0;
		int row = 0;
		vec2d_fault fault_array;
	};
}



#endif