#define MAIN1
#ifdef MAIN1
#include "network.h"

#include "fstream"
#include "iostream"
#include <string>
#include <sstream>

using namespace mlp;
using namespace std;

#define GENERATE_WEIGHT
//#define GENERATE_FAULT
//#define RETRAIN

int main(int argc,char* argv[]){
	vec2d_t train_x;
	vec_t train_y;
	vec2d_t test_x;
	vec_t test_y;
	//std::ofstream ofile("weight.txt");
	/*加载MNIST数据集*/
	LOAD_MNIST_TEST(test_x, test_y);
	LOAD_MNIST_TRAIN(train_x, train_y);
	
	Mlp n(0.03, 0.01);

	n.add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6, new sigmoid_activation));
	n.add_layer(new MaxpoolingLayer(28, 28, 6, new sigmoid_activation));
	n.add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16, new sigmoid_activation));
	n.add_layer(new MaxpoolingLayer(10, 10, 16, new sigmoid_activation));
	n.add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100, new sigmoid_activation));
	n.add_layer(new FullyConnectedLayer(100, 10, new sigmoid_activation));

#ifdef GENERATE_WEIGHT
	n.train(train_x, train_y, 60000);
	std::ofstream weight_file("weight.txt");
	n.fout_weight(weight_file);
	n.test(test_x,test_y,10000);
	weight_file.close();
#else
//	std::ifstream weight_file("weight.txt");
//	n.fin_weight(weight_file);
#endif

#ifdef GENERATE_FAULT
	//std::ofstream fault_file("fault.txt");
	for (int i = 0; i < 100;i++)
	{
		std::stringstream ss;
		ss << "fault/fault" << i << ".txt";
		std::ofstream fault_file(ss.str());
		n.generateFault_varition(0.8);
		cout << i << ' ';
		float tmp = n.fault_test(test_x, test_y, 10000);
		n.fout_fault(fault_file);
		fault_file.close();
	}
#endif


#if defined(RETRAIN) && !defined(GENERATE_FAULT)
	
	//std::cout << "openfault:fault" << argv[1] << endl;
	for (int i = 0; i < 20;i++)
	{
		Mlp n(0.03, 0.01);
		n.add_layer(new FullyConnectedLayer(28 *28, 10, new sigmoid_activation));
		std::ifstream weight_file("weight.txt");
		n.fin_weight(weight_file);
		std::cout << "openfault:fault" << argv[1] <<"/fault"<<i<< endl;
		std::stringstream ss;
		ss << "model/fault" << argv[1] <<"/fault" << i << ".txt";
		std::ifstream fault_file(ss.str());
		n.fin_fault(fault_file);
		std::stringstream sst;
		sst << "model/faultsa_new/fault" << i << ".txt";
		std::ifstream fault_file_t(sst.str());
		n.fin_fault_type(fault_file_t);
		n.test(test_x, test_y, 10000);
		//cout<<"\t";
		n.fault_test(train_x, train_y, 60000);
		//cout<<"\t";
		n.remap_best();
		n.fault_test(train_x, train_y, 60000);
		//cout<<"\t";
		n.retrain_gpu_all(200, train_x, train_y, 60000, test_x, test_y, 10000);
		n.fault_test(train_x, train_y, 60000);
		cout<<endl;
		fault_file.close();
		fault_file_t.close();
		weight_file.close();
	}


	/* std::ifstream fault_file("model/fault5/fault18.txt");
	n.fin_fault(fault_file);
	std::ifstream fault_file_t("model/faultsa/fault18.txt");
	n.fin_fault_type(fault_file_t);
	n.test(test_x, test_y, 10000);
	n.fault_test(train_x, train_y, 60000);
	n.remap_best();
	n.fault_test(train_x, train_y, 60000);
	n.retrain_gpu_all(200, train_x, train_y, 60000, test_x, test_y, 10000);
	n.fault_test(train_x, train_y, 60000); */


#endif
	//std::cout << "mean_fault_error:" << mean_error_rate / 50 << std::endl;
	//weight_file.close();

#if /*defined(GENERATE_FAULT) ||*/ defined(RETRAIN)
	/* fault_file.close();
	fault_file_t.close(); */
#endif
	
	//ofile.close();
#if defined(_WIN32) || defined(_WIN64)
	//getchar();
#endif
	return 0;
}

#endif
