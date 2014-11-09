#include <nervMBP.h>

#include "CudaInit.h"
#include "BackPropagation.h"	

#define INITIAL_LEARNING_RATE (CUDA_VALUE(0.7))

extern "C" {

bool isCudaSupported()
{
	CudaDevice device;
	return device.SupportsCuda();
	return true;
}

void showCudaInfo()
{
	CudaDevice device;
	CHECK(device.SupportsCuda(),"Cannot show info if cuda is not supported.");
	device.ShowInfo();
}

bool trainBP(const std::vector<int>& lsizes, 
	int num_inputs, double* inputs,
	int num_outputs, double* outputs,
	int num_weights, double* weights,
	double& rms_stop, int max_iter)
{
	CHECK_RET(isCudaSupported(),false,"CUDA is not supported, cannot perform training.");

	using namespace GPUMLib;

	// first we prepare the layer size vector:
	int nl = (int)lsizes.size();
	HostArray<int> sizeLayers(nl);
	for(int i=0;i<nl;++i) {
		sizeLayers[i] = lsizes[i];
	}

	// build the input/output matrices from the raw values:
	int num_features = lsizes[0];
	int num_samples = num_inputs/num_features;
	int num_labels = lsizes[nl-1];
	int num_samples2 = num_outputs/num_labels;
	
	CHECK_RET(num_samples==num_samples2,false,"Mismatch in number of samples computation: "<<num_samples <<"!="<<num_samples2);
	CHECK_RET(sizeof(cudafloat)==8,false,"Invalid size for cudafloat: "<< sizeof(cudafloat) <<"!=8");

	HostMatrix<cudafloat> input_mat(num_samples,num_features,RowMajor);
	HostMatrix<cudafloat> output_mat(num_samples,num_labels,RowMajor);

	memcpy((void*)input_mat.Pointer(),(void*)inputs,sizeof(double)*num_inputs);
	memcpy((void*)output_mat.Pointer(),(void*)outputs,sizeof(double)*num_outputs);

	BackPropagation bp(sizeLayers, input_mat, output_mat, INITIAL_LEARNING_RATE);

	// Sanity check on the backpropagation class object:
	CHECK_RET(bp.GetNumberLayers()==(nl-1),false,"Invalid number of layers: "<<bp.GetNumberLayers()<<"!="<<(nl-1));
	CHECK_RET(bp.GetNumberInputs()==num_features,false,"Invalid number of inputs: "<<bp.GetNumberInputs()<<"!="<<num_features);
	CHECK_RET(bp.GetNumberOutputs()==num_labels,false,"Invalid number of outputs: "<<bp.GetNumberOutputs()<<"!="<<num_labels);

	// Set the network training parameters
	bp.SetRobustLearning(true);
	bp.SetRobustFactor(CUDA_VALUE(0.5));
	bp.SetMaxPercentageRMSGrow(CUDA_VALUE(0.001));

	bp.SetMaxStepSize(CUDA_VALUE(10.0));

	bp.SetUpStepSizeFactor(CUDA_VALUE(1.1));
	bp.SetDownStepSizeFactor(CUDA_VALUE(0.9));

	bp.SetMomentum(CUDA_VALUE(0.7));
	
	// Ensure we get the RMS once to always override any previous value.
	bp.GetRMS();

	logDEBUG("Starting training...")
	for(int i=0;i<max_iter;++i) {
		bp.TrainOneEpoch();
		if(bp.GetRMSestimate() < rms_stop)
			break;
	}

	// update the actual rms value:
	rms_stop = bp.GetRMS();
	
	logDEBUG("Training done in "<<bp.GetEpoch()<<" epochs. RMS="<< rms_stop)

	// Try the naive implementation to get the weights from the network:
	// note that we have to discard the first input layer in this process:
	nl=nl-1;
	double* ptr = weights;
	int count = 0;
	HostArray<cudafloat> ww;
	int len;
	for(int i=0;i<nl;++i) {
		ww = bp.GetLayerWeights(i);
		len = ww.Length();
		count += len;
		memcpy((void*)ptr,(void*)ww.Pointer(),sizeof(cudafloat)*len);
		ptr += len;
	}

	CHECK_RET(count == num_weights,false,"Invalid number of weights: "<<count<<"!="<<num_weights);

	return true;
}

}
