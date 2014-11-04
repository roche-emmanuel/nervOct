#include <nervMBP.h>

#include "CudaInit.h"
#include "MBP/BackPropagation.h"
// #include "common/Utilities.h"

extern "C" {

bool isCudaSupported()
{
	CudaDevice device;
	return device.SupportsCuda();
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
	int num_weights, double* weights)
{
	using namespace GPUMLib;
	
	// first we prepare the layer size vector:
	int nl = lsizes.size();
	HostArray<int> sizeLayers(nl);
	for(int i=0;i<nl;++i) {
		sizeLayers[i] = lsizes[i];
	}

	return false;
}

}
