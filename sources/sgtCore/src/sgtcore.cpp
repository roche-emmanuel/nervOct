#include <sgtcore.h>

namespace sgt {

bool writeFile(const std::string& filename,void* data, unsigned long long len)
{
	boost::system::error_code err;
	boost::filesystem::remove(filename,err);
	CHECK_RET(err==0,false,"Could not remove previous file " << filename << " error_code=" << err);
	
	FILE* pFile;
	pFile = fopen(filename.c_str(), "wb");
	CHECK_RET(pFile,false,"Could not open file " << filename << " for writting.")
	size_t res = fwrite(data, 1, (size_t)len, pFile);
	CHECK_RET(res==len,false,"Could not write all the bytes for file: "<< filename<< ", total=" <<len<<", written=" << res);
	int ret = fclose(pFile);
	CHECK_RET(ret==0,false,"Could not close file: "<< filename<< " properly.");
	
	return true;
}

}

