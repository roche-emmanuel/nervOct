#ifndef NERV_TRAININGSET_H_
#define NERV_TRAININGSET_H_

#include <nervcuda.h>
#include <sgtcore.h>

namespace nerv {

template<typename T>
class TrainingSet {
public:
    enum TrainMode {
        TRAIN_ZERO,
        TRAIN_DEBUG,
        TRAIN_RANDOM
    };

    typedef T value_type;

    TrainingSet();
    TrainingSet(unsigned int minL, unsigned int maxL, unsigned int minN, unsigned int maxN, 
        unsigned int minS, unsigned int maxS, unsigned int outputN = 0, unsigned int inputN = 0, unsigned int mode = TRAIN_DEBUG);
    TrainingSet(std::vector<unsigned int> lsizes, unsigned int nsamples, unsigned int mode = TRAIN_DEBUG);
    ~TrainingSet();

    void uninit();
    void init(std::vector<unsigned int> lsizes, unsigned int nsamples, unsigned int mode);

    inline unsigned int nl() const { return _nl; }
    inline unsigned int nt() const { return _nt; }    
    inline unsigned int nsamples() const { return _nsamples; }
    
    inline unsigned int* lsizes() const { return _lsizes; }

    inline value_type* X_train() const { return _X_train; }
    inline unsigned int X_train_size() const { return _nx; }
    
    inline value_type* y_train() const { return _y_train; }
    inline unsigned int y_train_size() const { return _ny; }

    inline value_type* params() const { return _params; }
    inline unsigned int np() const { return _np; }

    inline TrainingSet& lambda(value_type val) { _lambda = val; return *this; }
    inline value_type lambda() const { return _lambda; }

    inline TrainingSet& maxiter(int val) { _maxiter = val; return *this; }
    inline int maxiter() const { return _maxiter; }

    value_type random_real(value_type mini, value_type maxi);
    unsigned int random_uint(unsigned int mini, unsigned int maxi);

    void setupDebug();
    void setupRandom();

protected:
    unsigned int _nl; // number of layers
    unsigned int _nt; // number of theta matrices
    unsigned int _np; // number of parameters
    unsigned int _nx; // number of element in X
    unsigned int _ny; // number of element in y

    unsigned int _nsamples; // number of samples.

    unsigned int* _lsizes;

    int _maxiter; // max number of iterations.

    value_type _lambda;
    
    value_type* _X_train;
    value_type* _y_train;
    value_type* _params;
};

template <typename T>
TrainingSet<T>::TrainingSet(unsigned int minL, unsigned int maxL, unsigned int minN, unsigned int maxN, 
    unsigned int minS, unsigned int maxS, unsigned int outputN, unsigned int inputN, unsigned int mode)
    : _nl(0), _nt(0), _nsamples(0), _lambda(0), _maxiter(0), _np(0), _nx(0), _ny(0),
  _lsizes(nullptr), _X_train(nullptr), _y_train(nullptr), _params(nullptr)
{
    unsigned int nl = random_uint(minL,maxL);

    std::vector<unsigned int> lsizes;
    for(unsigned int i=0;i<nl;++i) {
        unsigned int ls = random_uint(minN,maxN);

        if(i==0 && inputN>0) {
            ls = inputN;
        }
        else if(i==(nl-1) && outputN>0) {
            ls = outputN;
        }

        lsizes.push_back(ls);
    }

    unsigned int ns = random_uint(minS,maxS);

    init(lsizes,ns,mode);
}

template<typename T>
TrainingSet<T>::TrainingSet(std::vector<unsigned int> lsizes, unsigned int nsamples, unsigned int mode)
  : _nl(0), _nt(0), _nsamples(0), _lambda(0), _maxiter(0), _np(0), _nx(0), _ny(0),
  _lsizes(nullptr), _X_train(nullptr), _y_train(nullptr), _params(nullptr) 
{
    init(lsizes, nsamples, mode);
}

template<typename T>
TrainingSet<T>::TrainingSet()
  : _nl(0), _nt(0), _nsamples(0), _lambda(0), _maxiter(0), _np(0), _nx(0), _ny(0),
  _lsizes(nullptr), _X_train(nullptr), _y_train(nullptr), _params(nullptr) 
{

}


template<typename T>
TrainingSet<T>::~TrainingSet()
{
    uninit();
}

template<typename T>
void TrainingSet<T>::uninit()
{
    delete [] _lsizes;
    _lsizes = nullptr;
    delete [] _X_train;
    _X_train = nullptr;
    delete [] _y_train;
    _y_train = nullptr;
    delete [] _params;
    _params = nullptr;
}

template<typename T>
void TrainingSet<T>::init(std::vector<unsigned int> lsizes, unsigned int nsamples, unsigned int mode)
{
    uninit();

    _nl = (unsigned int)lsizes.size();
    _nt = _nl-1;
    _nsamples = nsamples;
    _lsizes = new unsigned int[_nl];
    for(unsigned int i=0;i<_nl;++i) {
        _lsizes[i] = lsizes[i];
    }

    _nx = nsamples*lsizes[0];
    _X_train = new value_type[_nx];

    memset(_X_train,0,sizeof(value_type)*_nx);

    _ny = nsamples*lsizes[_nt];
    _y_train = new value_type[_ny];
    memset(_y_train,0,sizeof(value_type)*_ny);

    _np = 0;
    for(unsigned int i=0;i<_nt;++i) {
      _np += lsizes[i+1]*(lsizes[i]+1);
    }

    _params = new value_type[_np];
    memset(_params,0,sizeof(value_type)*_np);   

    if(mode==TRAIN_DEBUG)
        setupDebug();
    if(mode==TRAIN_RANDOM)
        setupRandom(); 
}

template<typename T>
T TrainingSet<T>::random_real(T mini, T maxi)
{
    return mini + (maxi-mini)*(T)rand()/(T)RAND_MAX;
}

template<typename T>
unsigned int TrainingSet<T>::random_uint(unsigned int mini, unsigned int maxi) {
  return mini + (unsigned int)floor(0.5 + (maxi-mini)*(double)rand()/(double)RAND_MAX);
}

template<typename T>
void TrainingSet<T>::setupDebug()
{
    value_type* ptr = _X_train;
    for(unsigned int i=0;i<_nx;++i) {
        (*ptr++) = (value_type)(sin(i)*10.0);
    }
   
    ptr = _y_train;
    for(unsigned int i=0;i<_ny;++i) {
        (*ptr++) = (value_type)(abs(cos(i)));
    }

    ptr = _params;
    for(unsigned int i=0;i<_np;++i) {
        (*ptr++) = (value_type)(sin(i+0.5));
    }
}

template<typename T>
void TrainingSet<T>::setupRandom()
{
    value_type* ptr = _X_train;
    for(unsigned int i=0;i<_nx;++i) {
        (*ptr++) = random_real(-10.0,10.0);
    }
   
    ptr = _y_train;
    for(unsigned int i=0;i<_ny;++i) {
        (*ptr++) = random_real(0.0,1.0);
    }

    ptr = _params;
    for(unsigned int i=0;i<_np;++i) {
        (*ptr++) = random_real(-1.0,1.0);
    }
}

};

#endif
