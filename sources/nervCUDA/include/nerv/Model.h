
#ifndef NERV_MODEL_H_
#define NERV_MODEL_H_

namespace nerv
{

enum ModelType
{
  MODEL_NLS_NETWORK
};

enum ModelDigestField
{
  MD_LONG_VOTE = 0,
  MD_SHORT_VOTE,
  MD_NONE_VOTE,
  MD_TOTAL_VOTE,
  MD_LONG_PROB,
  MD_SHORT_PROB,
  MD_NONE_PROB,
  MD_LONG_CONF_NUM,
  MD_SHORT_CONF_NUM,
  MD_NONE_CONF_NUM,
  MD_LONG_WEIGHT,
  MD_SHORT_WEIGHT,
  MD_NONE_WEIGHT,
  MD_LONG_CONFIDENCE,
  MD_SHORT_CONFIDENCE,
  MD_NONE_CONFIDENCE,
  MD_NUM_FIELDS
};

class NERVCUDA_EXPORT Model
{
public:
  typedef double value_type;

  struct CreationTraits
  {
    CreationTraits()
      : nl(0), lsizes(nullptr), params(nullptr),
      type(MODEL_NLS_NETWORK), mu(nullptr),
      sigma(nullptr) {}

    int type;
    unsigned int nl;
    unsigned int *lsizes;

    // parameters provided for normalization of the features:
    value_type* mu;
    value_type* sigma;

    value_type *params;
  };

  struct DigestTraits
  {
    DigestTraits()
      : input(nullptr), input_size(0)
    {
      memset((void *)field, 0, MD_NUM_FIELDS * sizeof(value_type));
    }

    value_type *input;
    unsigned int input_size;

    value_type field[MD_NUM_FIELDS];
  };

public:
  Model(const CreationTraits &traits);
  virtual ~Model();

  virtual void digest(DigestTraits &traits);

  virtual void predict(DigestTraits &traits, value_type &long_prob, value_type &short_prob, value_type &none_prob) = 0;

  inline value_type getWeight()
  {
    return _weight;
  }

protected:
  value_type _weight;
};

};

#endif
