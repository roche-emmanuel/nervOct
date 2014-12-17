
#ifndef NERV_MODEL_H_
#define NERV_MODEL_H_

namespace nerv
{

class NERVCUDA_EXPORT Model
{
public:
  typedef double value_type;

  struct CreationTraits
  {
    CreationTraits() {}
  };

  struct DigestTraits
  {
    DigestTraits() {}
  };

public:
  Model(const CreationTraits &traits);
  virtual ~Model();

  virtual void digest(DigestTraits& traits) const = 0;
};

};

#endif
