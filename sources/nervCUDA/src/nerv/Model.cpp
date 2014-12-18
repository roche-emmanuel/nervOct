#include <nervCUDA.h>
#include <nerv/Model.h>
#include <sgtCore.h>

using namespace nerv;

Model::Model(const CreationTraits &traits)
{
  _weight = 1.0; // initial model weight.
  logDEBUG("Creating Model.");
}

Model::~Model()
{
  logDEBUG("Destroying Model.");
}

template<typename T>
T conf(T base, T v2, T v3)
{
  return minimum(base - v2, base - v3);
}

void Model::digest(DigestTraits &traits)
{
  value_type lprob, sprob, nprob, cval;

  // retrieve the predictions:
  predict(traits, lprob, sprob, nprob);

  value_type mprob = maximum(lprob, sprob, nprob);
  value_type vote = mprob * _weight;

  // Add the contribution to the total vote:
  traits.field[MD_TOTAL_VOTE] += vote;

  if (mprob == lprob)
  {
    // This model is selecting the LONG pos.
    traits.field[MD_LONG_VOTE] += vote;

    // Update the confidence fields:
    cval = conf(lprob,sprob,nprob);
    CHECK(cval>=0,"Invalid confidence value");
    traits.field[MD_LONG_CONF_NUM] += cval;
    traits.field[MD_LONG_WEIGHT] += _weight;
    traits.field[MD_LONG_CONFIDENCE] = traits.field[MD_LONG_CONF_NUM]/traits.field[MD_LONG_WEIGHT];
  }
  else if (mprob == sprob)
  {
    traits.field[MD_SHORT_VOTE] += vote;

    // Update the confidence fields:
    cval = conf(sprob,lprob,nprob);
    CHECK(cval>=0,"Invalid confidence value");
    traits.field[MD_SHORT_CONF_NUM] += cval;
    traits.field[MD_SHORT_WEIGHT] += _weight;
    traits.field[MD_SHORT_CONFIDENCE] = traits.field[MD_SHORT_CONF_NUM]/traits.field[MD_SHORT_WEIGHT];
  }
  else if (mprob == nprob)
  {
		traits.field[MD_NONE_VOTE] += vote;

    // Update the confidence fields:
    cval = conf(nprob,sprob,lprob);
    CHECK(cval>=0,"Invalid confidence value");
    traits.field[MD_NONE_CONF_NUM] += cval;
    traits.field[MD_NONE_WEIGHT] += _weight;
    traits.field[MD_NONE_CONFIDENCE] = traits.field[MD_NONE_CONF_NUM]/traits.field[MD_NONE_WEIGHT];
  }
  else {
  	THROW("Invalid result for computation of max probability.")
  }

  // Update all probabilities (since the total vote changed)
  traits.field[MD_LONG_PROB] = traits.field[MD_LONG_VOTE]/traits.field[MD_TOTAL_VOTE];
  traits.field[MD_SHORT_PROB] = traits.field[MD_SHORT_VOTE]/traits.field[MD_TOTAL_VOTE];
  traits.field[MD_NONE_PROB] = traits.field[MD_NONE_VOTE]/traits.field[MD_TOTAL_VOTE];
}
