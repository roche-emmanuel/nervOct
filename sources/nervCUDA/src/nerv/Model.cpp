#include <nervCUDA.h>
#include <nerv/Model.h>
#include <sgtCore.h>

using namespace nerv;

Model::Model(const CreationTraits &traits)
{
  logDEBUG("Creating Model.");
}

Model::~Model()
{
  logDEBUG("Destroying Model.");
}

// void Model::digest(DigestTraits &traits) const
// {

// }
