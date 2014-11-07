#include <octave/oct.h>

DEFUN_DLD (train_bp, args, nargout,
           "train_bp function using nervMBP module")
{
  int nargin = args.length ();

  octave_stdout << "train_bp has "
                << nargin << " input arguments and "
                << nargout << " output arguments.\n";


  return octave_value_list ();
}
