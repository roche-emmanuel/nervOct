#include "sgtcore.h"

#include "sgt/log/StdLogger.h"

namespace sgt {

/**
Output a given message on the LogSink object.
*/
void StdLogger::output(int level, const std::string& trace, const std::string& msg) {
	std::cout << msg;
}

} // namespace scLog
