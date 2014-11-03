#include <sgtcore.h>

#include "sgt/log/LogManager.h"
#include "sgt/log/LogSink.h"
#include "sgt/log/StdLogger.h"
#include <boost/date_time/posix_time/posix_time.hpp>

static sgt::RefPtr<sgt::LogManager> singleton;
	
namespace sgt {

LogManager::~LogManager() {
	//std::cout << "WARNING: destroying LogManager object!" << std::endl;
}
	
/**
Log a given piece of information to the internal sinks.
*/
void LogManager::log(int level, const std::string& trace, const std::string& msg)
{	
	if(_handler)
		_handler->handle(level,trace,msg);
	else
		doLog(level,trace,msg);
}

void LogManager::doLog(int level, const std::string& trace, const std::string& msg)
{	
	if(_sinks.empty())
		_sinks.push_back(new StdLogger("default_console_sink")); // add a console logger by default.

	// iterate on all the available sinks:
	for(SinkVector::iterator it = _sinks.begin(); it != _sinks.end(); ++it) {
		(*it)->process(level,trace,msg);
	}
}

/**
Returns the flags for a given level value.
*/
int LogManager::getLevelFlags(int level) const
{
	LevelFlagMap::const_iterator it = _levelFlags.find(level);
	if(it==_levelFlags.end())
		return _defaultLevelFlags;
	return it->second;
}

/**
Set the flags for a given level.
*/
void LogManager::setLevelFlags(int level, int flags)
{
	_levelFlags[level] = flags;
}

/**
Returns the flags for a given trace value.
*/
int LogManager::getTraceFlags(std::string trace) const
{
	TraceFlagMap::const_iterator it = _traceFlags.find(trace);
	if(it==_traceFlags.end())
		return _defaultTraceFlags;
	return it->second;
}

/**
Set the flags for a given trace.
*/
void LogManager::setTraceFlags(std::string trace, int flags)
{
	_traceFlags[trace] = flags;
}

/**
Add a new LogSink to the internal list.
*/
void LogManager::addSink(LogSink * sink)
{
	if(sink)
		_sinks.push_back(sink);

	// DEBUG_MSG("Adding sink " + sink->getName())
}

bool LogManager::removeSink(LogSink * sink) {
	if(!sink)
		return false;

	for(SinkVector::iterator it = _sinks.begin(); it != _sinks.end(); ++it) {
		if((*it) == sink) {
			// DEBUG_MSG("Removing sink " + sink->getName())
			_sinks.erase(it);
			return true;
		}
	}
	return false;
}

bool LogManager::removeSink(const std::string& name) {
	for(SinkVector::iterator it = _sinks.begin(); it != _sinks.end(); ++it) {
		if((*it)->getName() == name) {
			// DEBUG_MSG("Removing sink " + name)
			_sinks.erase(it);
			return true;
		}
	}
	return false;
}

bool LogManager::removeAllSinks() {
	// DEBUG_MSG("Removing all sinks")
	_sinks.clear();
	return true;
};

LogSink* LogManager::getSink(const std::string& name) {
	for(SinkVector::iterator it = _sinks.begin(); it != _sinks.end(); ++it) {
		if((*it)->getName() == name) {
			return it->get();
		}
	}
	return NULL;
}

/**
Return current notify level.
*/
int LogManager::getNotifyLevel()
{
	return _notifyLevel;
}

/**
Set current notify level.
*/
void LogManager::setNotifyLevel(int level)
{
	_notifyLevel = level;
}

/**
Returns current verbosity mode.
*/
bool LogManager::getVerbose()
{
	return _verbose;
}

/**
Set current verbosity mode.
*/
void LogManager::setVerbose(bool verbose)
{
	_verbose = verbose;
}

std::string LogManager::getLevelString(int level)
{
	switch(level) {
	case SEV_FATAL: return "Fatal";
	case SEV_ERROR: return "Error";
	case SEV_WARNING: return "Warning";
	case SEV_NOTICE: return "Notice";
	case SEV_INFO: return "Info";
	case SEV_DEBUG0: return "Debug";
	case SEV_DEBUG1: return "Debug 1";
	case SEV_DEBUG2: return "Debug 2";
	case SEV_DEBUG3: return "Debug 3";
	case SEV_DEBUG4: return "Debug 4";
	case SEV_DEBUG5: return "Debug 5";
	default: return "Debug X";
	}
}

std::string LogManager::getTimeStamp()
{
	boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::local_time();
	return boost::posix_time::to_iso_extended_string(currentTime);
}

void LogManager::setDefaultTraceFlags(int val)
{
	_defaultTraceFlags = val;
}

void LogManager::setDefaultLevelFlags(int val)
{
	_defaultLevelFlags = val;
}

LogManager& LogManager::instance() {
	if(!singleton) {
		singleton = new sgt::LogManager;
		// trINFO_V("LogManager","Created LogManager singleton");
	}

	return *singleton;
}

void LogManager::destroy() {
	if(singleton) {
		// destroy the singleton:
		// trINFO_V("LogManager","Destroying LogManager singleton");
		singleton = NULL;
	}
}

} // namespace sgt
