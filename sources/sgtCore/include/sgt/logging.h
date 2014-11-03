#ifndef SGTCORE_LOGGING_
#define SGTCORE_LOGGING_

#include "sgt/log/LogManager.h"
#include "sgt/log/LogRecord.h"

// This file contains the macro definitions for the internal log system.

#define LOG(level,msg) \
	if(level > sgt::LogManager::instance().getNotifyLevel()) ; \
	else sgt::LogRecord().GetStream(level,__FILE__,__LINE__,"") << msg ;

#define TRACE(level,trace,msg) \
	if(level > sgt::LogManager::instance().getNotifyLevel()) ; \
	else sgt::LogRecord().GetStream(level,__FILE__,__LINE__,trace) << msg;

#define LOG_V(level,msg) \
	if(level > sgt::LogManager::instance().getNotifyLevel() || !sgt::LogManager::instance().getVerbose()) ; \
	else sgt::LogRecord().GetStream(level,__FILE__,__LINE__,"") << msg;

#define TRACE_V(level,trace,msg) \
	if(level > sgt::LogManager::instance().getNotifyLevel() || !sgt::LogManager::instance().getVerbose()) ; \
	else sgt::LogRecord().GetStream(level,__FILE__,__LINE__,trace) << msg;

#ifdef SGT_LOG_RELEASE_MODE
	// Do not define the debug log targets:
	#define LOG_D(level,msg)
	#define LOG_V_D(level,msg)
	#define TRACE_D(level,trace,msg)
	#define TRACE_V_D(level,trace,msg)
#else
	#define LOG_D(level,msg) LOG(level,msg)
	#define LOG_V_D(level,msg) LOG_V(level,msg)
	#define TRACE_D(level,trace,msg) TRACE(level,trace,msg)
	#define TRACE_V_D(level,trace,msg) TRACE_V(level,trace,msg)
#endif
	
#define logFATAL(msg) LOG(sgt::LogManager::SEV_FATAL,msg)
#define logERROR(msg) LOG(sgt::LogManager::SEV_ERROR,msg)
#define logWARN(msg) LOG(sgt::LogManager::SEV_WARNING,msg)
#define logINFO(msg) LOG(sgt::LogManager::SEV_INFO,msg)
#define logNOTICE(msg) LOG(sgt::LogManager::SEV_NOTICE,msg)
#define logDEBUG(msg) LOG_D(sgt::LogManager::SEV_DEBUG0,msg)
#define logDEBUG0(msg) LOG_D(sgt::LogManager::SEV_DEBUG0,msg)
#define logDEBUG1(msg) LOG_D(sgt::LogManager::SEV_DEBUG1,msg)
#define logDEBUG2(msg) LOG_D(sgt::LogManager::SEV_DEBUG2,msg)
#define logDEBUG3(msg) LOG_D(sgt::LogManager::SEV_DEBUG3,msg)
#define logDEBUG4(msg) LOG_D(sgt::LogManager::SEV_DEBUG4,msg)
#define logDEBUG5(msg) LOG_D(sgt::LogManager::SEV_DEBUG5,msg)

#define logWARN_V(msg) LOG_V(sgt::LogManager::SEV_WARNING,msg)
#define logINFO_V(msg) LOG_V(sgt::LogManager::SEV_INFO,msg)
#define logNOTICE_V(msg) LOG_V(sgt::LogManager::SEV_NOTICE,msg)
#define logDEBUG_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG0,msg)
#define logDEBUG0_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG0,msg)
#define logDEBUG1_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG1,msg)
#define logDEBUG2_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG2,msg)
#define logDEBUG3_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG3,msg)
#define logDEBUG4_V(msg) LOG_V_D(sgt::LogManager::SEV_DEBUG4,msg)
#define logDEBUG5_V(msg) LOG_V(sgt::LogManager::SEV_DEBUG5,msg)

#define trFATAL(trace,msg) TRACE(sgt::LogManager::SEV_FATAL,trace,msg)
#define trERROR(trace,msg) TRACE(sgt::LogManager::SEV_ERROR,trace,msg)
#define trWARN(trace,msg) TRACE(sgt::LogManager::SEV_WARNING,trace,msg)
#define trINFO(trace,msg) TRACE(sgt::LogManager::SEV_INFO,trace,msg)
#define trNOTICE(trace,msg) TRACE(sgt::LogManager::SEV_NOTICE,trace,msg)
#define trDEBUG(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG0,trace,msg)
#define trDEBUG0(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG0,trace,msg)
#define trDEBUG1(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG1,trace,msg)
#define trDEBUG2(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG2,trace,msg)
#define trDEBUG3(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG3,trace,msg)
#define trDEBUG4(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG4,trace,msg)
#define trDEBUG5(trace,msg) TRACE_D(sgt::LogManager::SEV_DEBUG5,trace,msg)

#define trWARN_V(trace,msg) TRACE_V(sgt::LogManager::SEV_WARNING,trace,msg)
#define trINFO_V(trace,msg) TRACE_V(sgt::LogManager::SEV_INFO,trace,msg)
#define trNOTICE_V(trace,msg) TRACE_V(sgt::LogManager::SEV_NOTICE,trace,msg)
#define trDEBUG_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG0,trace,msg)
#define trDEBUG0_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG0,trace,msg)
#define trDEBUG1_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG1,trace,msg)
#define trDEBUG2_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG2,trace,msg)
#define trDEBUG3_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG3,trace,msg)
#define trDEBUG4_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG4,trace,msg)
#define trDEBUG5_V(trace,msg) TRACE_V_D(sgt::LogManager::SEV_DEBUG5,trace,msg)

#endif
