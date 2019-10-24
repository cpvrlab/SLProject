/*!
 * \file
 * \brief Enables all the log levels down to Debug.
 */

#ifndef WAI_LOG_DEBUG
#define WAI_LOG_DEBUG

//THIS FILE IS ONLY SUPPOSED TO BE INCLUDED IN WAIHelper.h.
//OTHERWISE THE LOGGING SEVERITY PREPROCESSOR SWITCH WILL NOT WORK CORRECTLY!
#ifndef WAI_HELPER_H
#    error LogLevelDebug.h included before WAIHelper.h
#endif

#define WAI_LOG_LEVEL 1 //! failed because of multiple LogLevel includes

// disable nothing

#include "Logger.h"

#endif
