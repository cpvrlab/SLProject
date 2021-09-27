/*!
 * \file    peak_ipl_dll_defines.h
 *
 * \author  IDS Imaging Development Systems GmbH
 * \date    2019-05-01
 * \since   1.0
 *
 * Copyright (c) 2019, IDS Imaging Development Systems GmbH. All rights reserved.
 */

#pragma once

#ifdef PEAK_IPL_DYNAMIC_LOADING
#    define PEAK_IPL_C_ABI_PREFIX peak::ipl::dynamic::DynamicLoader::
#else
#    define PEAK_IPL_C_ABI_PREFIX // we could also set ::
#endif

#if defined(_WIN32)
#    ifdef PEAK_IPL_STATIC
#        define PEAK_IPL_PUBLIC
#    else
#        ifdef PEAK_IPL_EXPORT
#            define PEAK_IPL_PUBLIC __declspec(dllexport)
#        else
#            define PEAK_IPL_PUBLIC __declspec(dllimport)
#        endif
#    endif
#    if defined(_M_IX86) || defined(__i386__)
#        define PEAK_IPL_CALL_CONV __cdecl
#    else
#        define PEAK_IPL_CALL_CONV
#    endif
#elif defined(__linux__)
#    define PEAK_IPL_PUBLIC
#    if defined(__i386__)
#        define PEAK_IPL_CALL_CONV __attribute__((cdecl))
#    else
#        define PEAK_IPL_CALL_CONV
#    endif
#else
#    error Platform is not supported yet!
#endif
