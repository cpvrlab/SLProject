//#############################################################################
//  File:      SLVR.h
//  Author:    Marino von Wattenwyl
//  Date:      August 2021
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPROJECT_SLVR_H
#define SLPROJECT_SLVR_H

#include <iostream>

//! Utility macros for VR components
//-----------------------------------------------------------------------------
#define VR_DEBUG
//-----------------------------------------------------------------------------
#ifdef VR_DEBUG
#    define VR_LOG(message) std::cout << "SLVR: " << message << std::endl;
#else
#    define VR_LOG(message)
#endif
//-----------------------------------------------------------------------------
#define VR_WARNING(message) std::cerr << "SLVR Warning: " << message << std::endl;
#define VR_ERROR(message) \
    std::cerr << "SLVR Error: " << message << std::endl; \
    exit(-1);
//-----------------------------------------------------------------------------
#endif // SLPROJECT_SLVR_H
