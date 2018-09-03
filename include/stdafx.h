//#############################################################################
//  File:      SL/stdafx.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//  Purpose:   Include file for standard system include files, or project 
//             specific include files that are used frequently, but are changed 
//             infrequently. You must set the property C/C++/Precompiled Header
//             as "Use Precompiled Header"
//#############################################################################

#ifndef STDAFX_H
#define STDAFX_H

#define _USE_MATH_DEFINES
//#define NOMINMAX

// Include standard C++ libraries
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>
#include <queue>
#include <typeinfo>
#include <string>
#include <algorithm>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>
#include <functional>
#include <random>
#include <cstdarg>
#include <ctime>
#include <memory>
//-----------------------------------------------------------------------------
// Include standard C libraries
#include <stdio.h>               // for the old ANSI C IO functions
#include <stdlib.h>              // srand, rand
#include <float.h>               // for defines like FLT_MAX & DBL_MAX
#include <limits.h>              // for defines like UINT_MAX
#include <assert.h>              // for debug asserts
#include <time.h>                // for clock()
#include <sys/stat.h>            // for file info used in SLUtils
#include <math.h>                // for math functions
#include <string.h>              // for string functions
//-----------------------------------------------------------------------------
// Core header files used by all files
#include <SL.h>
#include <SLUtils.h>
#include <SLEnums.h>
#include <SLObject.h>
#include <SLMath.h>
#include <SLVector.h>
#include <SLVec2.h>
#include <SLVec3.h>
#include <SLVec4.h>
#include <SLMat3.h>
#include <SLMat4.h>
#include <SLQuat4.h>
#include <SLRect.h>
#include <SLPlane.h>
#include <SLGLState.h>
#include <SLFileSystem.h>
#include <SLTimer.h>
#include <SLAverageTiming.h>
//-----------------------------------------------------------------------------
#include <OrbSlam/Optimizer.h>
#include <OrbSlam/Converter.h>
#include <OrbSlam/ORBmatcher.h>
#include <OrbSlam/PnPsolver.h>
//-----------------------------------------------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#endif
