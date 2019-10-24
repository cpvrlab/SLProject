//#############################################################################
//  File:      SLOptixRaytracer.cpp
//  Author:    Nic Dorner
//  Date:      October 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

using namespace std::placeholders;
using namespace std::chrono;

#include <SLApplication.h>
#include <SLLightRect.h>
#include <SLSceneView.h>
#include <SLOptixRaytracer.h>

//-----------------------------------------------------------------------------
SLOptixRaytracer::SLOptixRaytracer()
{
    name("myCoolRaytracer");

    _state         = rtReady;
    _maxDepth      = 5;

    // set texture properties
    _min_filter   = GL_NEAREST;
    _mag_filter   = GL_NEAREST;
    _wrap_s       = GL_CLAMP_TO_EDGE;
    _wrap_t       = GL_CLAMP_TO_EDGE;
    _resizeToPow2 = false;
}
//-----------------------------------------------------------------------------
SLOptixRaytracer::~SLOptixRaytracer()
{
    SL_LOG("Destructor      : ~SLOptixRaytracer\n");
}