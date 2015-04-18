//#############################################################################
//  File:      SLPathtracer.h
//  Author:    Thomas Schneiter, Marcus Hudritsch
//  Date:      February 2014
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLPATHTRACER_H
#define SLPATHTRACER_H

#include <stdafx.h>
#include <SLRaytracer.h>

//-----------------------------------------------------------------------------
//! Classic Monte Carlo Pathtracing algorithm for real global illumination
class SLPathtracer : public SLRaytracer
{  public:           
                        SLPathtracer();
                       ~SLPathtracer(){SL_LOG("~SLPathtracer\n");;}
            
            // classic ray tracer functions
            SLbool      render      (SLSceneView* sv);
            void        renderSlices(const bool isMainThread, SLint currentSample);
            SLCol4f     trace       (SLRay* ray, SLbool em);
            SLCol4f     shade       (SLRay* ray, SLCol4f* mat);
            void        saveImage   ();

   private:
            SLfloat     _gamma;     //!< gamma correction
};
//-----------------------------------------------------------------------------
#endif
