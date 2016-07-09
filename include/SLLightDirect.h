//#############################################################################
//  File:      SLLightDirect.h
//  Author:    Marcus Hudritsch
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTDIRECT_H
#define SLLIGHTDIRECT_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLLight.h>
#include <SLSamples2D.h>

class SLSceneView;
class SLRay;

//-----------------------------------------------------------------------------
//! SLLightDirect class for a directional light source
/*!      
SLLightDirect is a node and a light that can have a sphere mesh with a line for 
its direction representation. 
For directional lights the position vector is in infinite distance
We use its homogeneos component w as zero as the directional light flag.
The spot direction is used in the shaders for the light direction.
*/
class SLLightDirect: public SLNode, public SLLight
{  public:
                        SLLightDirect  (SLfloat radius = 0.1f, 
                                        SLbool  hasMesh = true);
                        SLLightDirect  (SLfloat dirx, 
                                        SLfloat diry, 
                                        SLfloat dirz,
                                        SLfloat radius = 0.1f,
                                        SLfloat dirLength = 1.0f,
                                        SLfloat ambiPower = 1.0f,
                                        SLfloat diffPower = 10.0f,
                                        SLfloat specPower = 10.0f, 
                                        SLbool  hasMesh = true);
                       ~SLLightDirect  (){;}

            void        init           ();
            bool        hitRec         (SLRay* ray);
            void        statsRec       (SLNodeStats &stats);
            void        drawMeshes     (SLSceneView* sv);
            
            void        setState       ();
            SLfloat     shadowTest     (SLRay* ray,   
                                        const SLVec3f& L, 
                                        const SLfloat lightDist);
            SLfloat     shadowTestMC   (SLRay* ray,
                                        const SLVec3f& L,
                                        const SLfloat lightDist);
            
            // Getters
            SLfloat     radius         () {return _radius;}
            SLfloat     dirLength      () {return _dirLength;}
            
            // For directional lights the position vector is interpreted as a
            // direction with the homogeneous component equls zero:
            SLVec4f     positionWS     () {SLVec4f pos(updateAndGetWM().translation());
                                           pos.w = 0.0f;
                                           return pos;}

            SLVec3f     spotDirWS      () {return forward();}

   private:
            SLfloat     _radius;       //!< The sphere lights radius
            SLfloat     _dirLength;    //!< Length of direction line
            SLVec2f     _oldTouchPos1; //!< Old mouse/touch position in pixels
};
//-----------------------------------------------------------------------------
#endif
