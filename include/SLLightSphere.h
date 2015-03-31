//#############################################################################
//  File:      SLLightSphere.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTSPHERE_H
#define SLLIGHTSPHERE_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLLight.h>
#include <SLSamples2D.h>

class SLSceneView;
class SLRay;

//-----------------------------------------------------------------------------
//! SLLightSphere class for a spherical light source
/*!      
SLLightSphere is a node and a light that can have a sphere mesh for its 
representation.
*/
class SLLightSphere: public SLNode, public SLLight
{  public:
                        SLLightSphere  (SLfloat radius = 0.3f, 
                                        SLbool  hasMesh = true);
                        SLLightSphere  (SLfloat posx, 
                                        SLfloat posy, 
                                        SLfloat posz,
                                        SLfloat radius = 0.3f,
                                        SLfloat ambiPower = 1.0f,
                                        SLfloat diffPower = 10.0f,
                                        SLfloat specPower = 10.0f, 
                                        SLbool  hasMesh = true);
                       ~SLLightSphere  (){;}

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
            
            // Setters
            void        samples        (SLint x, SLint y)
                                       {_samples.samples(x, y, false);}
            
            // Getters
            SLfloat     radius         () {return _radius;}
            SLint       samples        () {return _samples.samples();}
            SLVec3f     positionWS     () {return updateAndGetWM().translation();}
            SLVec3f     spotDirWS      () {return forward();}

   private:
            SLfloat     _radius;       //!< The sphere lights radius
            SLSamples2D _samples;      //!< 2D samplepoints for soft shadows
};
//-----------------------------------------------------------------------------
#endif
