//#############################################################################
//  File:      SLLightSpot.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTSPHERE_H
#define SLLIGHTSPHERE_H

#include <SLNode.h>
#include <SLLight.h>
#include <SLSamples2D.h>

class SLSceneView;
class SLRay;

//-----------------------------------------------------------------------------
//! SLLightSpot class for a spot light source
/*!      
SLLightSpot is a node and a light that can have a spot mesh for its 
representation.
If a light node is added to the scene it stays fix in the scene.\n
If a light node is added to the camera it moves with the camera.\n
See the scene examples for Per-Vertex-Blinn or Per-Pixel-Blinn lighting where
all light node types are used. \n
All light nodes inherited from SLLight work automatically together with the
following shaders: \n
  - PerVrtBlinn.vert, PerVrtBlinn.frag \n
  - PerVrtBlinnTex.vert, PerVrtBlinnTex.frag \n
  - PerPixBlinn.vert, PerPixBlinn.frag \n
  - PerPixBlinnTex.vert, PerPixBlinnTex.frag \n
*/
class SLLightSpot: public SLNode, public SLLight
{   public:
                        SLLightSpot (SLfloat radius = 0.3f,
                                     SLfloat spotAngleDEG = 180.0f, 
                                     SLbool  hasMesh = true);
                        SLLightSpot (SLfloat posx, 
                                     SLfloat posy, 
                                     SLfloat posz,
                                     SLfloat radius = 0.3f,
                                     SLfloat spotAngleDEG = 180.0f, 
                                     SLfloat ambiPower = 1.0f,
                                     SLfloat diffPower = 10.0f,
                                     SLfloat specPower = 10.0f, 
                                     SLbool  hasMesh = true);
                       ~SLLightSpot (){;}

            void        init        ();
            bool        hitRec      (SLRay* ray);
            void        statsRec    (SLNodeStats &stats);
            void        drawMeshes  (SLSceneView* sv);
            
            void        setState    ();
            SLfloat     shadowTest  (SLRay* ray,   
                                     const SLVec3f& L, 
                                     const SLfloat lightDist);
            SLfloat     shadowTestMC(SLRay* ray,
                                     const SLVec3f& L,
                                     const SLfloat lightDist);
            
            // Setters
            void        samples     (SLint x, SLint y)
                                    {_samples.samples(x, y, false);}
            
            // Getters
            SLfloat     radius      () {return _radius;}
            SLint       samples     () {return _samples.samples();}
            SLVec4f     positionWS  () {return translationWS();}
            SLVec3f     spotDirWS   () {return forwardWS();}

    private:
            SLfloat     _radius;    //!< The sphere lights radius
            SLSamples2D _samples;   //!< 2D samplepoints for soft shadows
};
//-----------------------------------------------------------------------------
#endif
