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
class SLLightDirect: public SLNode, public SLLight
{  public:
                        SLLightDirect  (SLfloat arrowLength = 0.5f, 
                                        SLbool  hasMesh = true);
                        SLLightDirect  (SLfloat posx, 
                                        SLfloat posy, 
                                        SLfloat posz,
                                        SLfloat arrowLength = 0.5f, 
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
            SLfloat     radius         () {return _arrowRadius;}
            SLfloat     dirLength      () {return _arrowLength;}
            
            // For directional lights the position vector is interpreted as a
            // direction with the homogeneous component equls zero:
            SLVec4f     positionWS     () {SLVec4f pos(updateAndGetWM().translation());
                                           pos.w = 0.0f;
                                           return pos;}

            SLVec3f     spotDirWS      () {return forwardOS();}

   private:
            SLfloat     _arrowRadius;   //!< The sphere lights radius
            SLfloat     _arrowLength;   //!< Length of direction line
};
//-----------------------------------------------------------------------------
#endif
