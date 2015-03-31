//#############################################################################
//  File:      SLLightRect.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTRECT_H
#define SLLIGHTRECT_H

#include <stdafx.h>
#include <SLNode.h>
#include <SLLight.h>

class SLSceneView;
class SLRay;

//-----------------------------------------------------------------------------
//! Light node class for a rectangular light source
/*!      
SLLightRect is a node that renders in OpenGL a light rectangle 
object and applies the OpenGL light settings through the SLLight class.
The light rectangle is defined with its width and height and lies initially 
centered in the x-y-plane. The light shines as a spotlight with 90 degrees 
cutoff angle towards the negative z-axis.
*/
class SLLightRect: public SLNode, public SLLight
{  public:
                        SLLightRect    (SLfloat width=1, 
                                        SLfloat height=1, 
                                        SLbool hasMesh=true);
                       ~SLLightRect    () {;}

            void        init           ();
            void        drawRec        (SLSceneView* sv);
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
            void        width          (const SLfloat w)   {_width  = w; _halfWidth =w*0.5f;}  
            void        height         (const SLfloat h)   {_height = h; _halfHeight=h*0.5f;}
            void        samples        (const SLVec2i samples);
            void        samplesXY      (const SLint x, const SLint y);
            
            // Getters
            SLfloat     width          () {return _width;}
            SLfloat     height         () {return _height;}
            SLVec3f     positionWS     () {return updateAndGetWM().translation();}
            SLVec3f     spotDirWS      () {return SLVec3f(_wm.m(8),
                                                          _wm.m(9),
                                                          _wm.m(10))*-1.0;}

   private:
            SLfloat     _width;        //!< Width of square light in x direction
            SLfloat     _height;       //!< Lenght of square light in y direction
            SLfloat     _halfWidth;    //!< Half width of square light in x dir
            SLfloat     _halfHeight;   //!< Half height of square light in y dir
            SLVec2i     _samples;      //!< Uneven NO. of samples in x and y dir
};
//-----------------------------------------------------------------------------
#endif
