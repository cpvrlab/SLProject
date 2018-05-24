//#############################################################################
//  File:      SLLightRect.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT
#include <nvwa/debug_new.h>   // memory leak detector
#endif

#include <SLApplication.h>
#include <SLScene.h>
#include <SLSceneView.h>
#include <SLLightRect.h>
#include <SLPolygon.h>
#include <SLRay.h>

extern SLfloat rnd01();

//-----------------------------------------------------------------------------
SLLightRect::SLLightRect(SLfloat w, SLfloat h, SLbool hasMesh) :
              SLNode("LightRect Node")
{  
    width(w);
    height(h);
   
    _samples.set(1,1);
   
    // make sample number even
    if (_samples.x%2==0) _samples.x++;
    if (_samples.y%2==0) _samples.y++;
   
    spotCutOffDEG(90.0f);
    spotExponent(1.0);

    if (hasMesh)
    {   SLMaterial* mat = new SLMaterial("LightRect Mesh Mat", 
                                          SLCol4f::BLACK, 
                                          SLCol4f::BLACK);
        addMesh(new SLPolygon(w, h, "LightRect Mesh", mat));
    }
    init();
}
//-----------------------------------------------------------------------------
/*! 
SLLightRect::init sets the light id, the light states & creates an 
emissive mat.
@todo properly remove this function and find a clean way to init lights in a scene
*/
void SLLightRect::init()
{  
    // Check if OpenGL lights are available
    if (SLApplication::scene->lights().size() >= SL_MAX_LIGHTS)
        SL_EXIT_MSG("Max. NO. of lights is exceeded!");

    // Add the light to the lights vector of the scene
    if (_id==-1)
    {   _id = (SLint)SLApplication::scene->lights().size();
        SLApplication::scene->lights().push_back(this);
    }
   
    // Set the OpenGL light states
    setState();
    _stateGL->numLightsUsed = (SLint)SLApplication::scene->lights().size();
   
    // Set emissive light material to the lights diffuse color
    if (_meshes.size() > 0)
        if (_meshes[0]->mat())
            _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::drawRec sets the light states and calls then the SLNode::drawRec 
method of its node.
*/
void SLLightRect::drawRec(SLSceneView* sv)
{  
    if (_id!=-1) 
    {  
        // Set the OpenGL light states
        setState();
        _stateGL->numLightsUsed = (SLint)SLApplication::scene->lights().size();
   
        // Set emissive light material to the lights diffuse color
        if (_meshes.size() > 0)
            if (_meshes[0]->mat())
                _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);
   
        // now draw the inherited object
        SLNode::drawRec(sv);
    }
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::hitRec calls the nodes intersection code.
*/
SLbool SLLightRect::hitRec(SLRay* ray)
{     
    // do not intersect shadow rays
    if (ray->type==SHADOW) return false;
   
    // only allow intersection with primary rays (no lights in reflections)
    //if (ray->type!=PRIMARY) return false;
   
    // call the intersection routine of the node   
    return SLNode::hitRec(ray);
}
//-----------------------------------------------------------------------------
//! SLLightSpot::statsRec updates the statistic parameters
void SLLightRect::statsRec(SLNodeStats &stats)
{  
    stats.numBytes += sizeof(SLLightRect);
    SLNode::statsRec(stats);
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::drawMeshes sets the light states and calls then the drawMeshes 
method of its node.
*/
void SLLightRect::drawMeshes(SLSceneView* sv)
{  
    if (_id!=-1) 
    {  
        // Set the OpenGL light states
        setState();
        _stateGL->numLightsUsed = (SLint)SLApplication::scene->lights().size();
   
        // Set emissive light material to the lights diffuse color
        if (_meshes.size() > 0)
        {   if (_meshes[0]->mat())
                _meshes[0]->mat()->emissive(_isOn ? diffuse() : SLCol4f::BLACK);   
        }
   
        // now draw the meshes of the node
        SLNode::drawMeshes(sv);
    }
}
//-----------------------------------------------------------------------------
/*!
SLLightRect::shadowTest returns 0.0 if the hit point is completely shaded and 
1.0 if it is 100% lighted. A return value inbetween is calculate by the ratio 
of the shadow rays not blocked to the total number of casted shadow rays.
*/
SLfloat SLLightRect::shadowTest(SLRay* ray, // ray of hit point
                               const SLVec3f& L, // vector from hit point to light
                               const SLfloat lightDist) // distance to light
{  
    if (_samples.x==1 && _samples.y==1)
    {  
        // define shadow ray
        SLRay shadowRay(lightDist, L, ray);
            
        SLApplication::scene->root3D()->hitRec(&shadowRay);

        return (shadowRay.length < lightDist) ? 0.0f : 1.0f;
    } 
    else // do light sampling for soft shadows
    {   SLfloat dw = (SLfloat)_width/(SLfloat)_samples.x; // width of a sample cell
        SLfloat dl = (SLfloat)_height/(SLfloat)_samples.y;// length of a sample cell
        SLint   x, y, hx=_samples.x/2, hy=_samples.y/2;
        SLint   samples = _samples.x*_samples.y;
        SLVbool isSampled;
        SLbool  importantPointsAreLighting = true;
        SLfloat lighted = 0.0f; // return value
        SLfloat invSamples = 1.0f/(SLfloat)(samples);
        SLVec3f SP; // vector hit point to sample point in world coords

        isSampled.resize(samples);

        for (y=0; y<_samples.y; ++y)
        {   for (x=0; x<_samples.x; ++x)
            {   SLint iSP = y*_samples.x + x;
                isSampled[iSP]=false;
            }
        }

        /*
        Important sample points (X) on a 7 by 5 rectangular light.
        If all of them are lighting the hit point the sample points
        in between (O) are not tested anymore.

             0   1   2   3   4   5   6         
           +---+---+---+---+---+---+---+
        0  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        1  | . | . | . | . | . | . | . |
           +---+---+---+---+---+---+---+
        2  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        3  | . | . | . | . | . | . | . |
           +---+---+---+---+---+---+---+
        4  | X | . | . | X | . | . | X |
           +---+---+---+---+---+---+---+
        */

        // Double loop for the important sample points
        for (y=-hy; y<=hy; y+=hy)
        {   for (x=-hx; x<=hx; x+=hx)
            {   SLint iSP = (y+hy)*_samples.x + x+hx;
                isSampled[iSP]=true;
            
                SP.set(updateAndGetWM().multVec(SLVec3f(x*dw, y*dl, 0)) - ray->hitPoint);
                SLfloat SPDist = SP.length();
                SP.normalize();
                SLRay shadowRay(SPDist, SP, ray);

                SLApplication::scene->root3D()->hitRec(&shadowRay);
            
                if (shadowRay.length >= SPDist-FLT_EPSILON) 
                    lighted += invSamples; // sum up the light
                else 
                    importantPointsAreLighting = false;
            }
        }

        if (importantPointsAreLighting)
            lighted = 1.0f;
        else
        {  // Double loop for the sample points in between
            for (y=-hy; y<=hy; ++y)
            {   for (x=-hx; x<=hx; ++x)
                {  SLint iSP = (y+hy)*_samples.x + x+hx;
                    if (!isSampled[iSP])
                    {   SP.set(updateAndGetWM().multVec(SLVec3f(x*dw, y*dl, 0)) - ray->hitPoint);
                        SLfloat SPDist = SP.length();
                        SP.normalize();
                        SLRay shadowRay(SPDist, SP, ray);

                        SLApplication::scene->root3D()->hitRec(&shadowRay);
                  
                        // sum up the light
                        if (shadowRay.length >= SPDist-FLT_EPSILON) 
                            lighted += invSamples;
                    }
                }
            }
        }
        return lighted;
    }
}

//-----------------------------------------------------------------------------
/*!
SLLightRect::shadowTestMC returns 0.0 if the hit point is shaded and 1.0 if it
lighted. Only one shadow sample is tested for path tracing.
*/

SLfloat SLLightRect::shadowTestMC(SLRay* ray, // ray of hit point
                                  const SLVec3f& L, // vector from hit point to light
                                  const SLfloat lightDist) // distance to light
{   SLfloat rndX = rnd01();
    SLfloat rndY = rnd01();
 
    // Sample point in object space
    SLVec3f spOS(SLVec3f(rndX*_width  - _width *0.5f,
                         rndY*_height - _height*0.5f,
                         0.0f));

    // Sample point in world space
    SLVec3f spWS(updateAndGetWM().multVec(spOS) - ray->hitPoint);
    
    SLfloat spDistWS = spWS.length();
    spWS.normalize();
    SLRay shadowRay(spDistWS, spWS, ray);

    SLApplication::scene->root3D()->hitRec(&shadowRay);

    return (shadowRay.length < spDistWS) ? 0.0f : 1.0f;
}

//-----------------------------------------------------------------------------
/*! SLLightRect::setState sets the global rendering state
*/
void SLLightRect::setState()
{  
    if (_id!=-1) 
    {   _stateGL->lightIsOn[_id]       = _isOn;
        _stateGL->lightPosWS[_id]      = positionWS();           
        _stateGL->lightSpotDirWS[_id]  = spotDirWS();           
        _stateGL->lightAmbient[_id]    = _ambient;              
        _stateGL->lightDiffuse[_id]    = _diffuse;              
        _stateGL->lightSpecular[_id]   = _specular;    
        _stateGL->lightSpotCutoff[_id] = _spotCutOffDEG;           
        _stateGL->lightSpotCosCut[_id] = _spotCosCutOffRAD;           
        _stateGL->lightSpotExp[_id]    = _spotExponent;         
        _stateGL->lightAtt[_id].x      = _kc;  
        _stateGL->lightAtt[_id].y      = _kl;    
        _stateGL->lightAtt[_id].z      = _kq; 
        _stateGL->lightDoAtt[_id]      = isAttenuated();
   }
}
//-----------------------------------------------------------------------------
void SLLightRect::samples(const SLVec2i samples)
{
    assert(samples.x%2 == 1 && samples.y%2 == 1);
    _samples = samples;
}
//-----------------------------------------------------------------------------
void SLLightRect::samplesXY(const SLint x, const SLint y)
{
    assert(x%2 == 1 && y%2 == 1);
    _samples.set(x,y);
}
//-----------------------------------------------------------------------------
