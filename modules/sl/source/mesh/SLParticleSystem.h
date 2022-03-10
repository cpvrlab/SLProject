//#############################################################################
//  File:      SLParticleSystem.h
//  Date:      February 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLParticleSystem_H
#define SLParticleSystem_H

#include <SLMesh.h>
#include <SLGLTexture.h>
#include <SLRnd3f.h>

//-----------------------------------------------------------------------------
//! SLParticleSystem creates
/*! The SLParticleSystem mesh object of witch the vertices are drawn as points.
 */
class SLParticleSystem : public SLMesh
{
public:
    //! Ctor for a given vector of points
    SLParticleSystem(SLAssetManager* assetMgr,
                     const int  amount,
                     const SLVec3f&  particleEmiPos,
                     const SLVec3f&  velocityRandomStart,
                     const SLVec3f&  velocityRandomEnd,
                     const SLfloat&  timeToLive,
                     const SLstring& shaderPath,
                     SLGLTexture* texC,
                     const SLstring& name  = "Particle system");
    
    void draw(SLSceneView* sv, SLNode* node);
    void generateVAO(SLGLVertexArray& vao);

    // Getters
    SLVec3f    pEPos() const { return _pEPos; }
    SLbool      worldSpace() { return _worldSpace; }
    SLbool      alphaOverLF() { return _alphaOverLF; }
    SLbool      sizeOverLF() { return _sizeOverLF; }
    SLbool      sizeRandom() { return _sizeRandom; }

    //Setters
    void pEPos(SLVec3f p) { _pEPos = p; }
    void worldSpace(SLbool b) { _worldSpace = b; }
    void alphaOverLF(SLbool b) { _alphaOverLF = b; }
    void sizeOverLF(SLbool b) { _sizeOverLF = b; }
    void sizeRandom(SLbool b) { _sizeRandom = b; }


protected:
    SLfloat     _ttl;           //!< Time to live of a particle
    SLVec3f     _pEPos;         //!< Position of the particle emitter
    SLGLVertexArray _vao1;      //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2;      //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:

    void  initMat(SLAssetManager* am, SLGLTexture* texC, const SLstring& shaderPath);
    float randomFloat(float a, float b);


    SLVVec3f V;     //!< Pointer to vertex velocity vector
    SLVfloat ST;    //!< Pointer to start time vector
    SLVVec3f InitV; //!< Pointer to vertex velocity vector
    SLVfloat R;     //!< Pointer to rotation vector

    int _drawBuf = 0;   //!< Boolean to switch buffer
    int _amount;        //!< Amount of a particle

    SLbool _worldSpace = false; //!< Boolean for world space position
    SLbool _alphaOverLF = true; //!< Boolean for alpha over life time
    SLbool _sizeOverLF = true; //!< Boolean for size over life time
    SLbool _sizeRandom = false; //!< Boolean for size over life time

};
//-----------------------------------------------------------------------------
#endif
