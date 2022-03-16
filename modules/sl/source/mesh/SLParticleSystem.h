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
                     const SLint     amount,
                     const SLVec3f&  particleEmiPos,
                     const SLVec3f&  velocityRandomStart,
                     const SLVec3f&  velocityRandomEnd,
                     const SLfloat&  timeToLive,
                     SLGLTexture* texC,
                     const SLstring& name  = "Particle system");
    
    void draw(SLSceneView* sv, SLNode* node);
    void generateVAO(SLGLVertexArray& vao);
    void regenerate();

    // Getters
    SLVec3f    pEPos() const { return _pEPos; }
    SLbool      acc() { return _acc; }
    SLVec3f     accV() { return _accV; }
    SLVec3f     vRandS() { return _vRandS; }
    SLVec3f     vRandE() { return _vRandE; }
    SLbool      worldSpace() { return _worldSpace; }
    SLbool      alphaOverLF() { return _alphaOverLF; }
    SLbool      sizeOverLF() { return _sizeOverLF; }
    SLbool      sizeRandom() { return _sizeRandom; }
    SLint      amount() { return _amount; }

    //Setters
    void pEPos(SLVec3f p) { _pEPos = p; }
    void acc(SLbool b) { _acc = b; }
    void accV(SLVec3f v) { _accV = v; }
    void accV(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _accV.x = vX;
        _accV.y = vY;
        _accV.z = vZ;
    }
    void vRandS(SLVec3f v) { _vRandS = v; }
    void vRandS(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _vRandS.x = vX;
        _vRandS.y = vY;
        _vRandS.z = vZ;
    }
    void vRandE(SLVec3f v) { _vRandE = v; }
    void vRandE(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _vRandE.x = vX;
        _vRandE.y = vY;
        _vRandE.z = vZ;
    }
    void worldSpace(SLbool b) { _worldSpace = b; }
    void alphaOverLF(SLbool b) { _alphaOverLF = b; }
    void sizeOverLF(SLbool b) { _sizeOverLF = b; }
    void sizeRandom(SLbool b) { _sizeRandom = b; }
    void amount(SLint i) { _amount = i; }


protected:
    SLfloat     _ttl;           //!< Time to live of a particle
    SLVec3f     _pEPos;         //!< Position of the particle emitter
    SLVec3f         _accV = SLVec3f(1.0f, 1.0f, 1.0f);      //!< Vector for acceleration
    SLVec3f         _vRandS = SLVec3f(0.0f, 0.0f, 0.0f);      //!< Vector for acceleration
    SLVec3f         _vRandE = SLVec3f(1.0f, 1.0f, 1.0f);      //!< Vector for acceleration
    SLGLVertexArray _vao1;      //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2;      //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:

    void  initMat(SLAssetManager* am, SLGLTexture* texC);
    float randomFloat(float a, float b);


    SLVVec3f V;     //!< Pointer to vertex velocity vector
    SLVfloat ST;    //!< Pointer to start time vector
    SLVVec3f InitV; //!< Pointer to vertex velocity vector
    SLVfloat R;     //!< Pointer to rotation vector

    int _drawBuf = 0;   //!< Boolean to switch buffer
    SLint _amount;      //!< Amount of a particle

    SLbool _acc  = false;       //!< Boolean for acceleration
    SLbool _worldSpace = false; //!< Boolean for world space position
    SLbool _alphaOverLF = true; //!< Boolean for alpha over life time
    SLbool _sizeOverLF = true; //!< Boolean for size over life time
    SLbool _sizeRandom = false; //!< Boolean for size over life time


};
//-----------------------------------------------------------------------------
#endif
