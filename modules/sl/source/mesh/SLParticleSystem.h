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
                     SLGLTexture* texFlipbook,
                     const SLstring& name  = "Particle system");
    
    void draw(SLSceneView* sv, SLNode* node);
    void generateVAO(SLGLVertexArray& vao);
    void regenerate();
    void generateBernsteinP(float ContP[4], float StaEnd[4]);
    void changeTexture();

    // Getters
    SLVec3f    pEPos() const { return _pEPos; }
    SLbool      acc() { return _acc; }
    SLbool      rot() { return _rot; }
    SLVec3f     accV() { return _accV; }
    SLVec3f     vRandS() { return _vRandS; }
    SLVec3f     vRandE() { return _vRandE; }
    SLCol4f     color() { return _color; }
    SLbool      tree() { return _tree; }
    SLfloat       angle() { return _angle; }
    SLint       numBranch() { return _numBranch; }
    SLbool      worldSpace() { return _worldSpace; }
    SLbool      alphaOverLF() { return _alphaOverLF; }
    SLbool      colorOverLF() { return _colorOverLF; }
    SLbool      alphaOverLFCurve() { return _alphaOverLFCurve; }
    SLbool        flipBookTexture() { return _flipBookTexture; }
    SLbool      sizeOverLF() { return _sizeOverLF; }
    SLbool      sizeRandom() { return _sizeRandom; }
    SLint      amount() { return _amount; }
    SLGLTexture* textureFirst() { return _textureFirst; }
    SLGLTexture* textureFlipbook() { return _textureFlipbook; }

    //Setters
    void pEPos(SLVec3f p) { _pEPos = p; }
    void acc(SLbool b) { _acc = b; }
    void rot(SLbool b) { _rot = b; }
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
    void color(SLCol4f c) { _color = c; }
    void tree(SLbool b) { _tree = b; }
    void angle(SLfloat f) { _angle = f; }
    void numBranch(SLint i) { _numBranch = i; }
    void worldSpace(SLbool b) { _worldSpace = b; }
    void alphaOverLF(SLbool b) { _alphaOverLF = b; }
    void colorOverLF(SLbool b) { _colorOverLF = b; }
    void alphaOverLFCurve(SLbool b) { _alphaOverLFCurve = b; }
    void flipBookTexture(SLbool b) { _flipBookTexture = b; }
    void sizeOverLF(SLbool b) { _sizeOverLF = b; }
    void sizeRandom(SLbool b) { _sizeRandom = b; }
    void amount(SLint i) { _amount = i; }
    void colorArr(SLfloat* arr) { std::copy(arr, arr + 256 * 3, _colorArr); }
    void textureFirst(SLGLTexture* t) { _textureFirst = t; }
    void textureFlipbook(SLGLTexture* t) { _textureFlipbook = t; }
    


protected:
    SLfloat     _ttl;           //!< Time to live of a particle 
    SLVec3f     _pEPos;         //!< Position of the particle emitter
    SLVec4f         _bernsteinPY = SLVec4f(2.0f, -3.0f, 0.0f, 1.0f);      //!< Vector for bezier curve (default linear function)
    SLVec3f         _accV = SLVec3f(1.0f, 1.0f, 1.0f);      //!< Vector for acceleration
    SLVec3f         _vRandS = SLVec3f(0.0f, 0.0f, 0.0f);      //!< Vector for acceleration
    SLVec3f         _vRandE = SLVec3f(1.0f, 1.0f, 1.0f);      //!< Vector for acceleration
    SLCol4f         _color    = SLCol4f(0.66f, 0.0f, 0.66f, 0.2f); //!< Color for particle
    SLfloat         _angle = 30.0f;     //!< Angle of branches (for tree fractal)
    SLint           _numBranch = 4;     //!< Number of branches (for tree fractal)
    SLGLVertexArray _vao1;      //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2;      //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:

    void  initMat(SLAssetManager* am, SLGLTexture* texC);
    float randomFloat(float a, float b);
    int randomInt(int min, int max);


    SLVVec3f V;     //!< Pointer to vertex velocity vector
    SLVfloat ST;    //!< Pointer to start time vector
    SLVVec3f InitV; //!< Pointer to vertex velocity vector
    SLVfloat R;     //!< Pointer to rotation vector
    SLVuint  TexNum; //!< Pointer to texture number vector

    int _drawBuf = 0;   //!< Boolean to switch buffer
    SLint _col     = 8;       //!< Number of texture by column
    SLint _row     = 8;       //!< Number of texture by row
    SLint _amount;      //!< Amount of a particle
    SLfloat _colorArr[256 * 3]; //!< Color values of color gradient widget
    SLGLTexture* _textureFirst;
    SLGLTexture* _textureFlipbook;

    SLbool _tree  = false;       //!< Boolean for tree fractal
    SLbool _acc  = false;       //!< Boolean for acceleration
    SLbool _rot  = true;       //!< Boolean for rotation
    SLbool _worldSpace = false; //!< Boolean for world space position
    SLbool _alphaOverLF = true; //!< Boolean for alpha over life time
    SLbool _colorOverLF = false; //!< Boolean for color over life time
    SLbool _alphaOverLFCurve = false; //!< Boolean for alpha over life time curve
    SLbool _sizeOverLF = true; //!< Boolean for size over life time
    SLbool _flipBookTexture = false; //!< Boolean for flipbook texture
    SLbool _sizeRandom = false; //!< Boolean for size over life time


};
//-----------------------------------------------------------------------------
#endif
