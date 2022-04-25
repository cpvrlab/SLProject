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
#include <Averaged.h>
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
                     const SLstring& name  = "Particle system",
                     SLGLTexture* texFlipbook = nullptr
                     );
    
    void draw(SLSceneView* sv, SLNode* node);
    void buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    void generateVAO(SLGLVertexArray& vao);
    void regenerate();
    void generateBernsteinPAlpha(float ContP[4], float StaEnd[4]);
    void generateBernsteinPSize(float ContP[4], float StaEnd[4]);
    void changeTexture();
    void notVisibleFrustrumCulling();

    // Getters
    SLVec3f    pEPos() const { return _pEPos; }
    SLbool      acc() { return _acc; }
    SLbool        accDiffDir() { return _accDiffDir; }
    SLbool      rot() { return _rot; }
    SLVec3f     accV() { return _accV; }
    SLVec3f     vRandS() { return _vRandS; }
    SLVec3f     vRandE() { return _vRandE; }
    SLCol4f     colorV() { return _colorV; }
    SLbool      tree() { return _tree; }
    SLbool        blendingBrigh() { return _blendingBrigh; }
    SLfloat       angle() { return _angle; }
    SLfloat       accConst() { return _accConst; }
    SLint       numBranch() { return _numBranch; }
    SLbool      worldSpace() { return _worldSpace; }
    SLbool      alphaOverLF() { return _alphaOverLF; }
    SLbool      colorOverLF() { return _colorOverLF; }
    SLbool      alphaOverLFCurve() { return _alphaOverLFCurve; }
    SLbool        flipBookTexture() { return _flipBookTexture; }
    SLbool      sizeOverLF() { return _sizeOverLF; }
    SLbool        sizeOverLFCurve() { return _sizeOverLFCurve; }
    SLbool      sizeRandom() { return _sizeRandom; }
    SLbool        color() { return _color; }
    SLbool        shape() { return _shape; }
    SLint      amount() { return _amount; }
    SLint         billoardType() { return _billoardType; }
    SLint         shapeType() { return _shapeType; }
    SLint      col() { return _col; }
    SLint      row() { return _row; }
    SLfloat       ttl() { return _ttl; }
    SLfloat       radiusW() { return _radiusW; }
    SLfloat       radiusH() { return _radiusH; }
    SLfloat       scale() { return _scale; }
    SLfloat       radiusSphere() { return _radiusSphere; }
    AvgFloat&     updateTime() { return _updateTime; }
    int       frameRateFB() { return _frameRateFB; }
    SLGLTexture* textureFirst() { return _textureFirst; }
    SLGLTexture* textureFlipbook() { return _textureFlipbook; }

    //Setters
    void pEPos(SLVec3f p) { _pEPos = p; }
    void acc(SLbool b) { _acc = b; }
    void accDiffDir(SLbool b) { _accDiffDir = b; }
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
    void colorV(SLCol4f c) { _colorV = c; }
    void tree(SLbool b) { _tree = b; }
    void blendingBrigh(SLbool b) { _blendingBrigh = b; }
    void angle(SLfloat f) { _angle = f; }
    void accConst(SLfloat f) { _accConst = f; }
    void numBranch(SLint i) { _numBranch = i; }
    void worldSpace(SLbool b) { _worldSpace = b; }
    void alphaOverLF(SLbool b) { _alphaOverLF = b; }
    void colorOverLF(SLbool b) { _colorOverLF = b; }
    void alphaOverLFCurve(SLbool b) { _alphaOverLFCurve = b; }
    void flipBookTexture(SLbool b) { _flipBookTexture = b; }
    void sizeOverLF(SLbool b) { _sizeOverLF = b; }
    void sizeOverLFCurve(SLbool b) { _sizeOverLFCurve = b; }
    void sizeRandom(SLbool b) { _sizeRandom = b; }
    void color(SLbool b) { _color = b; }
    void shape(SLbool b) { _shape = b; }
    void amount(SLint i) { _amount = i; }
    void shapeType(SLint i) { _shapeType = i; }
    void billoardType(SLint i) { _billoardType = i; }
    void col(SLint i) { _col = i; }
    void row(SLint i) { _row = i; }
    void ttl(SLfloat f) { _ttl = f; }
    void radiusW(SLfloat f) { _radiusW = f; }
    void radiusH(SLfloat f) { _radiusH = f; }
    void scale(SLfloat f) { _scale = f; }
    void radiusSphere(SLfloat f) { _radiusSphere = f; }
    void frameRateFB(int i) { _frameRateFB = i; }
    void colorArr(SLfloat* arr) { std::copy(arr, arr + 256 * 3, _colorArr); }
    void textureFirst(SLGLTexture* t) { _textureFirst = t; }
    void textureFlipbook(SLGLTexture* t) { _textureFlipbook = t; }
    


protected:
    SLfloat     _ttl;           //!< Time to live of a particle 
    SLfloat     _radiusW = 0.4f;           //!< width radius of a particle 
    SLfloat     _radiusH = 0.4f;           //!< height radius of a particle 
    SLfloat     _scale = 1.0f;           //!< Scale of a particle (Scale the radius)
    SLVec3f     _pEPos;         //!< Position of the particle emitter
    SLVec4f         _bernsteinPYAlpha = SLVec4f(2.0f, -3.0f, 0.0f, 1.0f); //!< Vector for bezier curve (default linear function)
    SLVec4f         _bernsteinPYSize = SLVec4f(-1.4f, 1.8f, 0.6f, 0.0f);      //!< Vector for bezier curve (default linear function)
    SLVec3f         _accV = SLVec3f(1.0f, 1.0f, 1.0f);      //!< vec for acceleration (different direction as the velocity)
    SLfloat         _accConst = 0.0f;           //!< Acceleration constant (same direction as the velocity)
    SLVec3f         _vRandS = SLVec3f(0.0f, 0.0f, 0.0f);      //!< vec start for random velocity
    SLVec3f         _vRandE = SLVec3f(1.0f, 1.0f, 1.0f);      //!< vec end for random velocity
    SLCol4f         _colorV    = SLCol4f(0.66f, 0.0f, 0.66f, 0.2f); //!< Color for particle

    SLfloat         _angle = 30.0f;     //!< Angle of branches (for tree fractal)
    SLint           _numBranch = 4;     //!< Number of branches (for tree fractal)

    SLGLVertexArray _vao1;      //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2;      //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:

    void  initMat(SLAssetManager* am, SLGLTexture* texC);
    float randomFloat(float a, float b);
    int randomInt(int min, int max);
    SLVec3f getPointSphere(float radius);


    SLVVec3f V;         //!< Vector for particle velocity
    SLVfloat ST;        //!< Vector for particle start time 
    SLVVec3f InitV;     //!< Vector for particle inital velocity
    SLVfloat R;         //!< Vector for particle rotation
    SLVuint  TexNum;    //!< Vector for particle texture number
    SLVVec3f InitP;     //!< Vector for particle inital position

    int _drawBuf = 0;   //!< Boolean to switch buffer
    int _frameRateFB = 16;       //!< Number of update of flipbook by second
    float _lastUpdateFB = 0.0f;    //!< Last time flipbook was updated
    AvgFloat    _updateTime;     //!< Averaged time for updating in MS
    SLint _col     = 8;       //!< Number of texture by column
    SLint _row     = 8;       //!< Number of texture by row
    SLint _amount;      //!< Amount of a particle
    SLfloat _colorArr[256 * 3]; //!< Color values of color gradient widget
    SLfloat      _startUpdateTimeMS = 0.0f; //!< Time since start for updating in MS
    SLfloat      _startUpdateTimeS = 0.0f; //!< Time since start for updating in S
    SLfloat      _deltaTimeUpdateS   = 0.0f; //!< Time since start for updating in S
    SLfloat      _notVisibleTimeS   = 0.0f; //!< Time since start when node not visible in S

    SLfloat      _radiusSphere   = 1.0f; //!< Radius of sphere "Shape -> sphere"

    SLGLTexture* _textureFirst;
    SLGLTexture* _textureFlipbook;

    SLbool _isViFrustrumCulling = true; //!< Boolean to set time since node, not visible

    SLint  _billoardType = 0;                   //!< Billboard type
    SLint  _shapeType = 0;                   //!< Shape type
    SLbool _blendingBrigh  = false;       //!< Blending for glow/brightness on pixel with many particle
    SLbool _tree  = false;       //!< Boolean for tree fractal
    SLbool _acc  = false;       //!< Boolean for acceleration
    SLbool _accDiffDir = false;       //!< Boolean for acceleration (different direction)
    SLbool _rot  = true;       //!< Boolean for rotation
    SLbool _color  = true;       //!< Boolean for color
    SLbool _shape  = false;       //!< Boolean for shape feature
    SLbool _worldSpace = false; //!< Boolean for world space position
    SLbool _alphaOverLF = true; //!< Boolean for alpha over life time
    SLbool _colorOverLF = false; //!< Boolean for color over life time
    SLbool _alphaOverLFCurve = false; //!< Boolean for alpha over life time curve
    SLbool _sizeOverLF = true; //!< Boolean for size over life time
    SLbool _sizeOverLFCurve = false; //!< Boolean for size over life time curve
    SLbool _flipBookTexture = false; //!< Boolean for flipbook texture
    SLbool _sizeRandom = false; //!< Boolean for size over life time


};
//-----------------------------------------------------------------------------
#endif
