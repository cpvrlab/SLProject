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
                     SLGLTexture*    texC,
                     const SLstring& name        = "Particle system",
                     SLGLTexture*    texFlipbook = nullptr);

    void draw(SLSceneView* sv, SLNode* node);
    void buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    void generate();
    void generateBernsteinPAlpha();
    void generateBernsteinPSize();
    void changeTexture();
    void notVisibleFrustrumCulling();

    // Getters
    SLVec3f      emitterPos() const { return _emitterPos; }
    SLbool       doAcc() { return _doAcc; }
    SLbool       doAccDiffDir() { return _doAccDiffDir; }
    SLbool       doRot() { return _doRot; }
    SLbool       doRotRange() { return _doRotRange; }
    SLVec3f      acc() { return _acc; }
    SLVec3f      vRandS() { return _vRandS; }
    SLVec3f      vRandE() { return _vRandE; }
    SLVec3f      scaleBox() { return _scaleBox; }
    float*       bezierControlPointAlpha() { return _bezierControlPointAlpha; }
    float*       bezierStartEndPointAlpha() { return _bezierStartEndPointAlpha; }
    float*       bezierControlPointSize() { return _bezierControlPointSize; }
    float*       bezierStartEndPointSize() { return _bezierStartEndPointSize; }
    SLCol4f      color() { return _color; }
    SLbool       doTree() { return _doTree; }
    SLbool       isGenerated() { return _isGenerated; }
    SLbool       doBlendingBrigh() { return _doBlendingBrigh; }
    SLfloat      angle() { return _angle; }
    SLfloat      accConst() { return _accConst; }
    SLint        numBranch() { return _numBranch; }
    SLbool       doWorldSpace() { return _doWorldSpace; }
    SLbool       doAlphaOverL() { return _doAlphaOverL; }
    SLbool       doColorOverLF() { return _doColorOverLF; }
    SLbool       doAlphaOverLCurve() { return _doAlphaOverLCurve; }
    SLbool       doFlipBookTexture() { return _doFlipBookTexture; }
    SLbool       doSizeOverLF() { return _doSizeOverLF; }
    SLbool       doSizeOverLFCurve() { return _doSizeOverLFCurve; }
    SLbool       doSizeRandom() { return _doSizeRandom; }
    SLbool       doColor() { return _doColor; }
    SLbool       doShape() { return _doShape; }
    SLint        amount() { return _amount; }
    SLint        billoardType() { return _billboardType; }
    SLint        shapeType() { return _shapeType; }
    SLint        col() { return _col; }
    SLint        row() { return _row; }
    SLfloat      timeToLive() { return _timeToLive; }
    SLfloat      radiusW() { return _radiusW; }
    SLfloat      radiusH() { return _radiusH; }
    SLfloat      scale() { return _scale; }
    SLfloat      angularVelocityConst() { return _angularVelocityConst; }
    SLVec2f      angularVelocityRange() { return _angularVelocityRange; }
    SLfloat      radiusSphere() { return _radiusSphere; }
    AvgFloat&    updateTime() { return _updateTime; }
    int          frameRateFB() { return _frameRateFB; }
    SLGLTexture* textureFirst() { return _textureFirst; }
    SLGLTexture* textureFlipbook() { return _textureFlipbook; }

    // Setters
    void emitterPos(SLVec3f p) { _emitterPos = p; }
    void doAcc(SLbool b) { _doAcc = b; }
    void doAccDiffDir(SLbool b) { _doAccDiffDir = b; }
    void doRot(SLbool b) { _doRot = b; }
    void doRotRange(SLbool b) { _doRotRange = b; }
    void acc(SLVec3f v) { _acc = v; }
    void acc(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _acc.x = vX;
        _acc.y = vY;
        _acc.z = vZ;
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
    void scaleBox(SLVec3f v) { _scaleBox = v; }
    void scaleBox(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _scaleBox.x = vX;
        _scaleBox.y = vY;
        _scaleBox.z = vZ;
    }
    void bezierControlPointAlpha(float arrayPoint[4])
    {
        _bezierControlPointAlpha[0]   = arrayPoint[0];
        _bezierControlPointAlpha[1]   = arrayPoint[1];
        _bezierControlPointAlpha[2]   = arrayPoint[2];
        _bezierControlPointAlpha[3]   = arrayPoint[3];
    }
    void bezierStartEndPointAlpha(float arrayPoint[4])
    {
        _bezierStartEndPointAlpha[0] = arrayPoint[0];
        _bezierStartEndPointAlpha[1] = arrayPoint[1];
        _bezierStartEndPointAlpha[2] = arrayPoint[2];
        _bezierStartEndPointAlpha[3] = arrayPoint[3];
    }
    void bezierControlPointSize(float arrayPoint[4])
    {
        _bezierControlPointSize[0]  = arrayPoint[0];
        _bezierControlPointSize[1]  = arrayPoint[1];
        _bezierControlPointSize[2]  = arrayPoint[2];
        _bezierControlPointSize[3]  = arrayPoint[3];
    }
    void bezierStartEndPointSize(float arrayPoint[4])
    {
        _bezierStartEndPointSize[0]  = arrayPoint[0];
        _bezierStartEndPointSize[1]  = arrayPoint[1];
        _bezierStartEndPointSize[2]  = arrayPoint[2];
        _bezierStartEndPointSize[3]  = arrayPoint[3];
    }
    void color(SLCol4f c) { _color = c; }
    void doTree(SLbool b) { _doTree = b; }
    void isGenerated(SLbool b) { _isGenerated = b; }
    void doBlendingBrigh(SLbool b) { _doBlendingBrigh = b; }
    void angle(SLfloat f) { _angle = f; }
    void accConst(SLfloat f) { _accConst = f; }
    void numBranch(SLint i) { _numBranch = i; }
    void doWorldSpace(SLbool b) { _doWorldSpace = b; }
    void doAlphaOverL(SLbool b) { _doAlphaOverL = b; }
    void doColorOverLF(SLbool b) { _doColorOverLF = b; }
    void doAlphaOverLCurve(SLbool b) { _doAlphaOverLCurve = b; }
    void doFlipBookTexture(SLbool b) { _doFlipBookTexture = b; }
    void doSizeOverLF(SLbool b) { _doSizeOverLF = b; }
    void doSizeOverLFCurve(SLbool b) { _doSizeOverLFCurve = b; }
    void doSizeRandom(SLbool b) { _doSizeRandom = b; }
    void doColor(SLbool b) { _doColor = b; }
    void doShape(SLbool b) { _doShape = b; }
    void amount(SLint i) { _amount = i; }
    void shapeType(SLint i) { _shapeType = i; }
    void billboardType(SLint i) { _billboardType = i; }
    void col(SLint i) { _col = i; }
    void row(SLint i) { _row = i; }
    void timeToLive(SLfloat f) { _timeToLive = f; }
    void radiusW(SLfloat f) { _radiusW = f; }
    void radiusH(SLfloat f) { _radiusH = f; }
    void scale(SLfloat f) { _scale = f; }
    void    angularVelocityConst(SLfloat f) { _angularVelocityConst = f; }
    void    angularVelocityRange(SLVec2f v) { _angularVelocityRange = v; }
    void    angularVelocityRange(SLfloat vX, SLfloat vY) { 
        _angularVelocityRange.x = vX; 
        _angularVelocityRange.y = vY; 
    }
    void radiusSphere(SLfloat f) { _radiusSphere = f; }
    void frameRateFB(int i) { _frameRateFB = i; }
    void colorArr(SLfloat* arr) { std::copy(arr, arr + 256 * 3, _colorArr); }
    void textureFirst(SLGLTexture* t) { _textureFirst = t; }
    void textureFlipbook(SLGLTexture* t) { _textureFlipbook = t; }

protected:
    SLfloat _timeToLive;                                                  //!< Time to live of a particle
    SLfloat _radiusW = 0.4f;                                       //!< width radius of a particle
    SLfloat _radiusH = 0.4f;                                       //!< height radius of a particle
    SLfloat _scale   = 1.0f;                                       //!< Scale of a particle (Scale the radius)
    SLfloat _angularVelocityConst = 30.0f;                          //!< Rotation rate const (change in angular rotation divide by change in time)
    SLVec2f _angularVelocityRange = SLVec2f(-30.0f,30.0f);         //!< Rotation rate range (change in angular rotation divide by change in time)
    SLVec3f _emitterPos;                                                //!< Position of the particle emitter
    SLVec4f _bernsteinPYAlpha = SLVec4f(2.0f, -3.0f, 0.0f, 1.0f);  //!< Vector for bezier curve (default linear function)
    float   _bezierControlPointAlpha[4] = {0.0f, 1.0f, 1.0f, 0.0f};          //!< Floats for bezier curve control points (P1: 01 ; P2: 23)   
    float   _bezierStartEndPointAlpha[4] = {0.0f, 1.0f, 1.0f, 0.0f};         //!< Floats for bezier curve end start points (Start: 01 ; End: 23)
    SLVec4f _bernsteinPYSize  = SLVec4f(-1.4f, 1.8f, 0.6f, 0.0f);  //!< Vector for bezier curve (default linear function)
    float   _bezierControlPointSize[4]  = {0.0f, 0.0f, 1.0f, 1.0f};         //!< Floats for bezier curve control points (P1: 01 ; P2: 23)
    float   _bezierStartEndPointSize[4] = {0.0f, 0.0f, 1.0f, 1.0f};         //!< Floats for bezier curve end start points (Start: 01 ; End: 23)
    SLVec3f _acc             = SLVec3f(1.0f, 1.0f, 1.0f);         //!< vec for acceleration (different direction as the velocity)
    SLfloat _accConst         = 0.0f;                              //!< Acceleration constant (same direction as the velocity)
    SLVec3f _vRandS           = SLVec3f(0.0f, 0.0f, 0.0f);         //!< vec start for random velocity
    SLVec3f _vRandE           = SLVec3f(1.0f, 1.0f, 1.0f);         //!< vec end for random velocity
    SLCol4f _color           = SLCol4f(0.66f, 0.0f, 0.66f, 0.2f); //!< Color for particle
    SLfloat _angle            = 30.0f;                             //!< Angle of branches (for doTree fractal)
    SLint   _numBranch        = 4;                                 //!< Number of branches (for doTree fractal)

    SLGLVertexArray _vao1; //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2; //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:
    void    initMat(SLAssetManager* am, SLGLTexture* texC);
    SLVec3f getPointInSphere(float radius, SLVec3f randomX);
    SLVec3f getPointInBox(SLVec3f boxScale);

    int          _drawBuf      = 0;         //!< Boolean to switch buffer
    int          _frameRateFB  = 16;        //!< Number of update of flipbook by second
    float        _lastUpdateFB = 0.0f;      //!< Last time flipbook was updated
    AvgFloat     _updateTime;               //!< Averaged time for updating in MS
    SLint        _col = 8;                  //!< Number of texture by column
    SLint        _row = 8;                  //!< Number of texture by row
    SLint        _amount;                   //!< Amount of a particle
    SLfloat      _colorArr[256 * 3];        //!< Color values of color gradient widget
    SLfloat      _startUpdateTimeMS = 0.0f; //!< Time since start for updating in MS
    SLfloat      _startUpdateTimeS  = 0.0f; //!< Time since start for updating in S
    SLfloat      _deltaTimeUpdateS  = 0.0f; //!< Time since start for updating in S
    SLfloat      _notVisibleTimeS   = 0.0f; //!< Time since start when node not visible in S
    SLfloat      _radiusSphere      = 1.0f; //!< Radius of sphere "Shape -> sphere"
    SLVec3f      _scaleBox = SLVec3f(1.0f); //!< Scale of box edges "Shape -> box"
    SLGLTexture* _textureFirst;
    SLGLTexture* _textureFlipbook;

    SLbool _isViFrustrumCulling = true; //!< Boolean to set time since node, not visible
    SLbool _isGenerated          = false;//!< Boolean to generate particle system and load it on the GPU
    SLint  _billboardType       = 0;    //!< Billboard type
    SLint  _shapeType           = 0;    //!< Shape type

    SLbool _doBlendingBrigh    = false; //!< Blending for glow/brightness on pixel with many particle
    SLbool _doTree             = false; //!< Boolean for doTree fractal
    SLbool _doAcc              = false; //!< Boolean for acceleration
    SLbool _doAccDiffDir       = false; //!< Boolean for acceleration (different direction)
    SLbool _doRot              = true;  //!< Boolean for rotation
    SLbool _doRotRange         = false;  //!< Boolean for rotation range (random value between range)
    SLbool _doColor            = true;  //!< Boolean for color
    SLbool _doShape            = false; //!< Boolean for shape feature
    SLbool _doWorldSpace       = false; //!< Boolean for world space position
    SLbool _doAlphaOverL      = true;  //!< Boolean for alpha over life time
    SLbool _doColorOverLF      = false; //!< Boolean for color over life time
    SLbool _doAlphaOverLCurve = false; //!< Boolean for alpha over life time curve
    SLbool _doSizeOverLF       = true;  //!< Boolean for size over life time
    SLbool _doSizeOverLFCurve  = false; //!< Boolean for size over life time curve
    SLbool _doFlipBookTexture  = false; //!< Boolean for flipbook texture
    SLbool _doSizeRandom       = false; //!< Boolean for size over life time
};
//-----------------------------------------------------------------------------
#endif
