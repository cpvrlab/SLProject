//#############################################################################
//  File:      SLParticleSystem.h
//  Date:      February 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Affolter Marc in his bachelor thesis in spring 2022
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLParticleSystem_H
#define SLParticleSystem_H

#include <SLMesh.h>
#include <SLGLTexture.h>
#include <Averaged.h>
#include <SLRnd3f.h>
#include <SLTexColorLUT.h>

//-----------------------------------------------------------------------------
//! SLParticleSystem creates a particle meshes from a point primitive buffer.
/*! The SLParticleSystem mesh object of which the vertices are drawn as points.
 * An OpenGL transform feedback buffer is used to update the particle positions
 * on the GPU and a geometry shader is used the create two triangles per
 * particle vertex and orient them as a billboard to the viewer. Geometry
 * shaders are only supported under OpenGL  >= 4.0 and OpenGL ES >= 3.2. This is
 * the case on most desktop systems and on Android SDK > 24 but not on iOS which
 * has only OpenGL ES 3.0.\n.
 * The particle system supports many options of which many can be turned on the
 * do* methods. All options can also be modified in the UI when the mesh is
 * selected. See the different demo scenes in the app_demo_slproject under the
 * demo scene group Particle Systems.
 */
class SLParticleSystem : public SLMesh
{
public:
    SLParticleSystem(SLAssetManager* assetMgr,
                     const SLint     amount,
                     const SLVec3f&  velocityRandomStart,
                     const SLVec3f&  velocityRandomEnd,
                     const SLfloat&  timeToLive,
                     SLGLTexture*    texC,
                     const SLstring& name        = "Particle system",
                     SLGLTexture*    texFlipbook = nullptr);

    void draw(SLSceneView* sv, SLNode* node);
    void deleteData();
    void deleteDataGpu();
    void buildAABB(SLAABBox& aabb, const SLMat4f& wmNode);
    void generate();
    void generateBernsteinPAlpha();
    void generateBernsteinPSize();
    void changeTexture();
    void setNotVisibleInFrustum();
    void pauseOrResume();

    // Getters
    SLVec3f           acceleration() { return _acceleration; }
    SLfloat           accelerationConst() { return _accelerationConst; }
    SLint             amount() { return _amount; }
    SLfloat           angularVelocityConst() { return _angularVelocityConst; }
    SLVec2f           angularVelocityRange() { return _angularVelocityRange; }
    float*            bezierControlPointAlpha() { return _bezierControlPointAlpha; }
    float*            bezierStartEndPointAlpha() { return _bezierStartEndPointAlpha; }
    float*            bezierControlPointSize() { return _bezierControlPointSize; }
    float*            bezierStartEndPointSize() { return _bezierStartEndPointSize; }
    SLBillboardType   billboardType() { return _billboardType; }
    SLCol4f           color() { return _color; }
    SLVColorLUTPoint& colorPoints() { return _colorPoints; }
    SLbool            doDirectionSpeed() { return _doDirectionSpeed; }
    SLbool            doSpeedRange() { return _doSpeedRange; }
    SLbool            doAcc() { return _doAcceleration; }
    SLbool            doAccDiffDir() { return _doAccDiffDir; }
    SLbool            doAlphaOverLT() { return _doAlphaOverLT; }
    SLbool            doAlphaOverLTCurve() { return _doAlphaOverLTCurve; }
    SLbool            doBlendBrightness() { return _doBlendBrightness; }
    SLbool            doCounterGap() { return _doCounterGap; }
    SLbool            doColor() { return _doColor; }
    SLbool            doColorOverLT() { return _doColorOverLT; }
    SLbool            doGravity() { return _doGravity; }
    SLbool            doFlipBookTexture() { return _doFlipBookTexture; }
    SLbool            doRotation() { return _doRotation; }
    SLbool            doRotRange() { return _doRotRange; }
    SLbool            doSizeOverLT() { return _doSizeOverLT; }
    SLbool            doSizeOverLTCurve() { return _doSizeOverLTCurve; }
    SLbool            doShape() { return _doShape; }
    SLbool            doShapeSurface() { return _doShapeSurface; }
    SLbool            doShapeOverride() { return _doShapeOverride; }
    SLbool            doShapeSpawnBase() { return _doShapeSpawnBase; }
    SLbool            doWorldSpace() { return _doWorldSpace; }
    SLVec3f           direction() { return _direction; }
    AvgFloat&         drawTime() { return _drawTime; }
    SLVec3f           emitterPos() const { return _emitterPos; }
    SLVec3f           gravity() { return _gravity; }
    SLint             flipbookColumns() { return _flipbookColumns; }
    SLint             flipbookRows() { return _flipbookRows; }
    int               frameRateFB() { return _flipbookFPS; }
    SLbool            isGenerated() { return _isGenerated; }
    SLbool            isPaused() { return _isPaused; }
    SLfloat           radiusW() { return _radiusW; }
    SLfloat           radiusH() { return _radiusH; }
    SLfloat           scale() { return _scale; }
    SLShapeType       shapeType() { return _shapeType; }
    SLfloat           shapeAngle() { return _shapeAngle; }
    SLfloat           shapeHeight() { return _shapeHeight; }
    SLfloat           shapeRadius() { return _shapeRadius; }
    SLVec3f           shapeScale() { return _shapeScale; }
    SLfloat           shapeWidth() { return _shapeWidth; }
    SLfloat           speed() { return _speed; }
    SLVec2f           speedRange() { return _speedRange; }
    SLGLTexture*      textureFirst() { return _textureFirst; }
    SLGLTexture*      textureFlipbook() { return _textureFlipbook; }
    SLfloat           timeToLive() { return _timeToLive; }
    AvgFloat&         updateTime() { return _updateTime; }
    SLint             velocityType() { return _velocityType; }
    SLVec3f           velocityConst() { return _velocityConst; }
    SLVec3f           velocityRndMin() { return _velocityRndMin; }
    SLVec3f           velocityRndMax() { return _velocityRndMax; }

    // Setters
    void amount(SLint i) { _amount = i; }
    void accConst(SLfloat f) { _accelerationConst = f; }
    void acceleration(SLVec3f v) { _acceleration = v; }
    void acceleration(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _acceleration.x = vX;
        _acceleration.y = vY;
        _acceleration.z = vZ;
    }
    void angularVelocityConst(SLfloat f) { _angularVelocityConst = f; }
    void angularVelocityRange(SLVec2f v) { _angularVelocityRange = v; }
    void angularVelocityRange(SLfloat vX, SLfloat vY)
    {
        _angularVelocityRange.x = vX;
        _angularVelocityRange.y = vY;
    }
    void billboardType(SLBillboardType bt) { _billboardType = bt; }
    void bezierControlPointAlpha(float arrayPoint[4])
    {
        _bezierControlPointAlpha[0] = arrayPoint[0];
        _bezierControlPointAlpha[1] = arrayPoint[1];
        _bezierControlPointAlpha[2] = arrayPoint[2];
        _bezierControlPointAlpha[3] = arrayPoint[3];
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
        _bezierControlPointSize[0] = arrayPoint[0];
        _bezierControlPointSize[1] = arrayPoint[1];
        _bezierControlPointSize[2] = arrayPoint[2];
        _bezierControlPointSize[3] = arrayPoint[3];
    }
    void bezierStartEndPointSize(float arrayPoint[4])
    {
        _bezierStartEndPointSize[0] = arrayPoint[0];
        _bezierStartEndPointSize[1] = arrayPoint[1];
        _bezierStartEndPointSize[2] = arrayPoint[2];
        _bezierStartEndPointSize[3] = arrayPoint[3];
    }
    void color(SLCol4f c) { _color = c; }
    void colorArr(SLfloat* arr) { std::copy(arr, arr + 256 * 3, _colorArr); }
    void direction(SLVec3f v) { _direction = v; }
    void direction(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _direction.x = vX;
        _direction.y = vY;
        _direction.z = vZ;
    }
    void doAcceleration(SLbool b) { _doAcceleration = b; }
    void doAlphaOverLT(SLbool b) { _doAlphaOverLT = b; }
    void doAlphaOverLTCurve(SLbool b) { _doAlphaOverLTCurve = b; }
    void doAccDiffDir(SLbool b) { _doAccDiffDir = b; }
    void doBlendBrightness(SLbool b) { _doBlendBrightness = b; }
    void doColor(SLbool b) { _doColor = b; }
    void doColorOverLT(SLbool b) { _doColorOverLT = b; }
    void doCounterGap(SLbool b) { _doCounterGap = b; }
    void doDirectionSpeed(SLbool b) { _doDirectionSpeed = b; }
    void doGravity(SLbool b) { _doGravity = b; }
    void doFlipBookTexture(SLbool b) { _doFlipBookTexture = b; }
    void doRotation(SLbool b) { _doRotation = b; }
    void doRotRange(SLbool b) { _doRotRange = b; }
    void doSpeedRange(SLbool b) { _doSpeedRange = b; }
    void doShape(SLbool b) { _doShape = b; }
    void doShapeSurface(SLbool b) { _doShapeSurface = b; }
    void doShapeOverride(SLbool b) { _doShapeOverride = b; }
    void doShapeSpawnBase(SLbool b) { _doShapeSpawnBase = b; }
    void doSizeOverLT(SLbool b) { _doSizeOverLT = b; }
    void doSizeOverLTCurve(SLbool b) { _doSizeOverLTCurve = b; }
    void doWorldSpace(SLbool b) { _doWorldSpace = b; }
    void emitterPos(SLVec3f p) { _emitterPos = p; }
    void gravity(SLVec3f v) { _gravity = v; }
    void gravity(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _gravity.x = vX;
        _gravity.y = vY;
        _gravity.z = vZ;
    }
    void flipbookColumns(SLint i) { _flipbookColumns = i; }
    void flipbookRows(SLint i) { _flipbookRows = i; }
    void frameRateFB(int i) { _flipbookFPS = i; }
    void isGenerated(SLbool b) { _isGenerated = b; }
    void radiusW(SLfloat f) { _radiusW = f; }
    void radiusH(SLfloat f) { _radiusH = f; }
    void scale(SLfloat f) { _scale = f; }
    void speed(SLfloat f) { _speed = f; }
    void speedRange(SLVec2f v) { _speedRange = v; }
    void speedRange(SLfloat vX, SLfloat vY)
    {
        _speedRange.x = vX;
        _speedRange.y = vY;
    }
    void shapeType(SLShapeType st) { _shapeType = st; }
    void shapeAngle(SLfloat f) { _shapeAngle = f; }
    void shapeRadius(SLfloat r) { _shapeRadius = r; }
    void shapeScale(SLVec3f v) { _shapeScale = v; }
    void shapeScale(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _shapeScale.x = vX;
        _shapeScale.y = vY;
        _shapeScale.z = vZ;
    }
    void shapeHeight(SLfloat f) { _shapeHeight = f; }
    void shapeWidth(SLfloat f) { _shapeWidth = f; }
    void timeToLive(SLfloat f) { _timeToLive = f; }
    void textureFirst(SLGLTexture* t) { _textureFirst = t; }
    void textureFlipbook(SLGLTexture* t) { _textureFlipbook = t; }
    void velocityType(SLint i) { _velocityType = i; }
    void velocityConst(SLVec3f v) { _velocityConst = v; }
    void velocityConst(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _velocityConst.x = vX;
        _velocityConst.y = vY;
        _velocityConst.z = vZ;
    }
    void velocityRndMin(SLVec3f v) { _velocityRndMin = v; }
    void velocityRndMin(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _velocityRndMin.x = vX;
        _velocityRndMin.y = vY;
        _velocityRndMin.z = vZ;
    }
    void velocityRndMax(SLVec3f v) { _velocityRndMax = v; }
    void velocityRndMax(SLfloat vX, SLfloat vY, SLfloat vZ)
    {
        _velocityRndMax.x = vX;
        _velocityRndMax.y = vY;
        _velocityRndMax.z = vZ;
    }

private:
    // Random point getter functions depending on the PS shape
    SLVec3f getPointInSphere(float radius, SLVec3f randomX);
    SLVec3f getPointOnSphere(float radius, SLVec3f randomX);
    SLVec3f getDirectionSphere(SLVec3f position);
    SLVec3f getPointInBox(SLVec3f boxScale);
    SLVec3f getPointOnBox(SLVec3f boxScale);
    SLVec3f getDirectionBox(SLVec3f position);
    SLVec3f getPointInCone();
    SLVec3f getPointOnCone();
    SLVec3f getDirectionCone(SLVec3f position);
    SLVec3f getPointInPyramid();
    SLVec3f getPointOnPyramid();
    SLVec3f getDirectionPyramid(SLVec3f position);

    // Core values
    SLint   _amount;         //!< Amount of a particle
    SLVec3f _emitterPos;     //!< Position of the particle emitter
    SLfloat _timeToLive;     //!< Time to live of a particle
    SLfloat _radiusW = 0.4f; //!< width radius of a particle
    SLfloat _radiusH = 0.4f; //!< height radius of a particle
    SLfloat _scale   = 1.0f; //!< Scale of a particle (Scale the radius)

    // Bezier
    SLVec4f _bernsteinPYAlpha            = SLVec4f(2.0f, -3.0f, 0.0f, 1.0f); //!< Vector for bezier curve (default linear function)
    float   _bezierControlPointAlpha[4]  = {0.0f, 1.0f, 1.0f, 0.0f};         //!< Floats for bezier curve control points (P1: 01 ; P2: 23)
    float   _bezierStartEndPointAlpha[4] = {0.0f, 1.0f, 1.0f, 0.0f};         //!< Floats for bezier curve end start points (Start: 01 ; End: 23)
    SLVec4f _bernsteinPYSize             = SLVec4f(-1.4f, 1.8f, 0.6f, 0.0f); //!< Vector for bezier curve (default linear function)
    float   _bezierControlPointSize[4]   = {0.0f, 0.0f, 1.0f, 1.0f};         //!< Floats for bezier curve control points (P1: 01 ; P2: 23)
    float   _bezierStartEndPointSize[4]  = {0.0f, 0.0f, 1.0f, 1.0f};         //!< Floats for bezier curve end start points (Start: 01 ; End: 23)

    // Acceleration
    SLVec3f _acceleration      = SLVec3f(1, 1, 1);            //!< Vector for acceleration (different direction as the velocity)
    SLfloat _accelerationConst = 0.0f;                        //!< Acceleration constant (same direction as the velocity)
    SLVec3f _gravity           = SLVec3f(0.0f, -9.81f, 0.0f); //!< Vector for gravity (2nd. acceleration vector)

    // Velocity
    SLVec3f _velocityConst  = SLVec3f(0, 1, 0);    //!< Velocity constant (go in xyz direction)
    SLVec3f _velocityRndMin = SLVec3f(-1, -1, -1); //!< Min. random velocity
    SLVec3f _velocityRndMax = SLVec3f(1, 1, 1);    //!< Max. random velocity

    // Direction speed
    SLVec3f _direction  = SLVec3f(0, 0, 0); //!< Direction of particle
    SLfloat _speed      = 1.0f;             //!< Speed of particle
    SLVec2f _speedRange = SLVec2f(1, 2);    //!< Speed random between two value

    // Color
    SLCol4f          _color = SLCol4f(0.66f, 0.0f, 0.66f, 0.2f); //!< Color for particle
    SLfloat          _colorArr[256 * 3];                         //!< Color values of color gradient widget
    SLVColorLUTPoint _colorPoints;                               //! Color gradient points

    // Int (but boolean) to switch buffer
    int _drawBuf = 0; //!< Boolean to switch buffer

    // Flipbook
    int   _flipbookFPS         = 16;   //!< Number of update of flipbook by second
    float _flipboookLastUpdate = 0.0f; //!< Last time flipbook was updated
    SLint _flipbookColumns     = 8;    //!< Number of flipbook sub-textures by column
    SLint _flipbookRows        = 8;    //!< Number of flipbook sub-textures by row

    // Statistics
    AvgFloat _updateTime;               //!< Averaged time for updating in MS
    AvgFloat _drawTime;                 //!< Averaged time for drawing in MS
    SLfloat  _startDrawTimeMS   = 0.0f; //!< Time since start of draw in MS
    SLfloat  _startUpdateTimeMS = 0.0f; //!< Time since start for updating in MS
    SLfloat  _startUpdateTimeS  = 0.0f; //!< Time since start for updating in S
    SLfloat  _deltaTimeUpdateS  = 0.0f; //!< Delta time in between two frames S

    // For resume after frustumCulling
    SLfloat _notVisibleTimeS = 0.0f; //!< Time since start when node not visible in S

    // For drawing while pause
    SLfloat _lastTimeBeforePauseS = 0.0f; //!< Time since the particle system is paused

    // Shape parameters
    SLfloat _shapeRadius = 1.0f;          //!< Radius of sphere and cone shape
    SLVec3f _shapeScale  = SLVec3f(1.0f); //!< Scale of box shape
    SLfloat _shapeHeight = 3.0f;          //!< Height of cone and pyramid shapes
    SLfloat _shapeAngle  = 10.0f;         //!< Angle of cone and pyramid shapes
    SLfloat _shapeWidth  = 2.0f;          //!< Width of pyramid shape

    // Rotation
    SLfloat _angularVelocityConst = 30.0f;                  //!< Rotation rate const (change in angular rotation divide by change in time)
    SLVec2f _angularVelocityRange = SLVec2f(-30.0f, 30.0f); //!< Rotation rate range (change in angular rotation divide by change in time)

    // Type of selected feature
    SLBillboardType _billboardType = BT_Camera; //!< Billboard type (BT_Camera, BT_Vertical, BT_Horizontal
    SLShapeType     _shapeType     = ST_Sphere; //!< Shape type (ST_)
    SLint           _velocityType  = 0;         //!< Velocity type

    // Textures
    SLGLTexture* _textureFirst;    //!< Main texture of PS (non flipbook)
    SLGLTexture* _textureFlipbook; //!< Flipbook texture with e.g. multiple flames at subsequent frames

    // VAOs
    SLGLVertexArray _vao1; //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2; //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

    // Boolean for generation/resume
    SLbool _isVisibleInFrustum = true;  //!< Boolean to set time since node not visible
    SLbool _isPaused           = false; //!< Boolean to stop updating
    SLbool _isGenerated        = false; //!< Boolean to generate particle system and load it on the GPU

    // Boolean for features
    SLbool _doBlendBrightness  = false; //!< Boolean for glow/brightness on pixel with many particle
    SLbool _doDirectionSpeed   = false; //!< Boolean for direction and speed (override velocity)
    SLbool _doSpeedRange       = false; //!< Boolean for speed range
    SLbool _doCounterGap       = true;  //!< Boolean for counter lag/gap, can create flickering with few particle (explained in documentation) when enable
    SLbool _doAcceleration     = false; //!< Boolean for acceleration
    SLbool _doAccDiffDir       = false; //!< Boolean for acceleration (different direction)
    SLbool _doRotation         = true;  //!< Boolean for rotation
    SLbool _doRotRange         = false; //!< Boolean for rotation range (random value between range)
    SLbool _doColor            = true;  //!< Boolean for color
    SLbool _doShape            = false; //!< Boolean for shape feature
    SLbool _doShapeSurface     = false; //!< Boolean for shape surface (particle will be spawned on surface)
    SLbool _doShapeOverride    = false; //!< Boolean for override direction for shape direction
    SLbool _doShapeSpawnBase   = false; //!< Boolean for spawn at base of shape (for cone and pyramid)
    SLbool _doWorldSpace       = false; //!< Boolean for world space position
    SLbool _doGravity          = false; //!< Boolean for gravity
    SLbool _doAlphaOverLT      = true;  //!< Boolean for alpha over life time
    SLbool _doColorOverLT      = false; //!< Boolean for color over life time
    SLbool _doAlphaOverLTCurve = false; //!< Boolean for alpha over life time curve
    SLbool _doSizeOverLT       = true;  //!< Boolean for size over life time
    SLbool _doSizeOverLTCurve  = false; //!< Boolean for size over life time curve
    SLbool _doFlipBookTexture  = false; //!< Boolean for flipbook texture
};
//-----------------------------------------------------------------------------
#endif
