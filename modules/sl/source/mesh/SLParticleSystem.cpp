//#############################################################################
//  File:      SLParticleSystem.cpp
//  Date:      February 2022
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Affolter Marc
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <climits>
#include <SLParticleSystem.h>
#include <SLMaterial.h>
#include <SLGLProgram.h>
#include <SLGLTexture.h>
#include <SLSceneView.h>
#include <GlobalTimer.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
//! SLParticleSystem ctor with some inital values. The number of particles
//! The particle emmiter position, the start and end random value range
//! The time to live of the particle (lifetime). A texture, a name
//! , and a flipbook texture.
//! particleEmiPos is not used (need to be removed)
SLParticleSystem::SLParticleSystem(SLAssetManager* assetMgr,
                                   const SLint     amount,
                                   const SLVec3f&  particleEmiPos,
                                   const SLVec3f&  velocityRandomStart,
                                   const SLVec3f&  velocityRandomEnd,
                                   const SLfloat&  timeToLive,
                                   SLGLTexture*    texC,
                                   const SLstring& name,
                                   SLGLTexture*    texFlipbook) : SLMesh(assetMgr, name)
{
    assert(!name.empty());

    _primitive = PT_points;

    if (amount > UINT_MAX) // Need to change for number of floats
        SL_EXIT_MSG("SLParticleSystem supports max. 2^32 vertices.");

    _timeToLive = timeToLive;
    _amount     = amount;
    _vRandS     = velocityRandomStart;
    _vRandE     = velocityRandomEnd;

    P.resize(1); // To trick parent class

    // emitterPos(particleEmiPos);

    _textureFirst    = texC;
    _textureFlipbook = texFlipbook;

    // Initialize the drawing:
    SLMaterial* mDraw = new SLMaterial(assetMgr, "Drawing-Material", this, texC);
    mat(mDraw);

    _updateTime.init(60, 0.0f);
    _drawTime.init(60, 0.0f);
}
//-----------------------------------------------------------------------------
//! Function which return a position in a sphere
SLVec3f SLParticleSystem::getPointInSphere(float radius, SLVec3f randomXs)
{
    float u  = random(0.0f, radius);
    float x1 = randomXs.x;
    float x2 = randomXs.y;
    float x3 = randomXs.z;

    float mag = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
    x1 /= mag;
    x2 /= mag;
    x3 /= mag;

    float c = cbrt(u);

    return SLVec3f(x1 * c, x2 * c, x3 * c);
}
//-----------------------------------------------------------------------------
//! Function which return a position on a sphere
SLVec3f SLParticleSystem::getPointOnSphere(float radius, SLVec3f randomXs)
{
    float x1 = randomXs.x;
    float x2 = randomXs.y;
    float x3 = randomXs.z;

    float mag = sqrt(x1 * x1 + x2 * x2 + x3 * x3);
    x1 /= mag;
    x2 /= mag;
    x3 /= mag;

    float c = cbrt(radius);

    return SLVec3f(x1 * c, x2 * c, x3 * c);
}
//-----------------------------------------------------------------------------
//! Function which return the direction towards the exterior of a sphere
SLVec3f SLParticleSystem::getDirectionSphere(SLVec3f position)
{
    return (position - SLVec3f(0.0f, 0.0f, 0.0f)).normalized(); // Get unit vector center to position
}
//-----------------------------------------------------------------------------
//! Function which return a position in a box
SLVec3f SLParticleSystem::getPointInBox(SLVec3f boxScale)
{
    float x = random(-boxScale.x, boxScale.x);
    float y = random(-boxScale.y, boxScale.y);
    float z = random(-boxScale.z, boxScale.z);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
//! Function which return a position on a box
SLVec3f SLParticleSystem::getPointOnBox(SLVec3f boxScale)
{
    int   temp = random(0, 5);
    float x    = 0.0f;
    float y    = 0.0f;
    float z    = 0.0f;
    if (temp == 0)
    { // LEFT side
        x = -boxScale.x;
        y = random(-boxScale.y, boxScale.y);
        z = random(-boxScale.z, boxScale.z);
    }
    else if (temp == 1) // RIGHT side
    {
        x = boxScale.x;
        y = random(-boxScale.y, boxScale.y);
        z = random(-boxScale.z, boxScale.z);
    }
    else if (temp == 2) // FRONT side
    {
        x = random(-boxScale.x, boxScale.x);
        y = random(-boxScale.y, boxScale.y);
        z = boxScale.z;
    }
    else if (temp == 3) // BACK side
    {
        x = random(-boxScale.x, boxScale.x);
        y = random(-boxScale.y, boxScale.y);
        z = -boxScale.z;
    }
    else if (temp == 4) // TOP side
    {
        x = random(-boxScale.x, boxScale.x);
        y = boxScale.y;
        z = random(-boxScale.z, boxScale.z);
    }
    else if (temp == 5) // BOTTOM side
    {
        x = random(-boxScale.x, boxScale.x);
        y = -boxScale.y;
        z = random(-boxScale.z, boxScale.z);
    }

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
//! Function which return the direction towards the exterior of a box
SLVec3f SLParticleSystem::getDirectionBox(SLVec3f position)
{
    return (position - SLVec3f(0.0f, 0.0f, 0.0f)).normalized(); // Get unit vector center to position
}
//-----------------------------------------------------------------------------
//! Function which return a position in the cone define in the particle system
SLVec3f SLParticleSystem::getPointInCone()
{
    float y      = 0.0f;
    float radius = _radiusCone;
    if (!_doShapeSpawnBase) // Spawn inside volume
    {
        y      = random(0.0f, _heightCone); // NEED TO HAVE MORE value near 1 when we have smaller base that top
        radius = _radiusCone + tan(_angleCone * DEG2RAD) * y;
    }
    float r     = radius * sqrt(random(0.0f, 1.0f));
    float theta = random(0.0f, 1.0f) * 2 * PI;
    float x     = r * cos(theta);
    float z     = r * sin(theta);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
//! Function which return a position on the cone define in the particle system
SLVec3f SLParticleSystem::getPointOnCone()
{
    float y      = 0.0f;
    float radius = _radiusCone;
    if (!_doShapeSpawnBase) // Spawn inside volume
    {
        y      = random(0.0f, _heightCone); // NEED TO HAVE MORE value near 1 when we have smaller base that top
        radius = _radiusCone + tan(_angleCone * DEG2RAD) * y;
    }
    float r     = radius;
    float theta = random(0.0f, 1.0f) * 2 * PI;
    float x     = r * cos(theta);
    float z     = r * sin(theta);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
//! Function which return a direction following the cone shape define in the
//! particle system.
SLVec3f SLParticleSystem::getDirectionCone(SLVec3f position)
{
    float maxRadius = _radiusCone + tan(_angleCone * DEG2RAD) * _heightCone; // Calculate max radius
    float percentX  = position.x / maxRadius;                                // Calculate at which pourcent our x is, to know how much we need to adapt our angle
    float percentZ  = position.z / maxRadius;                                // Calculate at which pourcent our z is, to know how much we need to adapt our angle
    float newX      = position.x + tan(_angleCone * percentX * DEG2RAD) * _heightCone;
    float newZ      = position.z + tan(_angleCone * percentZ * DEG2RAD) * _heightCone;
    return SLVec3f(newX, _heightCone, newZ).normalize();
}
//-----------------------------------------------------------------------------
//! Function which return a position in the pyramid define in the particle system
SLVec3f SLParticleSystem::getPointInPyramid()
{
    float y      = 0.0f;
    float radius = _halfSidePyramid;
    if (!_doShapeSpawnBase) // Spawn inside volume
    {
        y      = random(0.0f, _heightPyramid);
        radius = _halfSidePyramid + tan(_anglePyramid * DEG2RAD) * y;
    }
    float x = random(-radius, radius);
    float z = random(-radius, radius);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
//! Function which return a position on the pyramid define in the particle system
SLVec3f SLParticleSystem::getPointOnPyramid()
{
    float y      = 0.0f;
    float radius = _halfSidePyramid;
    if (!_doShapeSpawnBase) // Spawn inside volume
    {
        y      = random(0.0f, _heightPyramid);
        radius = _halfSidePyramid + tan(_anglePyramid * DEG2RAD) * y;
    }

    // int   temp      = random(0, 5);
    int   temp = random(0, 3);
    float x    = 0.0f;
    float z    = 0.0f;
    if (temp == 0)
    { // LEFT
        x = -radius;
        z = random(-radius, radius);
    }
    else if (temp == 1) // RIGHT
    {
        x = radius;
        z = random(-radius, radius);
    }
    else if (temp == 2) // FRONT
    {
        x = random(-radius, radius);
        z = radius;
    }
    else if (temp == 3) // BACK
    {
        x = random(-radius, radius);
        z = -radius;
    }
    // Comments to have top and bottom not filled
    /* else if (temp == 4) //TOP
    {
        y = _heightPyramid;
        radius = _halfSidePyramid + tan(_anglePyramid * DEG2RAD) * y;

        x = random(-radius, radius);
        z = random(-radius, radius);
    }
    else if (temp == 5) // BOTTOM
    {
        y      = 0.0f;
        radius = _halfSidePyramid;
        x = random(-radius, radius);
        z = random(-radius, radius);
    }*/

    return SLVec3f(x, y, z);
}
// ----------------------------------------------------------------------------
//! Function which return a direction following the pyramide shape define in the
//! particle system.
SLVec3f SLParticleSystem::getDirectionPyramid(SLVec3f position)
{
    float maxRadius = _halfSidePyramid +
                      tan(_anglePyramid * DEG2RAD) * _heightPyramid;

    // Calculate at which percent our x & z is to know how much we need to adapt our angle
    float percentX = position.x / maxRadius;
    float percentZ = position.z / maxRadius;
    float newX     = position.x + tan(_anglePyramid * percentX * DEG2RAD) * _heightPyramid;
    float newZ     = position.z + tan(_anglePyramid * percentZ * DEG2RAD) * _heightPyramid;
    return SLVec3f(newX, _heightPyramid, newZ).normalize();
}
//-----------------------------------------------------------------------------
//! Function which will generate the particles, attribes them to the VAO
void SLParticleSystem::generate()
{
    SLuint                     seed = (SLuint)chrono::system_clock::now().time_since_epoch().count();
    default_random_engine      generator(seed);
    normal_distribution<float> distribution(0.0f, 1.0f);

    SLVVec3f tempP;
    SLVVec3f tempV;
    SLVfloat tempST;
    SLVVec3f tempInitV;
    SLVfloat tempR;
    SLVfloat tempAngulareVelo;
    SLVuint  tempTexNum;
    SLVVec3f tempInitP;

    tempP.resize(_amount);
    tempV.resize(_amount);
    tempST.resize(_amount);
    if (_doAcceleration || _doGravity)
        tempInitV.resize(_amount);
    if (_doRotation)
        tempR.resize(_amount);
    if (_doRotation && _doRotRange)
        tempAngulareVelo.resize(_amount);
    if (_doFlipBookTexture)
        tempTexNum.resize(_amount);
    if (_doShape)
        tempInitP.resize(_amount);

    // Normal generation
    for (SLint i = 0; i < _amount; i++)
    {
        if (_doShape && _shapeType == 0) // Position in or on sphere
        {
            if (!_doShapeSurface) // In volume
                tempP[i] = getPointInSphere(_radiusSphere,
                                            SLVec3f(distribution(generator),
                                                    distribution(generator),
                                                    distribution(generator)));
            else // On surface
                tempP[i] = getPointOnSphere(_radiusSphere,
                                            SLVec3f(distribution(generator),
                                                    distribution(generator),
                                                    distribution(generator)));
        }
        else if (_doShape && _shapeType == 1) // Position in or on box
            if (!_doShapeSurface)
                tempP[i] = getPointInBox(_scaleBox);
            else
                tempP[i] = getPointOnBox(_scaleBox);
        else if (_doShape && _shapeType == 2) // Position in or on cone
            if (!_doShapeSurface)
                tempP[i] = getPointInCone();
            else
                tempP[i] = getPointOnCone();
        else if (_doShape && _shapeType == 3) // Position in or on pyramid
            if (!_doShapeSurface)
                tempP[i] = getPointInPyramid();
            else
                tempP[i] = getPointOnPyramid();
        else // Position is not a volumn, spawn from start point (particle emmiter position)
            tempP[i] = SLVec3f(0, 0, 0); 

        if (!_doDirectionSpeed) // Use normal velocity
        {
            if (_velocityType == 0) // Random value
            {
                tempV[i].x = random(_vRandS.x, _vRandE.x); // Random value for x velocity
                tempV[i].y = random(_vRandS.y, _vRandE.y); // Random value for y velocity
                tempV[i].z = random(_vRandS.z, _vRandE.z); // Random value for z velocity
            }
            else if (_velocityType == 1) // Constant
            {
                tempV[i].x = _velocityConst.x; // Constant value for x velocity
                tempV[i].y = _velocityConst.y; // Constant value for y velocity
                tempV[i].z = _velocityConst.z; // Constant value for z velocity
            }
        }
        else // DO direction and speed
        {
            SLVec3f tempDirection;
            if (_doShapeOverride) // Direction for shape
            {
                if (_doShape && _shapeType == 0)
                    tempDirection = getDirectionSphere(tempP[i]);
                else if (_doShape && _shapeType == 1)
                    tempDirection = getDirectionBox(tempP[i]);
                else if (_doShape && _shapeType == 2)
                    tempDirection = getDirectionCone(tempP[i]);
                else if (_doShape && _shapeType == 3)
                    tempDirection = getDirectionPyramid(tempP[i]);
            }
            else // Normal direction
                tempDirection = _direction;

            if (_doSpeedRange) // Apply speed
                tempV[i] = tempDirection * random(_speedRange.x, _speedRange.y);
            else
                tempV[i] = tempDirection * _speed;
        }

        // When the first particle dies the last one begin to live
        tempST[i] = GlobalTimer::timeS() + ((float)i * (_timeToLive / (float)_amount)); // Time to start

        if (_doAcceleration || _doGravity) // Acceleration
            tempInitV[i] = tempV[i];
        if (_doRotation) // Rotation (constant angular velocity)
            tempR[i] = random(0.0f * DEG2RAD, 360.0f * DEG2RAD); // Start rotation of the particle
        if (_doRotation && _doRotRange) // Random angular velocity for each particle
            tempAngulareVelo[i] = random(_angularVelocityRange.x * DEG2RAD,
                                         _angularVelocityRange.y * DEG2RAD); // Start rotation of the particle
        if (_doFlipBookTexture) // Flipbook texture
            tempTexNum[i] = random(0, _row * _col - 1);
        if (_doShape)   // Shape feature
            tempInitP[i] = tempP[i];
    }

    // Need to have two VAo for transform feedback swapping
    _vao1.deleteGL();
    _vao2.deleteGL();

    _vao1.setAttrib(AT_position, AT_position, &tempP);
    _vao1.setAttrib(AT_velocity, AT_velocity, &tempV);
    _vao1.setAttrib(AT_startTime, AT_startTime, &tempST);
    if (_doAcceleration || _doGravity)
        _vao1.setAttrib(AT_initialVelocity, AT_initialVelocity, &tempInitV);
    if (_doRotation)
        _vao1.setAttrib(AT_rotation, AT_rotation, &tempR);
    if (_doRotation && _doRotRange)
        _vao1.setAttrib(AT_angularVelo, AT_angularVelo, &tempAngulareVelo);
    if (_doFlipBookTexture)
        _vao1.setAttrib(AT_texNum, AT_texNum, &tempTexNum);
    if (_doShape)
        _vao1.setAttrib(AT_initialPosition, AT_initialPosition, &tempInitP);
    _vao1.generateTF((SLuint)tempP.size());

    _vao2.setAttrib(AT_position, AT_position, &tempP);
    _vao2.setAttrib(AT_velocity, AT_velocity, &tempV);
    _vao2.setAttrib(AT_startTime, AT_startTime, &tempST);
    if (_doAcceleration || _doGravity)
        _vao2.setAttrib(AT_initialVelocity, AT_initialVelocity, &tempInitV);
    if (_doRotation)
        _vao2.setAttrib(AT_rotation, AT_rotation, &tempR);
    if (_doRotation && _doRotRange)
        _vao2.setAttrib(AT_angularVelo, AT_angularVelo, &tempAngulareVelo);
    if (_doFlipBookTexture)
        _vao2.setAttrib(AT_texNum, AT_texNum, &tempTexNum);
    if (_doShape)
        _vao2.setAttrib(AT_initialPosition, AT_initialPosition, &tempInitP);
    _vao2.generateTF((SLuint)tempP.size());
}
//-----------------------------------------------------------------------------
/*!
Generate Bernstein Polynomial with 4 controls points for alpha over life.
ContP contains 2 and 3 controls points
StatEnd contains 1 and 4 controls points
*/
void SLParticleSystem::generateBernsteinPAlpha()
{
    float* ContP  = _bezierControlPointAlpha;
    float* StaEnd = _bezierStartEndPointAlpha;
    // For Y bezier curve
    // T^3
    _bernsteinPYAlpha.x = -StaEnd[1] + ContP[1] * 3 - ContP[3] * 3 + StaEnd[3];
    // T^2
    _bernsteinPYAlpha.y = StaEnd[1] * 3 - ContP[1] * 6 + ContP[3] * 3;
    // T
    _bernsteinPYAlpha.z = -StaEnd[1] * 3 + ContP[1] * 3;
    // 1
    _bernsteinPYAlpha.w = StaEnd[1];
}
//-----------------------------------------------------------------------------
/*!
Generate Bernstein Polynomial with 4 controls points for size over life.
ContP contains 2 and 3 controls points
StatEnd contains 1 and 4 controls points
*/
void SLParticleSystem::generateBernsteinPSize()
{
    // For Y bezier curve
    // T^3
    float* ContP       = _bezierControlPointSize;
    float* StaEnd      = _bezierStartEndPointSize;
    _bernsteinPYSize.x = -StaEnd[1] + ContP[1] * 3 - ContP[3] * 3 + StaEnd[3];
    // T^2
    _bernsteinPYSize.y = StaEnd[1] * 3 - ContP[1] * 6 + ContP[3] * 3;
    // T
    _bernsteinPYSize.z = -StaEnd[1] * 3 + ContP[1] * 3;
    // 1
    _bernsteinPYSize.w = StaEnd[1];
}
//-----------------------------------------------------------------------------
/*!
Change the current use texture, this will switch between the normal texture and
the flipbook texture (and vice versa)
*/
void SLParticleSystem::changeTexture()
{
    if (_doFlipBookTexture)
    {
        mat()->removeTextureType(TT_diffuse);
        mat()->addTexture(_textureFlipbook);
    }
    else
    {
        mat()->removeTextureType(TT_diffuse);
        mat()->addTexture(_textureFirst);
    }
}
//-----------------------------------------------------------------------------
/*!
Function called inside SLNode cull3DRec(..) which set a boolean and a time for the next
draw call to update the start time of the particles.
*/
void SLParticleSystem::notVisibleFrustumCulling()
{
    if (_isViFrustumCulling)
    {
        _isViFrustumCulling = false;
        _notVisibleTimeS    = GlobalTimer::timeS();
    }
}
//-----------------------------------------------------------------------------
/*!
Function called by the user to pause or resume the particle system. This will
freeze the particle system there won't be any updating, only the drawing
*/
void SLParticleSystem::pauseOrResume()
{
    if (!_isPaused)
    {
        _isPaused             = true;
        _lastTimeBeforePauseS = GlobalTimer::timeS();
    }
    else
        _isPaused = false;
}
//-----------------------------------------------------------------------------
/*!
Draw function override from SLMesh. In this function I start by generate the
particle and programs if they are not, then I check if the particle have been culled
by the frstum culling or if they have been resumed by the user. After I update
the particle in the update pass, then and finally I draw them.
*/
void SLParticleSystem::draw(SLSceneView* sv, SLNode* node)
{
    /////////////////////////////////////
    // Init particles vector and init VAO
    /////////////////////////////////////

    if (!_isGenerated)
    {
        // emitterPos(node->translationWS()); //To init first position
        generate();
        _isGenerated = true;
    }

    /////////////////////////////
    // Generate programs
    /////////////////////////////

    if (!_mat->program() || !_mat->programTF())
        _mat->generateProgramPS();


    /////////////////////////////
    // Calculate time and paused and frustum culling
    /////////////////////////////

    float difTime   = 0.0f;
    float deltaTime = GlobalTimer::timeS() - _startUpdateTimeS; // Actual delta time

    // Calculate time difference for frustum culling and paused
    if (!_isViFrustumCulling && !_isPaused && _lastTimeBeforePauseS != 0.0f) // If particle system was not visible, and was resumed when it was not visible
    {
        _isViFrustumCulling = true;
        difTime             = GlobalTimer::timeS() - min(_lastTimeBeforePauseS, _notVisibleTimeS); // Paused was set before not visible (will take _lastTimeBeforePauseS), if set after (will take _notVisibleTimeS)
        // maybe add later average delta time (because maybe bug when fast not visible long time, visible, not visible, visible
        deltaTime             = _deltaTimeUpdateS; // Last delta time, because when culled draw is not called therefore the actual delta time will be too big
        _notVisibleTimeS      = 0.0f;              // No more culling, the difference time has been applied, no further need
        _lastTimeBeforePauseS = 0.0f;              // No more paused, the difference time has been applied, no further need
    }
    else if (!_isViFrustumCulling) // If particle system was not visible, this one is called just once when the particle is draw again (Do nothing if paused, because update call is not done)
    {
        _isViFrustumCulling = true;
        difTime             = GlobalTimer::timeS() - _notVisibleTimeS; // Use time since the particle system was not visible
        // maybe add later average delta time (because maybe bug when fast not visible long time, visible, not visible, visible
        deltaTime = _deltaTimeUpdateS;                // Last delta time, because when culled draw is not called therefore the actual delta time will be too big
        if (_lastTimeBeforePauseS > _notVisibleTimeS) // If was paused when not visible. Need to take _notVisibleTimeS because it's since this value that the particle system is not drew.
            _lastTimeBeforePauseS = _notVisibleTimeS; // Get the value of since the particle system is not drew
        _notVisibleTimeS = 0.0f;                      // No more culling, the difference time has been applied, no further need
    }
    else if (!_isPaused && _lastTimeBeforePauseS != 0.0f) // If particle system was resumed
    {
        difTime               = GlobalTimer::timeS() - _lastTimeBeforePauseS; // Use time since the particle system was paused
        _lastTimeBeforePauseS = 0.0f;                                         // No more paused, the difference time has been applied, no further need

        // Take default delta time, because when just paused no need to take last delta time, the draw call continue to be called
    }

    // Calculate the elapsed time for the updating, need to change to use a real profiler (can't measure time like this on the GPU)
    _startUpdateTimeMS = GlobalTimer::timeMS(); 

    // MS above, S below
    _deltaTimeUpdateS = GlobalTimer::timeS() - _startUpdateTimeS;
    _startUpdateTimeS = GlobalTimer::timeS();

    if (!_isPaused) // The updating is paused, therefore no need to send uniforms
    {
        /////////////////////////////
        // UPDATING
        /////////////////////////////

        // Now use the updating program
        SLGLProgram* spTF = _mat->programTF();
        spTF->useProgram();

        /////////////////////////////
        // Apply Uniform Variables
        /////////////////////////////

        // Time difference, between when the particle system was culled or paused or both
        spTF->uniform1f("u_difTime", difTime);

        // Time between each draw call, take delta time from last draw called after frustum culling
        spTF->uniform1f("u_deltaTime", deltaTime);

        // Time since process start
        spTF->uniform1f("u_time", _startUpdateTimeS);

        if (_doAcceleration)
        {
            if (_doAccDiffDir)
                spTF->uniform3f("u_acceleration", _acceleration.x, _acceleration.y, _acceleration.z);
            else
                spTF->uniform1f("u_accConst", _accelerationConst);
        }

        if (_doGravity)
            spTF->uniform3f("u_gravity", _gravity.x, _gravity.y, _gravity.z);

        spTF->uniform1f("u_tTL", _timeToLive);

        emitterPos(node->translationWS());

        // Worldspace
        if (_doWorldSpace)
            spTF->uniform3f("u_pGPosition",
                            _emitterPos.x,
                            _emitterPos.y,
                            _emitterPos.z);
        else
            spTF->uniform3f("u_pGPosition",
                            0.0f,
                            0.0f,
                            0.0f);

        // Flipbook
        if (_doFlipBookTexture)
        {
            spTF->uniform1i("u_col", _col);
            spTF->uniform1i("u_row", _row);
            _lastUpdateFB += _deltaTimeUpdateS;

            if (_lastUpdateFB > (1.0f / (float)_frameRateFB))
            {
                // Last time FB was updated is bigger than the time needed for each update
                spTF->uniform1i("u_condFB", 1);
                _lastUpdateFB = 0.0f;
            }
            else
                spTF->uniform1i("u_condFB", 0);
        }

        // Rotation
        if (_doRotation && !_doRotRange)
            spTF->uniform1f("u_angularVelo", _angularVelocityConst * DEG2RAD);

        /////////////////////////////
        // Draw call to update
        /////////////////////////////

        if (_drawBuf == 0)
        {
            _vao1.beginTF(_vao2.tfoID());
            _vao1.drawArrayAs(PT_points);
            _vao1.endTF();
            _vao = _vao2;
        }
        else
        {
            _vao2.beginTF(_vao1.tfoID());
            _vao2.drawArrayAs(PT_points);
            _vao2.endTF();
            _vao = _vao1;
        }
        _updateTime.set(GlobalTimer::timeMS() - _startUpdateTimeMS);
    }

    /////////////////////////////
    // DRAWING
    /////////////////////////////

    // Give uniform for drawing and find for linking vao vbo
    SLGLProgram* spD = _mat->program();
    spD->useProgram();

    SLGLState* stateGL = SLGLState::instance();

    // Start calculation of the elapsed time for the drawing, need to change to use a real profiler (can't measure time like this on the GPU)
    _startDrawTimeMS = GlobalTimer::timeMS();

    // Billboard type
    // World space
    if (_doWorldSpace) {
        
        if (_billboardType == BT_Vertical)
        {
            SLMat4f vMat = stateGL->viewMatrix; // Just view matrix because world space is enabled

            vMat.m(0, 1.0f);
            vMat.m(1, 0.0f);
            vMat.m(2, 0.0f);

            vMat.m(8, 0.0f);
            vMat.m(9, 0.0f);
            vMat.m(10, 1.0f);

            spD->uniformMatrix4fv("u_vYawPMatrix",
                                  1,
                                  (SLfloat*)&vMat); // TO change for custom shader generation
        }
        else
        {
            spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
        }
    }   
    else
    {
        SLMat4f mvMat = stateGL->viewMatrix * stateGL->modelMatrix; // Model-View Matrix

        if (_billboardType == BT_Vertical)
        {
            mvMat.m(0, 1.0f);
            mvMat.m(1, 0.0f);
            mvMat.m(2, 0.0f);

            mvMat.m(8, 0.0f);
            mvMat.m(9, 0.0f);
            mvMat.m(10, 1.0f);

            spD->uniformMatrix4fv("u_vYawPMatrix",
                                  1,
                                  (SLfloat*)&mvMat); // TO change for custom shader generation
        }
        else
        {
            spD->uniformMatrix4fv("u_vOmvMatrix",
                                  1,
                                  (SLfloat*)&mvMat); // TO change for custom shader generation
        }
    }

    // Alpha over life Bézier curve
    if (_doAlphaOverLCurve)
    {
        spD->uniform4f("u_al_bernstein",
                       _bernsteinPYAlpha.x,
                       _bernsteinPYAlpha.y,
                       _bernsteinPYAlpha.z,
                       _bernsteinPYAlpha.w);
    }

    // Size over life Bézier curve
    if (_doSizeOverLFCurve)
    {
        spD->uniform4f("u_si_bernstein",
                       _bernsteinPYSize.x,
                       _bernsteinPYSize.y,
                       _bernsteinPYSize.z,
                       _bernsteinPYSize.w);
    }

    // Color over life by gradient color editor
    if (_doColorOverLF)
    {
        spD->uniform1fv("u_colorArr",
                        256 * 3,
                        _colorArr);
    }
    else
    {
        spD->uniform4f("u_color",
                       _color.x,
                       _color.y,
                       _color.z,
                       _color.w);
    }

    // Flipbook
    if (_doFlipBookTexture)
    {
        spD->uniform1i("u_col", _col);
        spD->uniform1i("u_row", _row);
    }

    if (_isPaused) // Take time when pause was enable
        spD->uniform1f("u_time", _lastTimeBeforePauseS);
    else
        spD->uniform1f("u_time", GlobalTimer::timeS());

    spD->uniform1f("u_tTL", _timeToLive);
    spD->uniform1f("u_scale", _scale);
    spD->uniform1f("u_radiusW", _radiusW);
    spD->uniform1f("u_radiusH", _radiusH);

    spD->uniform1f("u_oneOverGamma", 1.0f);

    // Check wireframe
    if (sv->drawBits()->get(SL_DB_MESHWIRED) || node->drawBits()->get(SL_DB_MESHWIRED))
        spD->uniform1i("u_doWireFrame", 1);
    else
        spD->uniform1i("u_doWireFrame", 0);

    if (_doColor && _doBlendBrightness)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE);

    ///////////////////////
    SLMesh::draw(sv, node);
    ///////////////////////

    if (_doColor && _doBlendBrightness)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // End calculation of the elapsed time for the drawing
    _drawTime.set(GlobalTimer::timeMS() - _startDrawTimeMS);

    // Swap buffer
    _drawBuf = 1 - _drawBuf;
}
//-----------------------------------------------------------------------------
//! deleteData deletes all mesh data and vbo's
void SLParticleSystem::deleteData()
{
    _vao1.deleteGL();
    _vao2.deleteGL();
    SLMesh::deleteData();
}
//-----------------------------------------------------------------------------
//! deleteData deletes all mesh data and vbo's
void SLParticleSystem::deleteDataGpu()
{
    _vao1.deleteGL();
    _vao2.deleteGL();
    SLMesh::deleteDataGpu();
}
//-----------------------------------------------------------------------------
/*!
SLParticleSystem::buildAABB builds the passed axis-aligned bounding box in OS and updates
the min & max points in WS with the passed WM of the node. Take into account features like
acceleration, gravity, shape, velocity.
*/
// SLMesh::buildAABB builds the passed axis-aligned bounding box in OS and updates
// the min& max points in WS with the passed WM of the node.
  //! Todo: Can ben enhance furthemore, the acceleration doesn't work wll for the moments
//! The negative value for the acceleration are not take into account and also acceleration
//! which goes agains the velocity. To adapt the acceleration to exactly the same as the gravity
//! Not enough time to do it. Need to adapt more accurently when direction speed is negative (for
//! shape override example with Cone)
//! 
void SLParticleSystem::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    // Radius of particle
    float rW = _radiusW * _scale;
    float rH = _radiusH * _scale;

    // Speed for direction if used
    float tempSpeed;
    if (_doSpeedRange)
        tempSpeed = max(_speedRange.x, _speedRange.y);
    else
        tempSpeed = _speed;

    // Empty point
    minP = SLVec3f();
    maxP = SLVec3f();

    // Empty velocity
    SLVec3f minV = SLVec3f();
    SLVec3f maxV = SLVec3f();

    // Shape
    if (_doShape)
    {
        if (_shapeType == 0)
        {
            float radius = cbrt(_radiusSphere);
            minP         = SLVec3f(-radius, -radius, -radius);
            maxP         = SLVec3f(radius, radius, radius);
            // Override direction
            if (_doDirectionSpeed && _doShapeOverride)
            {
                minV += SLVec3f(-radius, -radius, -radius) * tempSpeed;
                maxV += SLVec3f(radius, radius, radius) * tempSpeed;
            }
        }
        if (_shapeType == 1)
        {
            minP = SLVec3f(-_scaleBox.x, -_scaleBox.y, -_scaleBox.z);
            maxP = _scaleBox;
            if (_doDirectionSpeed && _doShapeOverride)
            {
                minV += SLVec3f(-_scaleBox.x, -_scaleBox.y, -_scaleBox.z) * tempSpeed;
                maxV += _scaleBox * tempSpeed;
            }
        }
        if (_shapeType == 2)
        {
            float radius = _radiusCone + tan(_angleCone * DEG2RAD) * _heightCone;
            minP         = SLVec3f(-radius, -0.0, -radius);
            if (!_doShapeSpawnBase) // Spawn inside volume
                maxP = SLVec3f(radius, _heightCone, radius);
            else // Spawn base volume
                maxP = SLVec3f(radius, 0.0f, radius);
            if (_doDirectionSpeed && _doShapeOverride)
            {
                SLVec3f temp = getDirectionCone(SLVec3f(-radius, -0.0, -radius)) * tempSpeed;
                temp.y       = 0.0;
                minV += temp;
                maxV += getDirectionCone(SLVec3f(radius, _heightCone, radius)) * tempSpeed;
            }
        }
        if (_shapeType == 3)
        {

            float radius = _halfSidePyramid + tan(_anglePyramid * DEG2RAD) * _heightPyramid;
            minP         = SLVec3f(-radius, -0.0, -radius);
            if (!_doShapeSpawnBase) // Spawn inside volume
                maxP = SLVec3f(radius, _heightPyramid, radius);
            else // Spawn base volume
                maxP = SLVec3f(radius, 0.0f, radius);
            if (_doDirectionSpeed && _doShapeOverride)
            {
                SLVec3f temp = getDirectionPyramid(SLVec3f(-radius, -0.0, -radius)) * tempSpeed;
                temp.y       = 0.0;
                minV += temp;
                maxV += getDirectionPyramid(SLVec3f(radius, _heightPyramid, radius)) * tempSpeed;
            }
        }
    }

    if (_doAcceleration || _doGravity) // If acceleration or gravity is enable
    {
        if (!_doDirectionSpeed) // If direction is not enable
        {
            if (_velocityType == 0)
            {
                SLVec3f minVTemp = SLVec3f(min(_vRandE.x, _vRandS.x), min(_vRandE.y, _vRandS.y), min(_vRandE.z, _vRandS.z)); // Apply velocity distance after time
                SLVec3f maxVTemp = SLVec3f(max(_vRandE.x, _vRandS.x), max(_vRandE.y, _vRandS.y), max(_vRandE.z, _vRandS.z)); // Apply velocity distance after time

                if (minVTemp.x > 0 && maxVTemp.x > 0) minVTemp.x = 0.0;
                if (minVTemp.y > 0 && maxVTemp.y > 0) minVTemp.y = 0.0;
                if (minVTemp.z > 0 && maxVTemp.z > 0) minVTemp.z = 0.0;

                if (minVTemp.x < 0 && maxVTemp.x < 0) maxVTemp.x = 0.0;
                if (minVTemp.y < 0 && maxVTemp.y < 0) maxVTemp.y = 0.0;
                if (minVTemp.z < 0 && maxVTemp.z < 0) maxVTemp.z = 0.0;

                minV += minVTemp;
                maxV += maxVTemp;
            }
            else // Constant acceleration
            {
                minV = SLVec3f(_velocityConst.x, 0.0, 0.0);
                maxV = SLVec3f(0.0, _velocityConst.y, _velocityConst.z);
            }
        }
        else if (_doDirectionSpeed && !_doShapeOverride) // Direction and speed, but no shape override, because I want a constant direction in one way I don't want the direction to be overrided
        {
            tempSpeed = 0.0f;
            if (_doSpeedRange)
                tempSpeed = max(_speedRange.x, _speedRange.y);
            else
                tempSpeed = _speed;
            
            minV = SLVec3f(_direction.x, 0.0, 0.0) * tempSpeed;
            maxV = SLVec3f(0.0, _direction.y, _direction.z) * tempSpeed;
            if (_direction.x > 0.0)
            {
                maxV.x = minV.x;
                minV.x = 0.0;
            }
            if (_direction.y < 0.0)
            {
                minV.y = maxV.y;
                maxV.y = 0.0;
            }
            if (_direction.z < 0.0)
            {
                minV.z = maxV.z;
                maxV.z = 0.0;
            }
        }

        // Apply time to velocity
        minP += minV * _timeToLive;
        maxP += maxV * _timeToLive;

        // GRAVITY
        if (_doGravity)
        {
            float timeForXGrav = 0.0f;
            float timeForYGrav = 0.0f;
            float timeForZGrav = 0.0f;
            // Time to have a velocity of 0
            if (_gravity.x != 0.0f) timeForXGrav = maxV.x / _gravity.x;
            if (_gravity.y != 0.0f) timeForYGrav = maxV.y / _gravity.y;
            if (_gravity.z != 0.0f) timeForZGrav = maxV.z / _gravity.z;

            if (timeForXGrav < 0.0f)                             // If the gravity go against the velocity
                maxP.x -= maxV.x * (_timeToLive + timeForXGrav); // I remove the position  with the velocity that it will not do because it go against the velocity (becareful! here timeForXGrav is negative)
            else if (timeForXGrav > 0.0f)                        // If the gravity go with the velocity
                maxP.x += 0.5f * _gravity.x * (_timeToLive * _timeToLive);
            if (timeForYGrav < 0.0f) // If the gravity go against the velocity
                maxP.y -= maxV.y * (_timeToLive + timeForYGrav);
            else if (timeForYGrav > 0.0f) // If the gravity go with the velocity
                maxP.y += 0.5f * _gravity.y * (_timeToLive * _timeToLive);
            if (timeForZGrav < 0.0f) // If the gravity go against the velocity
                maxP.z -= maxV.z * (_timeToLive + timeForZGrav);
            else if (timeForZGrav > 0.0f) // If the gravity go with the velocity
                maxP.z += 0.5f * _gravity.z * (_timeToLive * _timeToLive);

            // Time remaining after the gravity has nullified the velocity for the particle to die (for each axes)
            float xTimeRemaining = _timeToLive - abs(timeForXGrav);
            float yTimeRemaining = _timeToLive - abs(timeForYGrav);
            float zTimeRemaining = _timeToLive - abs(timeForZGrav);

            if (timeForXGrav < 0.0f)                                             // If the gravity go against the velocity
                minP.x += 0.5f * _gravity.x * (xTimeRemaining * xTimeRemaining); // We move down pour min point (becareful! here gravity is negative)
            if (timeForYGrav < 0.0f)
                minP.y += 0.5f * _gravity.y * (yTimeRemaining * yTimeRemaining);
            if (timeForZGrav < 0.0f)
                minP.z += 0.5f * _gravity.z * (zTimeRemaining * zTimeRemaining);
        }

        //ACCELERATION (Need to rework to work like gravity)
        if (_doAcceleration && _doAccDiffDir) // Need to be rework ( for negative value)
        {
            maxP += 0.5f * _acceleration * (_timeToLive * _timeToLive); // Apply acceleration after time
        }
        else if (_doAcceleration && !_doAccDiffDir) // Need to be rework
        {
            // minP += 0.5f * _accelerationConst * (_timeToLive * _timeToLive); //Apply constant acceleration
            maxP += 0.5f * _accelerationConst*maxV * (_timeToLive * _timeToLive); // Apply constant acceleration //Not good
        }
    }
    else // If acceleration and gravity is not enable
    {

        if (!_doDirectionSpeed)
        {
            if (_velocityType == 0)
            {
                SLVec3f minVTemp = SLVec3f(min(_vRandE.x, _vRandS.x), min(_vRandE.y, _vRandS.y), min(_vRandE.z, _vRandS.z)); // Apply velocity distance after time
                SLVec3f maxVTemp = SLVec3f(max(_vRandE.x, _vRandS.x), max(_vRandE.y, _vRandS.y), max(_vRandE.z, _vRandS.z)); // Apply velocity distance after time

                if (minVTemp.x > 0 && maxVTemp.x > 0) minVTemp.x = 0.0;
                if (minVTemp.y > 0 && maxVTemp.y > 0) minVTemp.y = 0.0;
                if (minVTemp.z > 0 && maxVTemp.z > 0) minVTemp.z = 0.0;

                if (minVTemp.x < 0 && maxVTemp.x < 0) maxVTemp.x = 0.0;
                if (minVTemp.y < 0 && maxVTemp.y < 0) maxVTemp.y = 0.0;
                if (minVTemp.z < 0 && maxVTemp.z < 0) maxVTemp.z = 0.0;

                minV += minVTemp;
                maxV += maxVTemp;
            }
            else
            {
                minV += SLVec3f(_velocityConst.x, 0.0, _velocityConst.z); // Apply velocity distance after time
                maxV += SLVec3f(0.0, _velocityConst.y, 0.0);              // Apply velocity distance after time
            }
        }
        else if (_doDirectionSpeed && !_doShapeOverride)
        {
            tempSpeed = 0.0f;
            if (_doSpeedRange)
                tempSpeed = random(_speedRange.x, _speedRange.y);
            else
                tempSpeed = _speed;

            minV += SLVec3f(_direction.x, 0.0, 0.0) * tempSpeed;
            maxV += SLVec3f(0.0, _direction.y, _direction.z) * tempSpeed;

            if (_direction.x > 0.0)
            {
                maxV.x       = minV.x;
                minV.x       = 0.0;
            }
            if (_direction.y < 0.0)
            {
                minV.y       = maxV.y;
                maxV.y       = 0.0;
            }
            if (_direction.z < 0.0)
            {
                minV.z       = maxV.z;
                maxV.z       = 0.0;
            }
        }
        // Apply time to velocity
        minP += minV * _timeToLive;
        maxP += maxV * _timeToLive;
    }

    // Add size particle
    minP.x += minP.x < maxP.x ? -rW : rW; // Add size of particle
    // if (!_doSizeOverLF) minP.y += minP.y < maxP.y ? -rH : rH; // Add size of particle if we don't have size over life
    minP.y += minP.y < maxP.y ? -rH : rH; // Add size of particle if we don't have size over life
    minP.z += minP.z < maxP.z ? -rW : rW; // Add size of particle

    maxP.x += maxP.x > minP.x ? rW : -rW; // Add size of particle
    maxP.y += maxP.y > minP.y ? rH : -rH; // Add size of particle
    maxP.z += maxP.z > minP.z ? rW : -rW; // Add size of particle

    // Apply world matrix
    aabb.fromOStoWS(minP, maxP, wmNode);
}
//-----------------------------------------------------------------------------
