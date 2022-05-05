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
//! SLParticleSystem ctor with a given vector of points
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

    _timeToLive    = timeToLive;
    _amount = amount;
    _vRandS = velocityRandomStart;
    _vRandE = velocityRandomEnd;

    P.resize(1); // To trick parent class

    emitterPos(particleEmiPos);

    _textureFirst    = texC;
    _textureFlipbook = texFlipbook;

    // Initialize the drawing:
    SLMaterial* mDraw = new SLMaterial(assetMgr, "Drawing-Material", this, texC);
    mat(mDraw);

    _updateTime.init(60, 0.0f);
}
//-----------------------------------------------------------------------------
void SLParticleSystem::initMat(SLAssetManager* am, SLGLTexture* texC)
{
    
}
//-----------------------------------------------------------------------------
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

    // Math.cbrt is cube root // Without only surface
    float c = cbrt(u);

    return SLVec3f(x1 * c, x2 * c, x3 * c);
}
//-----------------------------------------------------------------------------
SLVec3f SLParticleSystem::getPointInBox(SLVec3f boxScale)
{
    float x = random(-boxScale.x, boxScale.x);
    float y = random(-boxScale.y, boxScale.y);
    float z = random(-boxScale.z, boxScale.z);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
SLVec3f SLParticleSystem::getPointInCone()
{
    float y = random(0.0f, _heightCone); //NEED TO HAVE MORE value near 1 when we have smaller base that top
    float radius = _radiusCone + tan(_angleCone) * y;
    float r      = radius * sqrt(random(0.0f, 1.0f));
    float theta        = random(0.0f, 1.0f) * 2 * PI;
    float x            = r * cos(theta);
    float z            = r * sin(theta);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
SLVec3f SLParticleSystem::getPointInPyramid()
{
    float y      = random(0.0f, _heightPyramid);
    float radius = _halfSidePyramid + tan(_anglePyramid) * y;
    float x      = random(-radius, radius);
    float z      = random(-radius, radius);

    return SLVec3f(x, y, z);
}
//-----------------------------------------------------------------------------
void SLParticleSystem::generate()
{
    unsigned                   seed = chrono::system_clock::now().time_since_epoch().count();
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
    if (_doAcc || _doGravity)
        tempInitV.resize(_amount);
    if (_doRot)
        tempR.resize(_amount);
    if (_doRot && _doRotRange)
        tempAngulareVelo.resize(_amount);
    if (_doFlipBookTexture)
        tempTexNum.resize(_amount);
    if (_doShape)
        tempInitP.resize(_amount);


    //Normal generation
    for (unsigned int i = 0; i < _amount; i++)
    {
        if (_doShape && _shapeType == 0)
            tempP[i] = getPointInSphere(_radiusSphere, SLVec3f(distribution(generator), distribution(generator), distribution(generator)));
        else if (_doShape && _shapeType == 1)
            tempP[i] = getPointInBox(_scaleBox);
        else if (_doShape && _shapeType == 2)
            tempP[i] = getPointInCone();
        else if (_doShape && _shapeType == 3)
            tempP[i] = getPointInPyramid();
        else
            tempP[i] = _emitterPos;
        if (_velocityType == 0)
        {
            tempV[i].x = random(_vRandS.x, _vRandE.x); // Random value for x velocity
            tempV[i].y = random(_vRandS.y, _vRandE.y); // Random value for y velocity
            tempV[i].z = random(_vRandS.z, _vRandE.z); // Random value for z velocity
        }
        else if (_velocityType == 1)
        {
            tempV[i].x = _velocityConst.x;             // Constant value for x velocity
            tempV[i].y = _velocityConst.y;               // Constant value for y velocity
            tempV[i].z = _velocityConst.z;               // Constant value for z velocity
        }


        tempST[i]  = GlobalTimer::timeS() + (i * (_timeToLive / _amount)); // When the first particle dies the last one begin to live
        if (_doAcc || _doGravity)
            tempInitV[i] = tempV[i];
        if (_doRot)
            tempR[i] = random(0.0f * DEG2RAD, 360.0f * DEG2RAD); // Start rotation of the particle
        if (_doRot && _doRotRange)
            tempAngulareVelo[i] = random(_angularVelocityRange.x * DEG2RAD, _angularVelocityRange.y * DEG2RAD); // Start rotation of the particle
        if (_doFlipBookTexture)
            tempTexNum[i] = random(0, _row * _col - 1);
        if (_doShape)
            tempInitP[i] = tempP[i];
    }

    // Need to have two VAo for transform feedback swapping
    _vao1.deleteGL();
    _vao2.deleteGL();

    _vao1.setAttrib(AT_position, AT_position, &tempP);
    _vao1.setAttrib(AT_velocity, AT_velocity, &tempV);
    _vao1.setAttrib(AT_startTime, AT_startTime, &tempST);
    if (_doAcc || _doGravity)
        _vao1.setAttrib(AT_initialVelocity, AT_initialVelocity, &tempInitV);
    if (_doRot)
        _vao1.setAttrib(AT_rotation, AT_rotation, &tempR);
    if (_doRot && _doRotRange)
        _vao1.setAttrib(AT_angularVelo, AT_angularVelo, &tempAngulareVelo);
    if (_doFlipBookTexture)
        _vao1.setAttrib(AT_texNum, AT_texNum, &tempTexNum);
    if (_doShape)
        _vao1.setAttrib(AT_initialPosition, AT_initialPosition, &tempInitP);
    _vao1.generateTF((SLuint)tempP.size());

    _vao2.setAttrib(AT_position, AT_position, &tempP);
    _vao2.setAttrib(AT_velocity, AT_velocity, &tempV);
    _vao2.setAttrib(AT_startTime, AT_startTime, &tempST);
    if (_doAcc || _doGravity)
        _vao2.setAttrib(AT_initialVelocity, AT_initialVelocity, &tempInitV);
    if (_doRot)
        _vao2.setAttrib(AT_rotation, AT_rotation, &tempR);
    if (_doRot && _doRotRange)
        _vao2.setAttrib(AT_angularVelo, AT_angularVelo, &tempAngulareVelo);
    if (_doFlipBookTexture)
        _vao2.setAttrib(AT_texNum, AT_texNum, &tempTexNum);
    if (_doShape)
        _vao2.setAttrib(AT_initialPosition, AT_initialPosition, &tempInitP);
    _vao2.generateTF((SLuint)tempP.size());

}
//-----------------------------------------------------------------------------
/*!
Generate Bernstein Polynomial with 4 controls points.
ContP contains 2 and 3 controls points
StatEnd contains 1 and 4 controls points
*/
void SLParticleSystem::generateBernsteinPAlpha()
{
    // For Y bezier curve
    // T^3
    float *ContP         = _bezierControlPointAlpha;
    float* StaEnd       = _bezierStartEndPointAlpha;
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
Generate Bernstein Polynomial with 4 controls points.
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
void SLParticleSystem::notVisibleFrustrumCulling()
{
    if (_isViFrustrumCulling)
    {
        _isViFrustrumCulling = false;
        _notVisibleTimeS     = GlobalTimer::timeS();
    }
}
//-----------------------------------------------------------------------------
void SLParticleSystem::draw(SLSceneView* sv, SLNode* node)
{
    /////////////////////////////
    // Init particles vector and init VAO
    /////////////////////////////
    if (!_isGenerated) {
        emitterPos(node->translationWS()); //To init first position
        generate();
        _isGenerated = true;
    }
    

    /////////////////////////////
    // Generate programs
    /////////////////////////////

    _mat->generateProgramPS();

    /////////////////////////////
    // UPDATING
    /////////////////////////////

    // Now use the updating program
    SLGLProgram* sp = _mat->programTF();
    sp->useProgram();

    /////////////////////////////
    // Apply Uniform Variables
    /////////////////////////////

    _startUpdateTimeMS = GlobalTimer::timeMS();

    if (!_isViFrustrumCulling)
    {
        _isViFrustrumCulling = true;
        sp->uniform1f("u_difTime", GlobalTimer::timeS() - _notVisibleTimeS);
        sp->uniform1f("u_deltaTime", _deltaTimeUpdateS); // Last delta time, maybe add later average deltatime (because maybe bug when fast not visible long time, visible, not visible, visisble
        _notVisibleTimeS = 0.0f;
    }
    else
    {
        sp->uniform1f("u_difTime", 0.0f);
        sp->uniform1f("u_deltaTime", GlobalTimer::timeS() - _startUpdateTimeS);
    }
    //sp->uniform1f("u_deltaTime", GlobalTimer::timeS() - _startUpdateTimeS);

    // Calculate the elapsed time for the updating
    _deltaTimeUpdateS = GlobalTimer::timeS() - _startUpdateTimeS;
    _startUpdateTimeS = GlobalTimer::timeS();

    sp->uniform1f("u_time", _startUpdateTimeS);

    if (_doAcc)
    {
        if (_doAccDiffDir)
            sp->uniform3f("u_acceleration", _acc.x, _acc.y, _acc.z);
        else
            sp->uniform1f("u_accConst", _accConst);
    }
    if (_doGravity)
        sp->uniform3f("u_gravity", _gravity.x, _gravity.y, _gravity.z);

    sp->uniform1f("u_tTL", _timeToLive);

    emitterPos(node->translationWS());

    // Worldspace
    if (_doWorldSpace)
    {
        sp->uniform3f("u_pGPosition", _emitterPos.x, _emitterPos.y, _emitterPos.z);
    }
    else
    {
        sp->uniform3f("u_pGPosition", 0.0, 0.0, 0.0);
    }

    // Flipbook
    if (_doFlipBookTexture)
    {
        sp->uniform1i("u_col", _col);
        sp->uniform1i("u_row", _row);
        _lastUpdateFB += _deltaTimeUpdateS;
        if (_lastUpdateFB > (1.0f / _frameRateFB))
        { // Last time FB was updated is bigger than the time needed for each update
            sp->uniform1i("u_condFB", 1);
            _lastUpdateFB = 0.0f;
        }
        else
        {
            sp->uniform1i("u_condFB", 0);
        }
    }

    //Rotation
    if (_doRot && !_doRotRange)
        sp->uniform1f("u_angularVelo", _angularVelocityConst * DEG2RAD);

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

    /////////////////////////////
    // DRAWING
    /////////////////////////////

    // Give uniform for drawing and find for linking vao vbo
    SLGLProgram* spD = _mat->program();
    spD->useProgram();
    SLGLState* stateGL = SLGLState::instance();

    // Billboard type
    // World space
    if (_doWorldSpace)
    {
        spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    }
    else
    {
        if (_billboardType == 1)
        {
            SLMat4f mvMat = stateGL->modelViewMatrix;

            mvMat.m(0, 1.0f);
            mvMat.m(1, 0.0f);
            mvMat.m(2, 0.0f);

            mvMat.m(8, 0.0f);
            mvMat.m(9, 0.0f);
            mvMat.m(10, 1.0f);

            spD->uniformMatrix4fv("u_vYawPMatrix",
                                  1,
                                  (SLfloat*)&mvMat); // TO change for custom shader ganeration
        }
        else
        {
            spD->uniformMatrix4fv("u_vOmvMatrix",
                                  1,
                                  (SLfloat*)&stateGL->modelViewMatrix); // TO change for custom shader ganeration
        }
    }
    // Alpha over life bezier curve
    if (_doAlphaOverLCurve)
    {
        spD->uniform4f("u_al_bernstein",
                       _bernsteinPYAlpha.x,
                       _bernsteinPYAlpha.y,
                       _bernsteinPYAlpha.z,
                       _bernsteinPYAlpha.w);
    }
    // Size over life bezier curve
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

    spD->uniform1f("u_time", GlobalTimer::timeS());
    spD->uniform1f("u_tTL", _timeToLive);

    spD->uniform1f("u_scale", _scale);
    spD->uniform1f("u_radiusW", _radiusW);
    spD->uniform1f("u_radiusH", _radiusH);

    spD->uniform1f("u_oneOverGamma", 1.0f);

    if (_doColor && _doBlendingBrigh)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE);
    SLMesh::draw(sv, node);
    if (_doColor && _doBlendingBrigh)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _drawBuf = 1 - _drawBuf;
}
//-----------------------------------------------------------------------------
//NEEEEED TO BE ENHANCE AND CLEAN
void SLParticleSystem::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    // Radius of particle
    float rW = _radiusW * _scale;
    float rH = _radiusH * _scale;

    minP = SLVec3f();
    maxP = SLVec3f();

    SLVec3f minV = SLVec3f();
    SLVec3f maxV = SLVec3f();

    //Shape
    if (_doShape) {
        if (_shapeType == 0) {
            float radius = cbrt(_radiusSphere);
            minP         = SLVec3f(-radius, -radius, -radius);
            maxP         = SLVec3f(radius, radius, radius);
        }
        if (_shapeType == 1)
        {
            minP = SLVec3f(-_scaleBox.x, -_scaleBox.y, -_scaleBox.z);
            maxP = _scaleBox;
        }
        if (_shapeType == 2)
        {
            float radius = _radiusCone + tan(_angleCone) * _heightCone;
            minP         = SLVec3f(-radius, -0.0, -radius);
            maxP         = SLVec3f(radius, _heightCone, radius);
        }
        if (_shapeType == 3)
        {
           
            float radius = _halfSidePyramid + tan(_anglePyramid) * _heightPyramid;
            minP  = SLVec3f(-radius, -0.0, -radius);
            maxP  = SLVec3f(radius, _heightPyramid, radius);
        }
    }

    // Here calculate minP maxP
    if (_doAcc || _doGravity)
    {

        // Decide which one is the minV and maxV
        if (_velocityType == 0)
        {
            if (_vRandS.x < _vRandE.x)
            {
                maxV.x = _vRandE.x;
                minV.x = _vRandS.x;
            }
            else
            {
                maxV.x = _vRandS.x;
                minV.x = _vRandE.x;
            }
            if (_vRandS.y < _vRandE.y)
            {
                maxV.y = _vRandE.y;
                minV.y = 0;
            }
            else
            {
                maxV.y = _vRandS.y;
                minV.y = 0;
            }
            if (_vRandS.z < _vRandE.z)
            {
                maxV.z = _vRandE.z;
                minV.z = _vRandS.z;
            }
            else
            {
                maxV.z = _vRandS.z;
                minV.z = _vRandE.z;
            }

            // Inverse if acceleration is negative
            if (_acc.x < 0.0)
            {
                float temp = minV.x;
                minV.x     = maxV.x;
                maxV.x     = temp;
            }
            if (_acc.y < 0.0)
            {
                float temp = minV.y;
                minV.y     = maxV.y;
                maxV.y     = temp;
            }
            if (_acc.z < 0.0)
            {
                float temp = minV.z;
                minV.z     = maxV.z;
                maxV.z     = temp;
            }
        }
        else
        {
            minV = SLVec3f(_velocityConst.x, 0.0, _velocityConst.z);
            maxV = SLVec3f(0.0, _velocityConst.y, 0.0);
        }
        //Apply velocity
        minP += minV * _timeToLive;
        maxP += maxV * _timeToLive;

        //Time to have a velocity of 0
        float timeForXGrav = 0.0f;
        float timeForYGrav = 0.0f;
        float timeForZGrav = 0.0f;
        if (_gravity.x != 0.0f) timeForXGrav = maxV.x / _gravity.x;
        if (_gravity.y != 0.0f) timeForYGrav = maxV.y / _gravity.y;
        if (_gravity.z != 0.0f) timeForZGrav = maxV.z / _gravity.z;
        if (_doGravity)
        {
            if (timeForXGrav < 0.0f)
                maxP.x -= maxV.x * (_timeToLive + timeForXGrav);
            else if (timeForXGrav > 0.0f)
                maxP.x += 0.5f * _gravity.x * (_timeToLive * _timeToLive);
            if (timeForYGrav < 0.0f)
                maxP.y -= maxV.y * (_timeToLive + timeForYGrav);
            else if (timeForYGrav > 0.0f)
                maxP.y += 0.5f * _gravity.y * (_timeToLive * _timeToLive);
            if (timeForZGrav < 0.0f)
                maxP.z -= maxV.z * (_timeToLive + timeForZGrav);
            else if (timeForZGrav > 0.0f)
                maxP.z += 0.5f * _gravity.z * (_timeToLive * _timeToLive);
            float xTimeRemaining = _timeToLive - abs(timeForXGrav);
            float yTimeRemaining = _timeToLive - abs(timeForYGrav);
            float zTimeRemaining = _timeToLive - abs(timeForZGrav);

            if (timeForXGrav < 0.0f)
                minP.x += 0.5f * _gravity.x * (xTimeRemaining * xTimeRemaining);
            if (timeForYGrav < 0.0f)
                minP.y += 0.5f * _gravity.y * (yTimeRemaining * yTimeRemaining);
            if (timeForZGrav < 0.0f)
                minP.z += 0.5f * _gravity.z * (zTimeRemaining * zTimeRemaining);
        }

        
        if (_doAccDiffDir)
        {
            maxP += 0.5f * _acc * (_timeToLive * _timeToLive); // Apply acceleration after time
        }
        else
        {
            // minP += 0.5f * _accConst * (_timeToLive * _timeToLive); //Apply constant acceleration
            maxP += 0.5f * _accConst * (_timeToLive * _timeToLive); // Apply constant acceleration
        }
        
    }
    else
    {
        if (_velocityType == 0)
        {
            minP += SLVec3f(_vRandS.x, 0.0, _vRandS.z) * _timeToLive; // Apply velocity distance after time
            maxP += _vRandE * _timeToLive;                            // Apply velocity distance after time
        }
        else
        {
            minP += SLVec3f(_velocityConst.x, 0.0, _velocityConst.z) * _timeToLive; // Apply velocity distance after time
            maxP += SLVec3f(0.0, _velocityConst.y, 0.0) * _timeToLive;              // Apply velocity distance after time
        }
    }

    // Add size particle
    minP.x += minP.x < maxP.x ? -rW : rW;                   // Add size of particle
    if (!_doSizeOverLF) minP.y += minP.y < maxP.y ? -rH : rH; // Add size of particle if we don't have size over life
    minP.z += minP.z < maxP.z ? -rW : rW;                   // Add size of particle

    maxP.x += maxP.x > minP.x ? rW : -rW; // Add size of particle
    maxP.y += maxP.y > minP.y ? rH : -rH; // Add size of particle
    maxP.z += maxP.z > minP.z ? rW : -rW; // Add size of particle

    // Apply world matrix
    aabb.fromOStoWS(minP, maxP, wmNode);
}
//-----------------------------------------------------------------------------
