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

//-----------------------------------------------------------------------------
void SLParticleSystem::initMat(SLAssetManager* am, SLGLTexture* texC)
{
    // Initialize the updating:
    SLMaterial* mUpdate = new SLMaterial(am, "Update-Material", this);

    // Initialize the drawing:
    SLMaterial* mDraw = new SLMaterial(am, "Drawing-Material", texC, this);
    
    
    mat(mDraw);
    matOut(mUpdate);
}
//-----------------------------------------------------------------------------
float SLParticleSystem::randomFloat(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff   = b - a;
    float r      = random * diff;
    return a + r;
}

int SLParticleSystem::randomInt(int min, int max)
{
    int n         = max - min + 1;
    int remainder = RAND_MAX % n;
    int x;
    do {
        x = rand();
    } while (x >= RAND_MAX - remainder);
    return min + x % n;
}

//-----------------------------------------------------------------------------
//! SLParticleSystem ctor with a given vector of points
SLParticleSystem::SLParticleSystem(SLAssetManager* assetMgr,
                   const SLint     amount,
                   const SLVec3f& particleEmiPos,
                   const SLVec3f& velocityRandomStart,
                   const SLVec3f& velocityRandomEnd,
                   const SLfloat& timeToLive,
                   SLGLTexture* texC,
                   SLGLTexture* texFlipbook,
                   const SLstring& name) : SLMesh(assetMgr, name)
{
    assert(!name.empty());

    _primitive = PT_points;

    if (amount > UINT_MAX) // Need to change for number of floats
        SL_EXIT_MSG("SLParticleSystem supports max. 2^32 vertices.");


    _ttl = timeToLive;
    _amount = amount;

    _vRandS = velocityRandomStart;
    _vRandE = velocityRandomEnd;

    P.resize(_amount);
 
    pEPos(particleEmiPos);

    _textureFirst = texC;
    _textureFlipbook = texFlipbook;
    initMat(assetMgr, texC);

    _updateTime.init(60, 0.0f);

}

//TODO Delete VAO2 with function DeleteData of Mesh.h


void SLParticleSystem::generateVAO(SLGLVertexArray& vao)
{
    vao.setAttrib(AT_position, AT_position, &P);
    vao.setAttrib(AT_velocity, AT_velocity, &V);
    vao.setAttrib(AT_startTime, AT_startTime, &ST);
    if (_acc)
        vao.setAttrib(AT_initialVelocity, AT_initialVelocity, &InitV);
    if (_rot)
        vao.setAttrib(AT_rotation, AT_rotation, &R);
    if (_flipBookTexture)
        vao.setAttrib(AT_texNum, AT_texNum, &TexNum);

    //Need to have two VAo for transform feedback swapping
    vao.generateTF((SLuint)P.size());
}

void SLParticleSystem::regenerate()
{
    P.resize(_amount);
    V.resize(_amount);
    ST.resize(_amount);
    InitV.resize(_amount);
    R.resize(_amount);
    TexNum.resize(_amount);

    for (unsigned int i = 0; i < _amount; i++)
    {
        P[i]     = _pEPos;
        V[i].x   = randomFloat(_vRandS.x, _vRandE.x);                       // Random value for x velocity
        V[i].y   = randomFloat(_vRandS.y, _vRandE.y);                       // Random value for y velocity
        V[i].z   = randomFloat(_vRandS.z, _vRandE.z);                       // Random value for z velocity
        ST[i]    = GlobalTimer::timeS() + (i * (_ttl / _amount));           // When the first particle dies the last one begin to live
        InitV[i] = V[i];
        R[i]     = randomFloat(0.0f, 360.0f); // Start rotation of the particle
        TexNum[i] = randomInt(0, _row * _col - 1);
    }

    _vao1.deleteGL();
    _vao2.deleteGL();

}
/*!
Generate Bernstein Polynome with 4 controls points.
ContP contains 2 and 3 controls points
StatEnd contains 1 and 4 controls points
*/
void SLParticleSystem::generateBernsteinPAlpha(float ContP[4], float StaEnd[4])
{
    //For Y bezier curve
    //T^3
    _bernsteinPYAlpha.x = -StaEnd[1] + ContP[1] * 3 - ContP[3] * 3 + StaEnd[3];
    //T^2
    _bernsteinPYAlpha.y = StaEnd[1] * 3 - ContP[1] * 6 + ContP[3] * 3;
    //T
    _bernsteinPYAlpha.z = -StaEnd[1] * 3 + ContP[1] * 3;
    //1
    _bernsteinPYAlpha.w = StaEnd[1];
}

/*!
Generate Bernstein Polynome with 4 controls points.
ContP contains 2 and 3 controls points
StatEnd contains 1 and 4 controls points
*/
void SLParticleSystem::generateBernsteinPSize(float ContP[4], float StaEnd[4])
{
    //For Y bezier curve
    //T^3
    _bernsteinPYSize.x = -StaEnd[1] + ContP[1] * 3 - ContP[3] * 3 + StaEnd[3];
    //T^2
    _bernsteinPYSize.y = StaEnd[1] * 3 - ContP[1] * 6 + ContP[3] * 3;
    //T
    _bernsteinPYSize.z = -StaEnd[1] * 3 + ContP[1] * 3;
    //1
    _bernsteinPYSize.w = StaEnd[1];
}


void SLParticleSystem::changeTexture()
{
    if (_flipBookTexture) {
        mat()->removeTextureType(TT_diffuse);
        mat()->addTexture(_textureFlipbook);
    }
    else
    {
        mat()->removeTextureType(TT_diffuse);
        mat()->addTexture(_textureFirst);
    }
}

void SLParticleSystem::notVisibleFrustrumCulling()
{
    if (_isViFrustrumCulling) {
        _isViFrustrumCulling = false;
        _notVisibleTimeS     = GlobalTimer::timeS();
    }
}




void SLParticleSystem::draw(SLSceneView* sv, SLNode* node)
{
    /////////////////////////////
    // Init particles vector
    /////////////////////////////

    if (ST.size()==0) {
        regenerate();
    }

    /////////////////////////////
    // Init VAO
    /////////////////////////////

    //Do updating
    if (!_vao1.vaoID())
        generateVAO(_vao1);
    if (!_vao2.vaoID())
        generateVAO(_vao2);

    /////////////////////////////
    // UPDATING
    /////////////////////////////

    // Now use the updating material
    _matOut->generateProgramPS();
    SLGLProgram* sp = _matOut->program();
    sp->useProgram();

    /////////////////////////////
    // Apply Uniform Variables
    /////////////////////////////
    _startUpdateTimeMS = GlobalTimer::timeMS();

    if (!_isViFrustrumCulling) {
        _isViFrustrumCulling = true;
        sp->uniform1f("u_difTime", GlobalTimer::timeS() - _notVisibleTimeS);
        sp->uniform1f("u_deltaTime", _deltaTimeUpdateS); // Last delta time, maybe add later average deltatime (because maybe bug when fast not visible long time, visible, not visible, visisble
    }
    else
    {
        sp->uniform1f("u_difTime", 0.0f);
        sp->uniform1f("u_deltaTime", GlobalTimer::timeS() - _startUpdateTimeS);
    }


    // Calculate the elapsed time for the updating
    _deltaTimeUpdateS  = GlobalTimer::timeS() - _startUpdateTimeS;
    _startUpdateTimeS = GlobalTimer::timeS();

    sp->uniform1f("u_time", _startUpdateTimeS);

    
    if (_acc) {
        if (_accDiffDir)
            sp->uniform3f("u_acceleration", _accV.x, _accV.y, _accV.z);
        else
            sp->uniform1f("u_accConst", _accConst);
    }
    
    sp->uniform1f("u_tTL", _ttl);

    pEPos(node->translationWS());

    // Worldspace
    if (_worldSpace) {
        sp->uniform3f("u_pGPosition", _pEPos.x, _pEPos.y, _pEPos.z);
    }
    else{
        sp->uniform3f("u_pGPosition", 0.0, 0.0, 0.0);
    }
    // Flipbook
    if (_flipBookTexture){
        sp->uniform1i("u_col", _col);
        sp->uniform1i("u_row", _row);
        _lastUpdateFB += _deltaTimeUpdateS;
        if (_lastUpdateFB > (1.0f / _frameRateFB)) { // Last time FB was updated is bigger than the time needed for each update 
            sp->uniform1i("u_condFB", 1);
            _lastUpdateFB = 0.0f;
        }else
        {
            sp->uniform1i("u_condFB", 0);
        }
    }

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

    //Generate a program to draw if no one is bound
    _mat->generateProgramPS();

    //Give uniform for drawing and find for linking vao vbo
    SLGLProgram* spD = _mat->program();
    spD->useProgram();
    SLGLState* stateGL = SLGLState::instance();

    //World space
    if (_worldSpace){
        spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    }
    else{
        spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->modelViewMatrix); // TO change for custom shader ganeration
    }
    //Alpha over life bezier curve
    if (_alphaOverLFCurve) {
        spD->uniform4f("u_al_bernstein", _bernsteinPYAlpha.x, _bernsteinPYAlpha.y, _bernsteinPYAlpha.z, _bernsteinPYAlpha.w);
    }
    //Size over life bezier curve
    if (_sizeOverLFCurve)
    {
        spD->uniform4f("u_si_bernstein", _bernsteinPYSize.x, _bernsteinPYSize.y, _bernsteinPYSize.z, _bernsteinPYSize.w);
    }
    //Color over life by gradient color editor
    if (_colorOverLF) {
        spD->uniform1fv("u_colorArr", 256 * 3, _colorArr);
    }
    else{
        spD->uniform4f("u_color", _colorV.x, _colorV.y, _colorV.z, _colorV.w);
    }
    // Flipbook
    if (_flipBookTexture)
    {
        spD->uniform1i("u_col", _col);
        spD->uniform1i("u_row", _row);
    }
    
    spD->uniform1f("u_time", GlobalTimer::timeS());
    spD->uniform1f("u_tTL", _ttl);
   
    
    spD->uniform1f("u_scale", _scale);
    spD->uniform1f("u_radiusW", _radiusW);
    spD->uniform1f("u_radiusH", _radiusH);

    spD->uniform1f("u_oneOverGamma", 1.0f);

    if (_color && _blendingBrigh)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE);
    SLMesh::draw(sv, node);
    if (_color && _blendingBrigh)
        stateGL->blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _drawBuf = 1 - _drawBuf;
}
void SLParticleSystem::buildAABB(SLAABBox& aabb, const SLMat4f& wmNode)
{
    //Radius of particle
    float rW = _radiusW * _scale;
    float rH = _radiusH * _scale;

    //Here calculate minP maxP
    if (_acc)
    {
            minP = SLVec3f();
            maxP = SLVec3f();
            // Decide which one is the minP and maxP
            if (_vRandS.x < _vRandE.x)
            {
                maxP.x = _vRandE.x;
                minP.x = _vRandS.x;
            }
            else
            {
                maxP.x = _vRandS.x;
                minP.x = _vRandE.x;
            }
            if (_vRandS.y < _vRandE.y)
            {
                maxP.y = _vRandE.y;
                minP.y = 0;
            }
            else
            {
                maxP.y = _vRandS.y;
                minP.y = 0;
            }
            if (_vRandS.z < _vRandE.z)
            {
                maxP.z = _vRandE.z;
                minP.z = _vRandS.z;
            }
            else
            {
                maxP.z = _vRandS.z;
                minP.z = _vRandE.z;
            }

            // Inverse if acceleration is negative
            if (_accV.x < 0.0)
            {
                float temp = minP.x;
                minP.x     = maxP.x;
                maxP.x     = temp;
            }
            if (_accV.y < 0.0)
            {
                float temp = minP.y;
                minP.y     = maxP.y;
                maxP.y     = temp;
            }
            if (_accV.z < 0.0)
            {
                float temp = minP.z;
                minP.z     = maxP.z;
                maxP.z     = temp;
            }

            minP = minP * _ttl;
            maxP = maxP * _ttl;                   //Apply velocity distance after time
            if (_accDiffDir)
            {
                maxP += 0.5f * _accV * (_ttl * _ttl); //Apply acceleration after time
            }
            else
            {
                //minP += 0.5f * _accConst * (_ttl * _ttl); //Apply constant acceleration
                maxP += 0.5f * _accConst * (_ttl * _ttl); //Apply constant acceleration
            }
    }
    else
    {
        minP = SLVec3f(_vRandS.x, 0.0, _vRandS.z) * _ttl;                 //Apply velocity distance after time
        maxP = _vRandE * _ttl;                          //Apply velocity distance after time
    }


    //Add size particle
    minP.x += minP.x < maxP.x ? -rW : rW;                              // Add size of particle
    if (!_sizeOverLF) minP.y += minP.y < maxP.y ? -rH : rH; // Add size of particle if we don't have size over life
    minP.z += minP.z < maxP.z ? -rW : rW;                              // Add size of particle
    
    maxP.x += maxP.x > minP.x ? rW : -rW;            // Add size of particle
    maxP.y += maxP.y > minP.y ? rH : -rH;      // Add size of particle
    maxP.z += maxP.z > minP.z ? rW : -rW;            // Add size of particle
    
    // Apply world matrix
    aabb.fromOStoWS(minP, maxP, wmNode);
}
//-----------------------------------------------------------------------------
