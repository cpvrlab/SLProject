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

//-----------------------------------------------------------------------------
//! SLParticleSystem ctor with a given vector of points
SLParticleSystem::SLParticleSystem(SLAssetManager* assetMgr,
                   const SLint     amount,
                   const SLVec3f& particleEmiPos,
                   const SLVec3f& velocityRandomStart,
                   const SLVec3f& velocityRandomEnd,
                   const SLfloat& timeToLive,
                   SLGLTexture* texC,
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

    P.resize(amount);
    V.resize(amount);
    ST.resize(amount);
    InitV.resize(amount);
    R.resize(amount);

    for (unsigned int i = 0; i < amount; i++)
    {
        P[i] = particleEmiPos;
        V[i].x = randomFloat(velocityRandomStart.x, velocityRandomEnd.x); // Random value for x velocity
        V[i].y = randomFloat(velocityRandomStart.y, velocityRandomEnd.y); // Random value for y velocity
        V[i].z = randomFloat(velocityRandomStart.z, velocityRandomEnd.z); // Random value for z velocity
        ST[i]  = GlobalTimer::timeS() + (i * (timeToLive / amount));     // When the first particle dies the last one begin to live
        InitV[i] = V[i];
        R[i]     = randomFloat(0.0f, 360.0f); // Start rotation of the particle
    }
    pEPos(particleEmiPos);
    //Need to add the rest

    initMat(assetMgr, texC);

}

//TODO Delete VAO2 with function DeleteData of Mesh.h


void SLParticleSystem::generateVAO(SLGLVertexArray& vao)
{
    vao.setAttrib(AT_position, AT_position, &P);
    vao.setAttrib(AT_velocity, AT_velocity, &V);
    vao.setAttrib(AT_startTime, AT_startTime, &ST);
    vao.setAttrib(AT_initialVelocity, AT_initialVelocity, &InitV);
    vao.setAttrib(AT_rotation, AT_rotation, &R);

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

    for (unsigned int i = 0; i < _amount; i++)
    {
        P[i]     = _pEPos;
        V[i].x   = randomFloat(_vRandS.x, _vRandE.x);                       // Random value for x velocity
        V[i].y   = randomFloat(_vRandS.y, _vRandE.y);                       // Random value for y velocity
        V[i].z   = randomFloat(_vRandS.z, _vRandE.z);                       // Random value for z velocity
        ST[i]    = GlobalTimer::timeS() + (i * (_ttl / _amount));           // When the first particle dies the last one begin to live
        InitV[i] = V[i];
        R[i]     = randomFloat(0.0f, 360.0f); // Start rotation of the particle
    }

    _vao1.deleteGL();
    _vao2.deleteGL();

}




void SLParticleSystem::draw(SLSceneView* sv, SLNode* node)
{
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
    sp->uniform1f("u_time", GlobalTimer::timeS());
    sp->uniform1f("u_deltaTime", sv->s()->elapsedTimeSec());
    if (_acc) {
        sp->uniform3f("u_acceleration", _accV.x, _accV.y, _accV.z);
    }
    
    sp->uniform1f("u_tTL", _ttl);

    pEPos(node->translationWS());

    if (_worldSpace) {
        sp->uniform3f("u_pGPosition", _pEPos.x, _pEPos.y, _pEPos.z);
    }
    else{
        sp->uniform3f("u_pGPosition", 0.0, 0.0, 0.0);
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

    /////////////////////////////
    // DRAWING
    /////////////////////////////

    //Generate a program to draw if no one is bound
    _mat->generateProgramPS();

    //Give uniform for drawing and find for linking vao vbo
    SLGLProgram* spD = _mat->program();
    spD->useProgram();
    SLGLState* stateGL = SLGLState::instance();

    if (_worldSpace){
        spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    }
    else{
        spD->uniformMatrix4fv("u_vOmvMatrix", 1, (SLfloat*)&stateGL->modelViewMatrix); // TO change for custom shader ganeration
    }
    
    spD->uniform1f("u_time", GlobalTimer::timeS());
    spD->uniform1f("u_tTL", _ttl);
   
    spD->uniform4f("u_color", 0.66f, 0.0f, 0.66f, 0.2f);
    spD->uniform1f("u_scale", 1.0f);
    spD->uniform1f("u_radius", 0.4f);

    spD->uniform1f("u_oneOverGamma", 1.0f);

    SLMesh::draw(sv, node);

    _drawBuf = 1 - _drawBuf;
}
//-----------------------------------------------------------------------------
