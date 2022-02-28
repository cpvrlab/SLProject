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
#include <SLSceneView.h>
#include <GlobalTimer.h>

//-----------------------------------------------------------------------------
//! Struct definition for particle attribute position, velocity, start time, initial velocity and rotation
struct Particle
{
    SLVec3f p;     // particle position [x,y,z]
    SLVec3f v;     // particle velocity [x,y,z]
    float   st;    // particle start time
    SLVec3f initV; // particle initial velocity [x,y,z]
    float   r;     // particle rotation

    Particle()
      : p(0.0f), v(0.0f), st(0.0f), initV(0.0f), r(0.0f) {}
};
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
                   const SLfloat& amount,
                   const SLVec3f& particleEmiPos,
                   const SLVec3f& velocityRandomStart,
                   const SLVec3f& velocityRandomEnd,
                   const SLfloat& timeToLive,
                   const SLstring& name,
                   SLMaterial*     materialUpdate,
                   SLMaterial*     materialDraw) : SLMesh(assetMgr, name)
{
    assert(!name.empty());

    _primitive = PT_points;

    if (amount > UINT_MAX) // Need to change for number of floats
        SL_EXIT_MSG("SLParticleSystem supports max. 2^32 vertices.");

    _ttl = timeToLive;

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
        ST[i]  = GlobalTimer::timeMS() + (i * (timeToLive / amount));     // When the first particle dies the last one begin to live
        InitV[i] = V[i];
        R[i]     = randomFloat(0.0f, 360.0f); // Start rotation of the particle
    }
    pEPos(particleEmiPos);
    //Need to add the rest

    mat(materialDraw);
    matUpdate(materialUpdate);
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

void SLParticleSystem::draw(SLSceneView* sv, SLNode* node)
{
    //Do updating
    if (!_vao1.vaoID())
        generateVAO(_vao);
    if (!_vao2.vaoID())
        generateVAO(_vao2);

    // Now use the updating material
    SLGLProgram* sp    = _matUpdate->program();
    sp->useProgram();

    sp->uniform1f("u_tTL", _ttl);
    sp->uniform3f("u_pGPosition", _pEPos.x, _pEPos.y, _pEPos.z);
    sp->uniform1f("u_deltaTime", sv->s()->elapsedTimeMS());
    sp->uniform1f("u_time", GlobalTimer::timeMS());

    if (_drawBuf == 0) {
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
        _vao = _vao2;
    }
    

    //Give uniform for drawing and find for linking vao vbo
    SLGLProgram* spD = _mat->program();
    spD->useProgram();
    
    sp->uniform1f("u_time", GlobalTimer::timeMS());
    sp->uniform1f("u_tTL", _ttl);
    sp->uniform3f("u_pGPosition", _pEPos.x, _pEPos.y, _pEPos.z);

    sp->uniform4f("u_color", 0.66f, 0.66f, 0.66f, 0.2f);
    sp->uniform1f("u_scale", 1.0f);
    sp->uniform1f("u_radius", 0.4f);

    sp->uniform1f("u_oneOverGamma", 1.0f);

    SLMesh::draw(sv, node);

    _drawBuf = 1 - _drawBuf;
}
//-----------------------------------------------------------------------------
