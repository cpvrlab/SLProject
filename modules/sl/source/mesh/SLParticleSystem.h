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
                     const SLfloat&  amount,
                     const SLVec3f&  particleEmiPos,
                     const SLVec3f&  velocityRandomStart,
                     const SLVec3f&  velocityRandomEnd,
                     const SLfloat&  timeToLive,
                     const SLstring& name     = "Particle system",
             SLMaterial*     materialUpdate = nullptr,
             SLMaterial*     materialDraw = nullptr);

    void draw(SLSceneView* sv, SLNode* node);
    void generateVAO(SLGLVertexArray& vao);

    // Getters
    SLMaterial* matUpdate() const { return _matUpdate; }
    SLMaterial* matDraw() const { return _matDraw; }
    SLVec3f    pEPos() const { return _pEPos; }

    //Setters
    void matUpdate(SLMaterial* m) { _matUpdate = m; }
    void matDraw(SLMaterial* m) { _matDraw = m; }
    void pEPos(SLVec3f p) { _pEPos = p; }


protected:
    SLMaterial* _matUpdate;     //!< Pointer to the updating material
    SLMaterial* _matDraw;       //!< Pointer to the drawing material
    SLfloat     _ttl;           //!< Time to live of a particle
    SLVec3f     _pEPos;         //!< Position of the particle emitter
    SLGLVertexArray _vao1;      //!< First OpenGL Vertex Array Object for swapping between updating/drawing
    SLGLVertexArray _vao2;      //!< Second OpenGL Vertex Array Object for swapping between updating/drawing

private:
    float randomFloat(float a, float b);

    SLVVec3f V;     //!< Pointer to vertex velocity vector
    SLVfloat ST;    //!< Pointer to start time vector
    SLVVec3f InitV; //!< Pointer to vertex velocity vector
    SLVfloat R;     //!< Pointer to rotation vector

    int _drawBuf = 0;   //!< Boolean to switch buffer

};
//-----------------------------------------------------------------------------
#endif
