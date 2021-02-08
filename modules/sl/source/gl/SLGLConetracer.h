//#############################################################################
//  File:      SLGLConetracer.h
//  Author:    Stefan Thoeni
//  Date:      Sept 2018
//  Copyright: Stefan Thoeni
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLCONETRACER_H
#define SLGLCONETRACER_H

#include <SLGLTexture.h>
#include <SLMaterial.h>
#include <SLGLConetracerTex3D.h>
#include <SLGLFbo.h>
#include <SLRectangle.h>
#include <SLBox.h>

//-----------------------------------------------------------------------------
class SLScene;
class SLSceneView;
class SLCamera;
//-----------------------------------------------------------------------------
class SLGLConetracer
{
public:
    SLGLConetracer(SLstring shaderFilePath);
    ~SLGLConetracer();

    void   init(SLint scrW, SLint scrH, const SLVec3f& minWs, const SLVec3f& maxWs);
    SLbool render(SLSceneView* sv);

    // Getters
    SLfloat diffuseConeAngle() const { return _diffuseConeAngle; };
    void    diffuseConeAngle(SLfloat angle) { _diffuseConeAngle = angle; };
    SLfloat specularConeAngle() const { return _specularConeAngle; };
    void    specularConeAngle(SLfloat angle) { _specularConeAngle = angle; };
    SLfloat lightMeshSize() const { return _lightMeshSize; };
    SLfloat shadowConeAngle() const { return _shadowConeAngle; };
    SLbool  showVoxels() const { return _showVoxels; }
    SLbool  doDirectIllum() const { return _doDirectIllum; }
    SLbool  doDiffuseIllum() const { return _doDiffuseIllum; }
    SLbool  doSpecularIllum() const { return _doSpecularIllum; }
    SLbool  shadows() const { return _doShadows; }
    SLfloat gamma() const { return _gamma; };

    // Setters
    void lightMeshSize(SLfloat size) { _lightMeshSize = size; };
    void shadowConeAngle(SLfloat angle) { _shadowConeAngle = angle; };
    void toggleVoxels() { _showVoxels = !_showVoxels; }
    void toggleDirectIllum() { _doDirectIllum = !_doDirectIllum; }
    void toggleDiffuseIllum() { _doDiffuseIllum = !_doDiffuseIllum; }
    void toggleSpecIllumination() { _doSpecularIllum = !_doSpecularIllum; }
    void toggleShadows() { _doShadows = !_doShadows; }
    void gamma(SLfloat gamma) { _gamma = gamma; };

protected:
    SLSceneView*         _sv;
    SLCamera*            _cam;
    SLMaterial*          _voxelizeMat  = nullptr;
    SLMaterial*          _worldMat     = nullptr;
    SLMaterial*          _visualizeMat = nullptr;
    SLMaterial*          _conetraceMat = nullptr;
    SLGLConetracerTex3D* _voxelTex;
    SLuint               _voxelTexSize = 64; // power of 2
    SLGLFbo*             _visualizeBackfaceFBO;
    SLGLFbo*             _visualizeFrontfaceFBO;
    SLRectangle*         _quadMesh = nullptr;
    SLBox*               _cubeMesh = nullptr;

private:
    void    voxelize();
    void    visualizeVoxels();
    void    renderNode(SLNode* node, SLGLProgram* sp);
    void    renderSceneGraph(SLGLProgram* sp);
    void    uploadRenderSettings(SLGLProgram* sp);
    //void    uploadLights(SLGLProgram* sp);
    void    calcWS2VoxelSpaceTransform(const SLVec3f& minWs, const SLVec3f& maxWs);
    void    voxelSpaceTransform(SLfloat l,
                                SLfloat r,
                                SLfloat b,
                                SLfloat t,
                                SLfloat n,
                                SLfloat f);
    SLfloat oneOverGamma() const { return (1.0f / _gamma); }

    SLfloat _gamma             = 2.2f;
    SLfloat _diffuseConeAngle  = 0.5f;
    SLfloat _specularConeAngle = 0.01f;
    SLfloat _shadowConeAngle   = 0.f;
    SLfloat _lightMeshSize     = 2.7f;

    SLbool _showVoxels      = false;
    SLbool _doDirectIllum   = true;
    SLbool _doDiffuseIllum  = true;
    SLbool _doSpecularIllum = true;
    SLbool _doShadows       = true;

    //path to shader files
    SLstring _shaderFilePath;

    vector<SLGLProgram*> _programs;

    SLMat4f* _wsToVoxelSpace = new SLMat4f();
};
//-----------------------------------------------------------------------------
#endif // SLCONETRACER_H
