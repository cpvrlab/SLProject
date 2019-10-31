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
#include <SLGLTexture3D.h>
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
    SLGLConetracer();
    ~SLGLConetracer();

    void    init(SLint scrW, SLint scrH);
    SLbool  render(SLSceneView* sv);
    void    renderSceneGraph(SLuint progId); // <-- this could ev. be private
    void    voxelize();
    void    renderNode(SLNode* node, const SLuint progid); // <-- renders a node with all its children
    void    uploadLights(SLuint programId);
    void    visualizeVoxelization();
    void    renderConetraced();
    SLfloat diffuseConeAngle() { return _diffuseConeAngle; };
    void    diffuseConeAngle(SLfloat angle) { _diffuseConeAngle = angle; };
    SLfloat specularConeAngle() { return _specularConeAngle; };
    void    specularConeAngle(SLfloat angle) { _specularConeAngle = angle; };

    void    setCameraOrthographic();
    SLfloat lightMeshSize() { return _lightMeshSize; };
    void    lightMeshSize(SLfloat size) { _lightMeshSize = size; };
    SLfloat shadowConeAngle() { return _shadowConeAngle; };
    void    shadowConeAngle(SLfloat angle) { _shadowConeAngle = angle; };
    void    toggleVoxelVisualization() { _voxelVisualize = !_voxelVisualize; }
    SLbool  voxelVisualization() { return _voxelVisualize; }
    void    toggleDirectIllum() { _directIllum = !_directIllum; }
    SLbool  directIllum() { return _directIllum; }
    void    toggleDiffuseIllum() { _diffuseIllum = !_diffuseIllum; }
    SLbool  diffuseIllum() { return _diffuseIllum; }
    void    toggleSpecIllumination() { _specularIllum = !_specularIllum; }
    SLbool  specularIllum() { return _specularIllum; }
    void    toggleShadows() { _shadows = !_shadows; }
    SLbool  shadows() { return _shadows; }
    SLfloat gamma() { return _gamma; };
    void    gamma(SLfloat gamma) { _gamma = gamma; };

protected:
    SLSceneView*   _sv;
    SLCamera*      _cam;
    SLMaterial*    _voxelizeMat;
    SLMaterial*    _worldMat;
    SLMaterial*    _visualizeMat;
    SLMaterial*    _conetraceMat;
    SLGLTexture3D* _voxelTex;
    SLuint         _voxelTexSize = 64; // power of 2
    SLGLFbo*       _visualizeFBO1;
    SLGLFbo*       _visualizeFBO2;
    SLRectangle*   _quadMesh;
    SLBox*         _cubeMesh;

private:
    void    uploadRenderSettings(SLuint progId);
    void    calcWS2VoxelSpaceTransform();
    void    voxelSpaceTransform(const SLfloat l,
                                const SLfloat r,
                                const SLfloat b,
                                const SLfloat t,
                                const SLfloat n,
                                const SLfloat f);
    SLfloat oneOverGamma() { return (1.0f / _gamma); }

    SLfloat      _gamma = 2.2f;
    SLProjection _oldProjection;
    SLEyeType    _oldET;
    SLbool       _first             = true;
    SLfloat      _diffuseConeAngle  = 0.5f;
    SLfloat      _specularConeAngle = 0.01f;
    SLfloat      _shadowConeAngle   = 0.f;
    SLfloat      _lightMeshSize     = 2.7f;

    SLbool _voxelVisualize = false;
    SLbool _directIllum    = true;
    SLbool _diffuseIllum   = true;
    SLbool _specularIllum  = true;
    SLbool _shadows        = true;

    SLMat4f* _wsToVoxelSpace = new SLMat4f();
};
//-----------------------------------------------------------------------------
#endif SLCONETRACER_H
