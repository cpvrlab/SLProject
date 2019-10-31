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
    void    visualizeVoxels();
    SLfloat diffuseConeAngle() { return _diffuseConeAngle; };
    void    diffuseConeAngle(SLfloat angle) { _diffuseConeAngle = angle; };
    SLfloat specularConeAngle() { return _specularConeAngle; };
    void    specularConeAngle(SLfloat angle) { _specularConeAngle = angle; };

    void    setCameraOrthographic();
    SLfloat lightMeshSize() { return _lightMeshSize; };
    void    lightMeshSize(SLfloat size) { _lightMeshSize = size; };
    SLfloat shadowConeAngle() { return _shadowConeAngle; };
    void    shadowConeAngle(SLfloat angle) { _shadowConeAngle = angle; };
    SLbool  showVoxels() { return _showVoxels; }
    SLbool  doDirectIllum() { return _doDirectIllum; }
    SLbool  doDiffuseIllum() { return _doDiffuseIllum; }
    SLbool  doSpecularIllum() { return _doSpecularIllum; }
    void    toggleVoxels() { _showVoxels = !_showVoxels; }
    void    toggleDirectIllum() { _doDirectIllum = !_doDirectIllum; }
    void    toggleDiffuseIllum() { _doDiffuseIllum = !_doDiffuseIllum; }
    void    toggleSpecIllumination() { _doSpecularIllum = !_doSpecularIllum; }
    void    toggleShadows() { _doShadows = !_doShadows; }
    SLbool  shadows() { return _doShadows; }
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

    SLbool _showVoxels      = false;
    SLbool _doDirectIllum   = true;
    SLbool _doDiffuseIllum  = true;
    SLbool _doSpecularIllum = true;
    SLbool _doShadows       = true;

    SLMat4f* _wsToVoxelSpace = new SLMat4f();
};
//-----------------------------------------------------------------------------
#endif SLCONETRACER_H
