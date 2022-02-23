//#############################################################################
//  File:      SLLightDirect.h
//  Date:      July 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTDIRECT_H
#define SLLIGHTDIRECT_H

#include <SLLight.h>
#include <SLNode.h>
#include <SLRaySamples2D.h>
#include <SLTexColorLUT.h>

class SLRay;
class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
//! SLLightDirect class for a directional light source
/*!
 SLLightDirect is a node and a light that can have a sphere mesh with a line for
 its direction representation. For directional lights the position vector is in
 infinite distance. We use its homogeneous component w as zero as the
 directional light flag. The spot direction is used in the shaders for the
 light direction.\n
 If a light node is added to the scene it stays fix in the scene.\n
 If a light node is added to the camera it moves with the camera.\n
 See the scene examples for Per-Vertex-Blinn or Per-Pixel-Blinn lighting where
 all light node types are used. \n
 All light nodes inherited from SLLight work automatically together with the
 automatically generated shader in SLGLProgramGenerated.
*/
class SLLightDirect
  : public SLNode
  , public SLLight
{
public:
    SLLightDirect(SLAssetManager* assetMgr,
                  SLScene*        s,
                  SLfloat         arrowLength       = 0.5f,
                  SLbool          hasMesh           = true,
                  SLbool          doCascadedShadows = false);
    SLLightDirect(SLAssetManager* assetMgr,
                  SLScene*        s,
                  SLfloat         posx,
                  SLfloat         posy,
                  SLfloat         posz,
                  SLfloat         arrowLength       = 0.5f,
                  SLfloat         ambiPower         = 1.0f,
                  SLfloat         diffPower         = 10.0f,
                  SLfloat         specPower         = 10.0f,
                  SLbool          hasMesh           = true,
                  SLbool          doCascadedShadows = false);
    ~SLLightDirect() override;

    void    init(SLScene* s);
    bool    hitRec(SLRay* ray) override;
    void    statsRec(SLNodeStats& stats) override;
    void    drawMesh(SLSceneView* sv) override;
    SLfloat shadowTest(SLRay*         ray,
                       const SLVec3f& L,
                       SLfloat        lightDist,
                       SLNode*        root3D) override;
    SLfloat shadowTestMC(SLRay*         ray,
                         const SLVec3f& L,
                         SLfloat        lightDist,
                         SLNode*        root3D) override;
    void    createShadowMap(float   clipNear = 0.1f,
                            float   clipFar  = 20.0f,
                            SLVec2f size     = SLVec2f(8, 8),
                            SLVec2i texSize  = SLVec2i(1024, 1024)) override;
    void    createShadowMapAutoSize(SLCamera* camera,
                                    SLVec2i   texSize     = SLVec2i(1024, 1024),
                                    int       numCascades = 4) override;
    SLCol4f calculateSunLight(SLfloat standardPower);

    // Setters
    void doSunPowerAdaptation(SLbool enabled) { _doSunPowerAdaptation = enabled; }
    void sunLightPowerMin(SLfloat minPower) { _sunLightPowerMin = minPower; }
    void doCascadedShadows(bool b) { _doCascadedShadows = b; }

    // Getters
    SLfloat        radius() const { return _arrowRadius; }
    SLfloat        dirLength() const { return _arrowLength; }
    SLbool         doSunPowerAdaptation() { return _doSunPowerAdaptation; }
    SLTexColorLUT* sunLightColorLUT() { return &_sunLightColorLUT; }
    SLbool         doCascadedShadows() const override { return _doCascadedShadows; }

    // For directional lights the position vector is interpreted as a
    // direction with the homogeneous component equals zero:
    SLVec4f positionWS() const override
    {
        SLVec4f pos(updateAndGetWM().translation());
        pos.w = 0.0f;
        return pos;
    }
    SLVec3f spotDirWS() override { return forwardOS(); }
    SLCol4f ambient() override;
    SLCol4f diffuse() override;
    SLCol4f specular() override;

protected:
    void renderShadowMap(SLSceneView* sv, SLNode* root) override;

private:
    SLfloat       _arrowRadius;          //!< The sphere lights radius
    SLfloat       _arrowLength;          //!< Length of direction line
    SLbool        _doSunPowerAdaptation; //!< Flag for sun power scaling
    SLfloat       _sunLightPowerMin;     //!< Min. zenith power scale factor for sun
    SLTexColorLUT _sunLightColorLUT;     //!< Sun light color LUT
    bool          _doCascadedShadows;    //!< Cascaded shadow
};
//-----------------------------------------------------------------------------
#endif
