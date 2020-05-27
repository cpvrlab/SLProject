//#############################################################################
//  File:      SLLight.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTGL_H
#define SLLIGHTGL_H

#include <SL.h>
#include <SLVec4.h>
#include "SLOptixDefinitions.h"
#include "SLOptixHelper.h"

class SLRay;
class SLNode;
class SLSceneView;
class SLShadowMap;

//-----------------------------------------------------------------------------
//! Abstract Light class for OpenGL light sources.
/*! The abstract SLLight class encapsulates an invisible light source according
to the OpenGL specification. The derivatives SLLightSpot and SLLightRect will
also derive from SLNode and can therefore be freely placed in space.
*/
class SLLight
{
public:
    explicit SLLight(SLfloat ambiPower = 0.1f,
                     SLfloat diffPower = 1.0f,
                     SLfloat specPower = 1.0f,
                     SLint   id        = -1);
    virtual ~SLLight() = default;

    virtual void setState() = 0;

    // Setters
    void id(const SLint id) { _id = id; }
    void isOn(const SLbool on) { _isOn = on; }

    //! Sets the ambient, diffuse and specular powers all with the same color
    void powers(SLfloat ambiPow,
                SLfloat diffPow,
                SLfloat specPow,
                SLCol4f ambiDiffSpecCol = SLCol4f::WHITE)
    {
        _ambientColor  = ambiDiffSpecCol;
        _diffuseColor  = ambiDiffSpecCol;
        _specularColor = ambiDiffSpecCol;
        _ambientPower  = ambiPow;
        _diffusePower  = diffPow;
        _specularPower = specPow;
    }

    //! Sets the ambient and diffuse powers with the same color
    void ambiDiffPowers(SLfloat ambiPow,
                        SLfloat diffPow,
                        SLCol4f ambiDiffCol = SLCol4f::WHITE)
    {
        _ambientColor = ambiDiffCol;
        _diffuseColor = ambiDiffCol;
        _ambientPower = ambiPow;
        _diffusePower = diffPow;
    }

    //! Sets the same color to the ambient and diffuse colors
    void ambiDiffColor(SLCol4f ambiDiffCol)
    {
        _ambientColor = ambiDiffCol;
        _diffuseColor = ambiDiffCol;
    }

    void ambientColor(const SLCol4f& ambi) { _ambientColor = ambi; }
    void ambientPower(const SLfloat ambPow) { _ambientPower = ambPow; }
    void diffuseColor(const SLCol4f& diff) { _diffuseColor = diff; }
    void diffusePower(const SLfloat diffPow) { _diffusePower = diffPow; }
    void specularColor(const SLCol4f& spec) { _specularColor = spec; }
    void specularPower(const SLfloat specPow) { _specularPower = specPow; }
    void spotExponent(const SLfloat exp) { _spotExponent = exp; }
    void spotCutOffDEG(SLfloat cutOffAngleDEG);
    void kc(SLfloat kc);
    void kl(SLfloat kl);
    void kq(SLfloat kq);
    void attenuation(const SLfloat kConstant,
                     const SLfloat kLinear,
                     const SLfloat kQuadratic)
    {
        kc(kConstant);
        kl(kLinear);
        kq(kQuadratic);
    }
    void createsShadows(SLbool createsShadows);
    void shadowMap(SLShadowMap* shadowMap) { _shadowMap = shadowMap; }
    void doesPCF(SLbool doesPCF) { _doesPCF = doesPCF; }

    // Getters
    SLint        id() const { return _id; }
    SLbool       isOn() const { return _isOn; }
    SLCol4f      ambient() { return _ambientColor * _ambientPower; } //!< return ambientColor * ambientPower
    SLCol4f      ambientColor() { return _ambientColor; }
    SLfloat      ambientPower() { return _ambientPower; }
    SLCol4f      diffuse() { return _diffuseColor * _diffusePower; } //!< return diffuseColor * diffusePower
    SLCol4f      diffuseColor() { return _diffuseColor; }
    SLfloat      diffusePower() { return _diffusePower; }
    SLCol4f      specular() { return _specularColor * _specularPower; } //!< return _specularColor * _specularPower
    SLCol4f      specularColor() { return _specularColor; }
    SLfloat      specularPower() { return _specularPower; }
    SLfloat      spotCutOffDEG() const { return _spotCutOffDEG; }
    SLfloat      spotCosCut() const { return _spotCosCutOffRAD; }
    SLfloat      spotExponent() const { return _spotExponent; }
    SLfloat      kc() const { return _kc; }
    SLfloat      kl() const { return _kl; }
    SLfloat      kq() const { return _kq; }
    SLbool       isAttenuated() const { return _isAttenuated; }
    SLfloat      attenuation(SLfloat dist) const { return 1.0f / (_kc + _kl * dist + _kq * dist * dist); }
    SLbool       createsShadows() { return _createsShadows; }
    SLShadowMap* shadowMap() { return _shadowMap; }
    SLbool       doesPCF() { return _doesPCF; }

#ifdef SL_HAS_OPTIX
    virtual ortLight optixLight(bool)
    {
        return {
          make_float4(diffuse()),
          make_float4(ambient()),
          make_float4(specular()),
          make_float3({positionWS().x, positionWS().y, positionWS().z}),
          spotCutOffDEG(),
          spotExponent(),
          spotCosCut(),
          make_float3(spotDirWS()),
          kc(),
          kl(),
          kq(),
          {1, 1},
          0.0f};
    }
#endif

    // some virtuals needed for ray tracing

    virtual SLVec4f positionWS() const           = 0;
    virtual SLVec3f spotDirWS()                  = 0;
    virtual SLfloat shadowTest(SLRay*         ray,
                               const SLVec3f& L,
                               SLfloat        lightDist,
                               SLNode*        root3D)   = 0;
    virtual SLfloat shadowTestMC(SLRay*         ray,
                                 const SLVec3f& L,
                                 SLfloat        lightDist,
                                 SLNode*        root3D) = 0;

    // create the depth buffer(s) for shadow mapping
    virtual void renderShadowMap(SLSceneView* sv, SLNode* root) = 0;

protected:
    SLint        _id;               //!< OpenGL light number (0-7)
    SLbool       _isOn;             //!< Flag if light is on or off
    SLCol4f      _ambientColor;     //!< Ambient light color (RGB 0-1)
    SLfloat      _ambientPower;     //!< Ambient light power (0-N)
    SLCol4f      _diffuseColor;     //!< Diffuse light color (RGB 0-1)
    SLfloat      _diffusePower;     //!< Diffuse light power (0-N)
    SLCol4f      _specularColor;    //!< Specular light color (RGB 0-1)
    SLfloat      _specularPower;    //!< Specular light power (0-N)
    SLfloat      _spotCutOffDEG;    //!< Half the spot cone angle
    SLfloat      _spotCosCutOffRAD; //!< cosine of spotCutoff angle
    SLfloat      _spotExponent;     //!< Spot attenuation from center to edge of cone
    SLfloat      _kc;               //!< Constant light attenuation
    SLfloat      _kl;               //!< Linear light attenuation
    SLfloat      _kq;               //!< Quadratic light attenuation
    SLbool       _isAttenuated;     //!< fast attenuation flag for ray tracing
    SLbool       _createsShadows;   //!< flag if light creates shadows or not
    SLShadowMap* _shadowMap;        //!< Used for shadow mapping
    SLbool       _doesPCF;          //!< flag if percentage-closer filtering is enabled
};
//-----------------------------------------------------------------------------
//! STL vector of light pointers
typedef vector<SLLight*> SLVLight;
//-----------------------------------------------------------------------------
#endif
