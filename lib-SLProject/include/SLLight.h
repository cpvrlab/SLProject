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

class SLNode;
class SLRay;
class SLSceneView;

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
    void ambient(const SLCol4f& ambi) { _ambient = ambi; }
    void diffuse(const SLCol4f& diff) { _diffuse = diff; }
    void specular(const SLCol4f& spec) { _specular = spec; }
    void spotExponent(const SLfloat exp) { _spotExponent = exp; }
    void spotCutOffDEG(SLfloat cutOffAngleDEG);
    void kc(SLfloat kc);
    void kl(SLfloat kl);
    void kq(SLfloat kq);
    void attenuation(const SLfloat kC,
                     const SLfloat kL,
                     const SLfloat kQ)
    {
        kc(kC);
        kl(kL);
        kq(kQ);
    }
    void createsShadows(SLbool createsShadows) { _createsShadows = createsShadows; }

    // Getters
    SLint   id() const { return _id; }
    SLbool  isOn() const { return _isOn; }
    SLCol4f ambient() { return _ambient; }
    SLCol4f diffuse() { return _diffuse; }
    SLCol4f specular() { return _specular; }
    SLfloat spotCutOffDEG() const { return _spotCutOffDEG; }
    SLfloat spotCosCut() const { return _spotCosCutOffRAD; }
    SLfloat spotExponent() const { return _spotExponent; }
    SLfloat kc() const { return _kc; }
    SLfloat kl() const { return _kl; }
    SLfloat kq() const { return _kq; }
    SLbool  isAttenuated() const { return _isAttenuated; }
    SLfloat attenuation(SLfloat dist) const { return 1.0f / (_kc + _kl * dist + _kq * dist * dist); }
    SLbool  createsShadows() { return _createsShadows; }

#ifdef SL_HAS_OPTIX
    virtual Light optixLight(bool)
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
    virtual void renderShadowMap(SLSceneView* sv, SLNode* root){};

protected:
    SLint   _id;               //!< OpenGL light number (0-7)
    SLbool  _isOn;             //!< Flag if light is on or off
    SLCol4f _ambient;          //!< Ambient light intensity Ia
    SLCol4f _diffuse;          //!< Diffuse light intensity Id
    SLCol4f _specular;         //!< Specular light intensity Is
    SLfloat _spotCutOffDEG;    //!< Half the spot cone angle
    SLfloat _spotCosCutOffRAD; //!< cosine of spotCutoff angle
    SLfloat _spotExponent;     //!< Spot attenuation from center to edge of cone
    SLfloat _kc;               //!< Constant light attenuation
    SLfloat _kl;               //!< Linear light attenuation
    SLfloat _kq;               //!< Quadratic light attenuation
    SLbool  _isAttenuated;     //!< fast attenuation flag for ray tracing
    SLbool  _createsShadows;   //!< Flag if light creates shadows or not
};
//-----------------------------------------------------------------------------
//! STL vector of light pointers
typedef vector<SLLight*> SLVLight;
//-----------------------------------------------------------------------------
#endif
