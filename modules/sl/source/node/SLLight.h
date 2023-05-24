//#############################################################################
//  File:      SLLight.h
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLLIGHTGL_H
#define SLLIGHTGL_H

#include <SL.h>
#include <SLVec4.h>
#include <SLShadowMap.h>

#ifdef SL_HAS_OPTIX
#    include <SLOptixDefinitions.h>
#    include <SLOptixHelper.h>
//#include <SLOptixVectorMath.h>
#endif

class SLRay;
class SLNode;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Struct for uniform buffer with std140 layout
struct lightDataStd140
{
    SLint   isOn;           //  1
    SLVec4f posWS;          //  2, 3, 4, 5
    SLVec4f posVS;          //  6, 7, 8, 9
    SLVec4f ambient;        // 10,11,12,13
    SLVec4f diffuse;        // 14,15,16,17
    SLVec4f specular;       // 18,19,20,21
    SLVec3f spotDirWS;      // 22,23,24
    SLfloat __pad25;        // 25
    SLVec3f spotDirVS;      // 26,27,28
    SLfloat __pad29;        // 29
    SLfloat spotCutoff;     // 30
    SLfloat spotCosCut;     // 31
    SLfloat spotExp;        // 32
    SLVec3f attentuation;   // 33,34,35
    SLfloat __pad36;        // 36
    SLint   doAttenuation;  // 37
    SLint   createsShadows; // 38
    SLint   doesPCF;        // 39
    SLuint  levelPCF;       // 40
    SLint   usesCubemap;    // 41
    SLMat4f space[6];       // 42-133 (92=4x4x6)
};
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

    // Setters
    void id(const SLint id) { _id = id; }
    void isOn(const SLbool on) { _isOn = on; }

    //! Sets the ambient, diffuse and specular powers all with the same color
    void powers(SLfloat        ambiPow,
                SLfloat        diffPow,
                SLfloat        specPow,
                const SLCol4f& ambiDiffSpecCol = SLCol4f::WHITE)
    {
        _ambientColor  = ambiDiffSpecCol;
        _diffuseColor  = ambiDiffSpecCol;
        _specularColor = ambiDiffSpecCol;
        _ambientPower  = ambiPow;
        _diffusePower  = diffPow;
        _specularPower = specPow;
    }

    //! Sets the ambient and diffuse powers with the same color
    void ambiDiffPowers(SLfloat        ambiPow,
                        SLfloat        diffPow,
                        const SLCol4f& ambiDiffCol = SLCol4f::WHITE)
    {
        _ambientColor = ambiDiffCol;
        _diffuseColor = ambiDiffCol;
        _ambientPower = ambiPow;
        _diffusePower = diffPow;
    }

    //! Sets the same color to the ambient and diffuse colors
    void ambiDiffColor(const SLCol4f& ambiDiffCol)
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
    void doSmoothShadows(SLbool doSS) { _doSoftShadows = doSS; }
    void smoothShadowLevel(SLuint ssLevel) { _softShadowLevel = ssLevel; }
    void shadowMinBias(SLfloat minBias) { _shadowMinBias = minBias; }
    void shadowMaxBias(SLfloat maxBias) { _shadowMaxBias = maxBias; }

    // Getters
    SLint          id() const { return _id; }
    SLbool         isOn() const { return _isOn; }
    SLCol4f        ambientColor() { return _ambientColor; }
    SLfloat        ambientPower() const { return _ambientPower; }
    SLCol4f        diffuseColor() { return _diffuseColor; }
    SLfloat        diffusePower() const { return _diffusePower; }
    SLCol4f        specularColor() { return _specularColor; }
    SLfloat        specularPower() const { return _specularPower; }
    SLfloat        spotCutOffDEG() const { return _spotCutOffDEG; }
    SLfloat        spotCosCut() const { return _spotCosCutOffRAD; }
    SLfloat        spotExponent() const { return _spotExponent; }
    SLfloat        kc() const { return _kc; }
    SLfloat        kl() const { return _kl; }
    SLfloat        kq() const { return _kq; }
    SLbool         isAttenuated() const { return _isAttenuated; }
    SLfloat        attenuation(SLfloat dist) const { return std::min(1.0f / (_kc + _kl * dist + _kq * dist * dist), 1.0f); }
    SLbool         createsShadows() const { return _createsShadows; }
    SLShadowMap*   shadowMap() { return _shadowMap; }
    SLbool         doSoftShadows() const { return _doSoftShadows; }
    SLuint         softShadowLevel() const { return _softShadowLevel; }
    SLfloat        shadowMinBias() const { return _shadowMinBias; }
    SLfloat        shadowMaxBias() const { return _shadowMaxBias; }
    virtual SLbool doCascadedShadows() const { return false; }

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

    // Virtual functions to be implemented by the inherited
    virtual SLCol4f ambient()                    = 0; //!< Return normally _ambientColor * _ambientPower
    virtual SLCol4f diffuse()                    = 0; //!< Returns normally _diffuseColor * _diffusePower
    virtual SLCol4f specular()                   = 0; //!< Returns normally _specularColor * _specularPower
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

    // Shadow Mapping functions
    virtual void createShadowMap(float   lightClipNear = 0.1f,
                                 float   lightClipFar  = 20.0f,
                                 SLVec2f size          = SLVec2f(8, 8),
                                 SLVec2i texSize       = SLVec2i(1024, 1024)) = 0;
    virtual void createShadowMapAutoSize(SLCamera* camera,
                                         SLVec2i   texSize     = SLVec2i(1024, 1024),
                                         int       numCascades = 0)           = 0;
    virtual void renderShadowMap(SLSceneView* sv, SLNode* root);

    // statics valid for overall lighting
    static SLCol4f globalAmbient;    //!< static global ambient light intensity
    static SLfloat oneOverGamma() { return 1.0f / gamma; }
    static SLfloat gamma;            //!< final output gamma value
    static SLbool  doColoredShadows; //!< flag if shadows should be displayed with colors for debugging

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
    SLbool       _doSoftShadows;    //!< flag if percentage-closer filtering for smooth shadows is enabled
    SLuint       _softShadowLevel;  //!< Radius to smoothing (1 = 3 * 3; 2 = 5 * 5; ...)
    SLfloat      _shadowMinBias;    //!< Min. bias at 0 deg. to use to prevent shadow acne
    SLfloat      _shadowMaxBias;    //!< Max. bias at 90 deg. to use to prevent shadow acne
};
//-----------------------------------------------------------------------------
//! STL vector of light pointers
typedef vector<SLLight*> SLVLight;
//-----------------------------------------------------------------------------
#endif
