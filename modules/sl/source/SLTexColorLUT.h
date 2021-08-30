//#############################################################################
//  File:      SLTexColorLUT.h
//  Purpose:   Declares a color look up table functionality
//  Date:      July 2017
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCOLORLUT_H
#define SLCOLORLUT_H

#include <SL.h>
#include <SLEventHandler.h>
#include <SLGLTexture.h>
#include <SLVec3.h>

class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Predefined color lookup tables
typedef enum
{
    CLUT_BW,       //!< black to white
    CLUT_WB,       //!< white to black
    CLUT_RYGCB,    //!< red to yellow to green to cyan to blue
    CLUT_BCGYR,    //!< blue to cyan to green to yellow to red
    CLUT_RYGCBK,   //!< red to yellow to green to cyan to blue to black
    CLUT_KBCGYR,   //!< black to blue to cyan to green to yellow to red
    CLUT_RYGCBM,   //!< red to yellow to green to cyan to blue to magenta
    CLUT_MBCGYR,   //!< magenta to blue to cyan to green to yellow to red
    CLUT_DAYLIGHT, //!< daylight from sunrise to sunset with noon in the middle
    CLUT_custom    //! enum for any other custom LUT
} SLColorLUTType;
//-----------------------------------------------------------------------------
//! Color point with color and position value between 0-1
struct SLColorLUTPoint
{
    SLColorLUTPoint(SLCol3f colorVal, SLfloat posVal)
      : color(colorVal), pos(posVal) { ; }
    SLCol3f color; // Transfer color RGB
    SLfloat pos;   // Transfer position (0-1)
};
typedef vector<SLColorLUTPoint> SLVColorLUTPoint;
//-----------------------------------------------------------------------------
//! Alpha point with alpha value and position value between 0-1
struct SLAlphaLUTPoint
{
    SLAlphaLUTPoint(SLfloat alphaVal, SLfloat posVal)
      : alpha(alphaVal), pos(posVal) { ; }
    SLfloat alpha; // Transfer alpha (0-1)
    SLfloat pos;   // Transfer position (0-1)
};
typedef vector<SLAlphaLUTPoint> SLVAlphaLUTPoint;
//-----------------------------------------------------------------------------
//! SLTexColorLUT defines a lookup table as an 1D texture of (256) RGBA values
/*! SLTexColorLUT defines an RGBA color lookup table (LUT). It can be used to
 define a transfer function that are generated from a set of color and alpha
 points where all in-between points are linearly interpolated.
 Such a color/alpha lookup table is used as a transfer function e.g. in
 volume rendering (see the shader VolumeRenderingRayCast).
 All values (RGB color, alpha and position) are values between 0-1 and are
 mapped to 0-255 when the texture image of size 1x256 is generated.
*/
class SLTexColorLUT : public SLGLTexture
  , public SLEventHandler
{
public:
    SLTexColorLUT(SLAssetManager* assetMgr,
                  SLColorLUTType  lutType,
                  SLuint          length = 256);
    SLTexColorLUT(SLAssetManager*  assetMgr,
                  SLVAlphaLUTPoint alphaVec,
                  SLColorLUTType   lutType = CLUT_RYGCB,
                  SLuint           length  = 256);
    SLTexColorLUT(SLAssetManager*  assetMgr,
                  SLVAlphaLUTPoint alphaValues,
                  SLVColorLUTPoint colorValues,
                  SLuint           length = 256);

    // Don't forget destructor to be virtual for texture deallocation
    virtual ~SLTexColorLUT();

    void generateTexture();

    // Setters
    void colors(SLColorLUTType lut);

    // Getters
    SLuint            length() { return _length; }
    SLVColorLUTPoint& colors() { return _colors; }
    SLVAlphaLUTPoint& alphas() { return _alphas; }
    SLVfloat          allAlphas();

protected:
    SLuint           _length;   //! Length of transfer function (default 256)
    SLColorLUTType   _colorLUT; //! Color LUT identifier
    SLVColorLUTPoint _colors;   //! vector of colors in TF
    SLVAlphaLUTPoint _alphas;   //! vector of alphas in TF
};
//-----------------------------------------------------------------------------
#endif
