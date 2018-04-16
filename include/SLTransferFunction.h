//#############################################################################
//  File:      SLTransferFunction.h
//  Purpose:   Declares a transfer function functionality
//  Author:    Marcus Hudritsch
//  Date:      July 2017
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLTRANSFERFUNCTUIN_H
#define SLTRANSFERFUNCTUIN_H

#include <SL.h>
#include <SLVec3.h>
#include <SLGLTexture.h>
#include <SLEventHandler.h>

class SLScene;
class SLSceneView;

//-----------------------------------------------------------------------------
//! Predefined color lookup tables
typedef enum
{   CLUT_BW,        //!< black to white
    CLUT_WB,        //!< white to black
    CLUT_RYGCB,     //!< red to yellow to green to cyan to blue
    CLUT_BCGYR,     //!< blue to cyan to green to yellow to red
    CLUT_RYGCBK,    //!< red to yellow to green to cyan to blue to black
    CLUT_KBCGYR,    //!< black to blue to cyan to green to yellow to red
    CLUT_RYGCBM,    //!< red to yellow to green to cyan to blue to magenta
    CLUT_MBCGYR,    //!< magenta to blue to cyan to green to yellow to red
    CLUT_custom
} SLColorLUT;
//-----------------------------------------------------------------------------
//! Transfer function color point
struct SLTransferColor
{
    SLTransferColor(SLCol3f colorVal, SLfloat posVal)
        : color(colorVal), pos(posVal) {;}
    SLCol3f color;  // Transfer color RGB
    SLfloat pos;    // Transfer position (0-1)
};
typedef vector<SLTransferColor> SLVTransferColor;
//-----------------------------------------------------------------------------
//! Transfer function alpha point
struct SLTransferAlpha
{
    SLTransferAlpha(SLfloat alphaVal, SLfloat posVal)
        : alpha(alphaVal), pos(posVal) {;}
    SLfloat alpha;  // Transfer alpha (0-1)
    SLfloat pos;    // Transfer position (0-1)
};
typedef vector<SLTransferAlpha> SLVTransferAlpha;
//-----------------------------------------------------------------------------
//! A transfer function defines a 1D texture of 256 RGBA values
/*! A transfer function defines a 1D texture of 256 RGBA values that are
generated from a set of color and alpha points where all in-between points are
linearly interpolated. Such a color/alpha lookup table is used as a transfer
function e.g. in volume rendering (see the shader VolumeRenderingRayCast).
All values (RGB color, alpha and position) are values between 0-1 and are
mapped to 0-255 when the texture image of size 1x256 is generated.
*/
class SLTransferFunction: public SLGLTexture, public SLEventHandler
{
    public:
                            SLTransferFunction  (SLVTransferAlpha alphaVec,
                                                 SLColorLUT lut=CLUT_RYGCB,
                                                 SLint length = 256);
                            SLTransferFunction  (SLVTransferAlpha alphaValues,
                                                 SLVTransferColor colorValues,
                                                 SLint length = 256);

                   // Don't forget destructor to be virtual for texture deallocation
                   virtual ~SLTransferFunction  ();

        void                generateTexture     ();

        // Setters
        void                colors              (SLColorLUT lut);

        // Getters
        SLint               length              () {return _length;}
        SLVTransferColor&   colors              () {return _colors;}
        SLVTransferAlpha&   alphas              () {return _alphas;}
        SLVfloat            allAlphas           ();

    private:
        SLint               _length;    //! Length of transfer function (default 256)
        SLColorLUT          _colorLUT;  //! Color LUT identifier
        SLVTransferColor    _colors;    //! vector of colors in TF
        SLVTransferAlpha    _alphas;    //! vector of alphas in TF
};
//-----------------------------------------------------------------------------
#endif
