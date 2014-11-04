//#############################################################################
//  File:      SL/SLTexFont.h
//  Author:    Marcus Hudritsch, original author is Philippe Decaudin
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Philippe Decaudin - http://www.antisphere.com
//             M. Hudritsch, Berner Fachhochschule
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>

#include <SLGLTexture.h>
#include <SLGLBuffer.h>

#ifndef SLTEXFONT
#define SLTEXFONT

//-----------------------------------------------------------------------------
//! Texture Font class inherits SLGLTexture for alpha blended font rendering.
/*!
This texture font class was originally inspired by the the font class of the
AntTweakBar library (http://www.antisphere.com/Wiki/tools:anttweakbar)
The source bitmap includes 224 characters starting from ascii char 32 (space) to 
ascii char 255:
<PRE>
 !"#$%&'()*+,-./0123456789:;<=>?
\@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_
`abcdefghijklmnopqrstuvwxyz{|}~
€‚ƒ„…†‡ˆ‰Š‹Œ‘’“”•–—˜™š›œŸ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿
ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏĞÑÒÓÔÕÖ×ØÙÚÛÜİŞß
àáâãäåæçèéêëìíîïğñòóôõö÷øùúûüışÿ
</PRE>
First column of a source bitmap is a delimiter with color=zero at the end of 
each line of characters. Last row of a line of characters is a delimiter with 
color=zero at the last pixel of each character.
The source bitmaps for the fonts are in the folder _data/fonts. The where used
for design only. Their data is directly included a binary array in the source 
file SLTexFont.cpp.
*/
class SLTexFont : public SLGLTexture
{  
public:
                    SLTexFont        (const SLuchar *bmp, 
                                      SLint bmpW, SLint bmpH);
                  ~SLTexFont         (){;}
                    
    void           create            (const SLuchar *bmp, 
                                      SLint bmpW, SLint bmpH);
                                        
    SLVec2f        calcTextSize      (SLstring text, 
                                      SLfloat maxWidth = 0.0f, 
                                      SLfloat lineHeightFactor = 1.5f);
    SLVstring      wrapTextToLines   (SLstring text,
                                      SLfloat  maxW);
    void           buildTextBuffers  (SLGLBuffer* bufP,
                                      SLGLBuffer* bufT,
                                      SLGLBuffer* bufI,
                                      SLstring text, 
                                      SLfloat maxWidth = 0.0f,
                                      SLfloat lineHeight = 1.5f);
                                          
    //! Single Character info struct w. min. and max. texcoords.
    typedef struct
    {
        SLfloat width; //!< Width of char. in texcoord.
        SLfloat tx1;   //!< Min. Texture x-coord.         
        SLfloat ty1;   //!< Max. Texture y-coord.
        SLfloat tx2;   //!< Max. Texture x-coord. 
        SLfloat ty2;   //!< Min. Texture y-coord.
    } SLTexFontChar;

    SLTexFontChar  chars[256];       //<! array of character info structs
    SLint          charsHeight;      //<! height of characters   
                         
    // Static method & font pointers
    static void       generateFonts();
    static void       deleteFonts();
    static SLTexFont* getFont(SLfloat heightMM, SLint dpi);

    static SLTexFont* font07;                              
    static SLTexFont* font08;
    static SLTexFont* font09;
    static SLTexFont* font10;
    static SLTexFont* font12;
    static SLTexFont* font14;
    static SLTexFont* font16;
    static SLTexFont* font18;
    static SLTexFont* font20;
    static SLTexFont* font22;
};
//-----------------------------------------------------------------------------
#endif
