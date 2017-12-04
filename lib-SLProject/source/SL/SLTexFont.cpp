//#############################################################################
//  File:      SL/SLTexFont.cpp
//  Author:    Marcus Hudritsch, original author is Philippe Decaudin
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Philippe Decaudin - http://www.antisphere.com
//             M. Hudritsch, Berner Fachhochschule
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>           // precompiled headers
#ifdef SL_MEMLEAKDETECT       // set in SL.h for debug config only
#include <debug_new.h>        // memory leak detector
#endif

#include "SLTexFont.h"
#include "SLScene.h"

//-----------------------------------------------------------------------------
// Initialize static font pointers
SLTexFont* SLTexFont::font07 = 0;
SLTexFont* SLTexFont::font08 = 0;
SLTexFont* SLTexFont::font09 = 0;
SLTexFont* SLTexFont::font10 = 0;
SLTexFont* SLTexFont::font12 = 0;
SLTexFont* SLTexFont::font14 = 0;
SLTexFont* SLTexFont::font16 = 0;
SLTexFont* SLTexFont::font18 = 0;
SLTexFont* SLTexFont::font20 = 0;
SLTexFont* SLTexFont::font22 = 0;
SLTexFont* SLTexFont::font24 = 0;
//-----------------------------------------------------------------------------
SLTexFont::SLTexFont(SLstring fontFilename)
{
    // Init texture members
    _texType = TT_font;
    _wrap_s = GL_CLAMP_TO_EDGE;
    _wrap_t = GL_CLAMP_TO_EDGE;
    _min_filter = GL_NEAREST;
    _mag_filter = GL_NEAREST;

    for (SLint i=0; i<256; ++i)
    {   chars[i].width = 0;
        chars[i].tx1 = 0;
        chars[i].tx2 = 0;
        chars[i].ty1 = 0;
        chars[i].ty2 = 0;
    }
    charsHeight = 0;

    create(fontFilename);
}
//-----------------------------------------------------------------------------
/*!
SLTexFont::create creates the inherited texture map with the passed image file.
The bitmap image is parsed for all 224 character positions to create the
according texture coordinate.
*/
void SLTexFont::create(SLstring fontFilename)
{
    // Check the font filename with path
    if (!SLFileSystem::fileExists(fontFilename))
    {   fontFilename = defaultPathFonts + fontFilename;
        if (!SLFileSystem::fileExists(fontFilename))
        {   SLstring msg = "SLTexFont::create: File not found: " + fontFilename;
            SL_EXIT_MSG(msg.c_str());
        }
    }

    SLCVImage img;
    img.load(fontFilename, false);

    // find height of the font
    SLint x, y;
    SLint bmpW = img.cvMat().cols;
    SLint bmpH = img.cvMat().rows;
    SLuchar* bmp = img.cvMat().data;
    SLint h = 0, hh = 0;
    SLint r, NbRow = 0;

    for (y=0; y<bmpH; ++y)
    {   if (bmp[y*bmpW] == 0)
        {   if ((hh<=0 && h<=0) || (h!=hh && h>0 && hh>0))
                SL_EXIT_MSG("Cannot determine font height (check first pixel column)");
            else if (h<=0) h = hh;
            else if (hh<=0) break;
            hh = 0;
            ++NbRow;
        } else ++hh;
    }

    // find width and position of each character
    SLint w = 0;
    SLint x0[224], y0[224], x1[224], y1[224];
    SLint ch = 32;
    SLint start;
    for (r=0; r<NbRow; ++r)
    {   start = 1;
        for (x=1; x<bmpW; ++x)
        {   if (bmp[(r*(h+1)+h)*bmpW+x]==0 || x==bmpW-1)
            {   if (x==start) break;  // next row
                if (ch<256)
                {  x0[ch-32] = start;
                    x1[ch-32] = x;
                    y0[ch-32] = r*(h+1);
                    y1[ch-32] = r*(h+1)+h-1;
                    w += x-start+1;
                    start = x+1;
                }
                ++ch;
            }
        }
    }

   for (x=ch-32; x<224; ++x)
   {  x0[x] = 0; y0[x] = 0;
      x1[x] = 0; y1[x] = 0;
   }

    // Repack: build 14 rows of 16 characters. First, find the largest row
    SLint l, lmax = 1;
    for (r=0; r<14; ++r)
    {   l = 0;
        for (x=0; x<16; ++x)
            l += x1[x+r*16]-x0[x+r*16]+1;
        if (l>lmax) lmax = l;
    }

    // A little empty margin is added between chars to avoid artefact when anti aliasing is on
    const SLint MARGIN_X = 2;
    const SLint MARGIN_Y = 2;
    lmax += 16*MARGIN_X;

    // 2) build the texture
    charsHeight = h;
    SLint texWidth  = nextPowerOf2(lmax);
    SLint texHeight = nextPowerOf2(14*(h+MARGIN_Y));

    //Fill up with 0
    SLuchar* bits = new SLuchar[texWidth * texHeight];
    memset(bits, 0, texWidth * texHeight);

    SLfloat du = 0.0f;
    SLfloat dv = 0.0f;

    for (r=0; r<14; ++r)
    {   for (SLint xx=0, ch=r*16; ch<(r+1)*16; ++ch)
        {   if (y1[ch]-y0[ch]==h-1)
            {   for (y=0; y<h; ++y)
                {   for (x=x0[ch]; x<=x1[ch]; ++x)
                    {   SLfloat alpha = ((SLfloat)(bmp[x+(y0[ch]+y)*bmpW]))/256.0f;
                        //alpha = alpha*sqrtf(alpha); // powf(alpha, 1.5f);   // some gamma correction
                        bits[(xx+x-x0[ch])+(r*(h+MARGIN_Y)+y)*texWidth] = (SLuchar)(alpha*256.0f);
                    }
                }
                chars[ch+32].tx1 = ((SLfloat)xx+du)/(SLfloat)texWidth;
                xx += x1[ch]-x0[ch]+1;
                chars[ch+32].tx2 = ((SLfloat)xx+du)/(SLfloat)texWidth;
                chars[ch+32].ty1 = ((SLfloat)(r*(h+MARGIN_Y))+dv)/(SLfloat)texHeight;
                chars[ch+32].ty2 = ((SLfloat)(r*(h+MARGIN_Y)+h)+dv)/(SLfloat)texHeight;
                chars[ch+32].width = (SLfloat)(x1[ch]-x0[ch]+1);
                xx += MARGIN_X;
            }
        }
    }

    //Allocate memory for image pixels using only the alpha channel
    _images.clear();
    SLPixelFormat format = _stateGL->pixelFormatIsSupported(PF_luminance) ? PF_luminance : PF_red;
    _images.push_back(new SLCVImage(texWidth, texHeight, format, fontFilename));
    _images[0]->load(texWidth, texHeight, format, format, bits, true, false);
    delete [] bits;

    // Set characters below 32 to default
    const SLuchar Undef = 127; // default character used as for undifined ones (having ascii codes from 0 to 31)
    for (ch=0; ch<32; ++ch)
    {   chars[ch].tx1 = chars[Undef].tx1;
        chars[ch].tx2 = chars[Undef].tx2;
        chars[ch].ty1 = chars[Undef].ty1;
        chars[ch].ty2 = chars[Undef].ty2;
        chars[ch].width = chars[Undef].width;
    }
}
//-----------------------------------------------------------------------------
/*! Returns the size (width & height) of the full text in float pixels. If a
max. width is passed text is first wrapped into multiple lines. For mulitline
text the line height is calculate as the font height * lineHeightFactor.
*/
SLVec2f SLTexFont::calcTextSize(SLstring text,
                                SLfloat  maxWidth,
                                SLfloat  lineHeightFactor)
{     
    SLVec2f size(0,0);
   
    if (maxWidth > 0.0f)
    {  
        SLfloat maxX = FLT_MIN;
        SLVstring lines = wrapTextToLines(text, maxWidth);
        for (SLuint i=0; i<lines.size(); ++i)
        {   SLVec2f size = calcTextSize(lines[i]);
            if (size.x > maxX) maxX = size.x;
        }
        size.x = maxX;
        size.y = (SLfloat)(lines.size()-1) * (SLfloat)charsHeight * lineHeightFactor;
        size.y += (SLfloat)charsHeight;
    }
    else
    {  // Loop through each character of text
        for (SLuint i=0; i<text.length(); ++i)
        {   SLchar c = text[i];
            size.x += chars[c].width;
        }
        size.y = (SLfloat)charsHeight;
    }
    return size;
}
//-----------------------------------------------------------------------------
/*! 
Returns a vector of strings of the text to be wrapped to a max. with of maxW.
The sum of all characters in lines must be equal to the length of the input text 
*/
SLVstring SLTexFont::wrapTextToLines(SLstring text, // text to wrap
                                     SLfloat  maxW) // max. width in pixels               
{  
    SLVstring   lines;
    SLint       numLines = 1;
    SLfloat     curX = 0.0f;
    SLfloat     maxX = FLT_MIN;
    SLfloat     xBlank = 0.0f;
    SLuint      iBlank = 0;
    SLuint      iLineStart = 0;
    SLuint      len = (SLuint)text.length();
   
    // Loop through each character of text
    for (SLuint i=0; i<len; ++i)
    {   SLuchar c = text[i];
      
        if (c=='\\' && i<len-1 && text[i+1]=='n')
        {   i++;
            if (curX > maxX) maxX = curX;
            lines.push_back(text.substr(iLineStart, i-iLineStart-1)+"  ");
            numLines++;
            iLineStart = i+1;
            curX = 0.0f;
        }
        else // add next character
        {  
            // keep last blank x
            if (c==' ')
            {   xBlank = curX;
                iBlank = i;
            }

            curX += chars[c].width;
         
            // if width exceeded wrap at last blank position
            if (curX > maxW) 
            {  // wrap at last blank
                if (xBlank > 0.0f)
                {   // keep greatest line width
                    if (xBlank > maxX) maxX = xBlank;
                    curX = curX - xBlank - chars[' '].width;
                    lines.push_back(text.substr(iLineStart, iBlank-iLineStart+1));
                    iLineStart = iBlank + 1;
                } else // wrap in the word
                {   if (curX - chars[c].width > maxX) 
                        maxX = curX - chars[c].width;
                    lines.push_back(text.substr(iLineStart, i-iLineStart));
                    curX = chars[c].width;
                    iLineStart = i+1;
                }
                numLines++;
            }
        }
    }
    SLstring newLine = text.substr(iLineStart, len-iLineStart);
    lines.push_back(newLine);
    return lines;
}
//-----------------------------------------------------------------------------
/*! 
Builds the buffer objects bufP, bufT & bufI with 2 texture mapped triangles per 
character. The text width < maxWidth the text will be on one line. If it is 
wider it will be split into multiple lines with a 
height = font height * lineHeight.
*/
void SLTexFont::buildTextBuffers(SLGLVertexArray& vao,
                                 SLstring text,       // text
                                 SLfloat maxWidth,    // max. width for multi-line text 
                                 SLfloat lineHeight)  // line height factor
{   
    SLVstring lines;  // Vector of text lines
    SLVVec2f  sizes;  // Sizes of text lines
    SLint     numP=0; // No. of vertices
    SLint     numI=0; // No. of indices (3 per triangle)
    SLfloat   x;      // current lower-left x position
    SLfloat   y;      // current lower-left y position
    SLuint    iV;     // current vertex index
    SLuint    iI;     // current vertex index index

    // Calculate number of vertices & indices
    if (maxWidth > 0.0f)
    {   // multiple text lines
        lines = wrapTextToLines(text, maxWidth);
        for (SLuint l=0; l<lines.size(); ++l)
        {   numP += (SLint)lines[l].length();
            sizes.push_back(calcTextSize(lines[l]));
        }
        numP *= 4;
        numI = numP*2*3;
    } else
    {   // single text line
        lines.push_back(text);
        numP = (SLint)text.length()*4;
        numI = (SLint)text.length()*2*3;
    }

    SLVVec3f  P; P.resize(numP);      // Vertex positions
    SLVVec2f  T; T.resize(numP);      // Vertex texture coords.
    SLVushort I; I.resize(numI);     // Indexes

    iV = iI = 0;
    y = (lines.size()-1) * (SLfloat)charsHeight * lineHeight;
   
    for (SLuint l=0; l<lines.size(); ++l)
    {  
        x = 0;
      
        // Loop through characters
        for (SLuint i=0; i<lines[l].length(); ++i)
        {  
            SLuchar c = lines[l][i];
      
            // Get width and height
            SLfloat w = chars[c].width;
            SLfloat h = (SLfloat)charsHeight;
 
            // Specify texture coordinates
            T[iV  ].set(chars[c].tx1, chars[c].ty2);
            T[iV+1].set(chars[c].tx2, chars[c].ty2);
            T[iV+2].set(chars[c].tx2, chars[c].ty1);
            T[iV+3].set(chars[c].tx1, chars[c].ty1);
 
            // vertices of the character quad
            P[iV  ].set(x,   y  );  
            P[iV+1].set(x+w, y  );
            P[iV+2].set(x+w, y+h);
            P[iV+3].set(x,   y+h);
      
            // triangle indices of the character quad
            I[iI++]=iV;   I[iI++]=iV+1; I[iI++]=iV+3;
            I[iI++]=iV+1; I[iI++]=iV+2; I[iI++]=iV+3;
 
            // Move to next character
            iV += 4;
            x  += w;
        }

        y -= (SLfloat)charsHeight * lineHeight;
    }
      
    // create buffers on GPU
    SLGLProgram* sp = SLScene::current->programs(SP_fontTex);
    sp->useProgram();
    vao.setAttrib(AT_position, sp->getAttribLocation("a_position"), &P);
    vao.setAttrib(AT_texCoord, sp->getAttribLocation("a_texCoord"), &T);
    vao.setIndices(&I);
    vao.generate(numP);
}
//-----------------------------------------------------------------------------
//! Generates all static fonts
void SLTexFont::generateFonts()
{
    font07 = new SLTexFont("Font07.png"); assert(font07);
    font08 = new SLTexFont("Font08.png"); assert(font08);
    font09 = new SLTexFont("Font09.png"); assert(font09);
    font10 = new SLTexFont("Font10.png"); assert(font10);
    font12 = new SLTexFont("Font12.png"); assert(font12);
    font14 = new SLTexFont("Font14.png"); assert(font14);
    font16 = new SLTexFont("Font16.png"); assert(font16);
    font18 = new SLTexFont("Font18.png"); assert(font18);
    font20 = new SLTexFont("Font20.png"); assert(font20);
    font22 = new SLTexFont("Font22.png"); assert(font22);
    font24 = new SLTexFont("Font24.png"); assert(font24);
}
//-----------------------------------------------------------------------------
//! Deletes all static fonts
void SLTexFont::deleteFonts()
{
    if (font07) delete font07; font07 = 0;
    if (font08) delete font08; font08 = 0;
    if (font09) delete font09; font09 = 0;
    if (font10) delete font10; font10 = 0;
    if (font12) delete font12; font12 = 0;
    if (font14) delete font14; font14 = 0;
    if (font16) delete font16; font16 = 0;
    if (font18) delete font18; font18 = 0;
    if (font20) delete font20; font20 = 0;
    if (font22) delete font22; font22 = 0;
    if (font24) delete font24; font24 = 0;
}
//-----------------------------------------------------------------------------
//! returns nearest font for a given height in mm
SLTexFont* SLTexFont::getFont(SLfloat heightMM, SLint dpi)
{
    SLfloat dpmm = (SLfloat)dpi / 25.4f;
    SLfloat targetH_PX = dpmm * heightMM;

    if (targetH_PX <  7) return font07;
    if (targetH_PX <  8) return font08;
    if (targetH_PX <  9) return font09;
    if (targetH_PX < 10) return font10;
    if (targetH_PX < 12) return font12;
    if (targetH_PX < 14) return font14;
    if (targetH_PX < 16) return font16;
    if (targetH_PX < 18) return font18;
    if (targetH_PX < 20) return font20;
    if (targetH_PX < 24) return font22;
    return font24;
}
//-----------------------------------------------------------------------------
