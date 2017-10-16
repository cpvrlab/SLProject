//#############################################################################
//  File:      SLGLTexture.h
//  Author:    Marcus Hudritsch, Martin Christen
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLTEXTURE_H
#define SLGLTEXTURE_H

#include <stdafx.h>
#include <SLCVImage.h>
#include <SLGLVertexArray.h>

//-----------------------------------------------------------------------------
// Special constants for anisotropic filtering
#define SL_ANISOTROPY_MAX (GL_LINEAR_MIPMAP_LINEAR + 1)
#define SL_ANISOTROPY_2   (GL_LINEAR_MIPMAP_LINEAR + 2)
#define SL_ANISOTROPY_4   (GL_LINEAR_MIPMAP_LINEAR + 4)
#define SL_ANISOTROPY_8   (GL_LINEAR_MIPMAP_LINEAR + 8)
#define SL_ANISOTROPY_16  (GL_LINEAR_MIPMAP_LINEAR + 16)
#define SL_ANISOTROPY_32  (GL_LINEAR_MIPMAP_LINEAR + 32)
//-----------------------------------------------------------------------------
// Extension constant for anisotropic filtering
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
//-----------------------------------------------------------------------------
//! Texture type enumeration & their filename appendix for auto type detection
enum SLTextureType 
{   TT_unknown,     // will be handled as color maps
    TT_color,       //*_C.{ext}
    TT_normal,      //*_N.{ext}
    TT_height,      //*_H.{ext}
    TT_gloss,       //*_G.{ext}
    TT_roughness,   //*_R.{ext} Cook-Torrance roughness 0-1
    TT_metallic,    //*_M.{ext} Cook-Torrance metallic 0-1
    TT_font         //*_F.{ext}
};
//-----------------------------------------------------------------------------
//! Texture object for OpenGL texturing
/*!      
The SLGLTexture class implements an OpenGL texture object that can be used by the 
SLMaterial class. A texture can have 1-n SLCVImages in the vector _images.
A simple 2D texture has just a single texture image (_images[0]). For cube maps
you will need 6 images (_images[0-5]). For 3D textures you can have as much
images of the same size than your GPU and/or CPU memory can hold.
The images are not released after the OpenGL texture creation. They may be needed
for ray tracing.
*/
class SLGLTexture : public SLObject
{
    public:        
                            //! Default constructor for fonts
                            SLGLTexture         ();

                            //! ctor for 1D texture with internal image allocation
                            SLGLTexture         (SLVCol4f       colors,
                                                 SLint          min_filter = GL_LINEAR,
                                                 SLint          mag_filter = GL_LINEAR,
                                                 SLint          wrapS = GL_REPEAT,
                                                 SLstring       name = "1D-Texture");

                            //! ctor for 2D textures with internal image allocation
                            SLGLTexture         (SLstring       imageFilename,
                                                 SLint          min_filte = GL_LINEAR_MIPMAP_LINEAR,
                                                 SLint          mag_filte = GL_LINEAR,
                                                 SLTextureType  type = TT_unknown,
                                                 SLint          wrapS = GL_REPEAT,
                                                 SLint          wrapT = GL_REPEAT);

                            //! ctor for 3D texture with internal image allocation
                            SLGLTexture         (SLVstring      imageFilenames,
                                                 SLint          min_filte = GL_LINEAR,
                                                 SLint          mag_filte = GL_LINEAR,
                                                 SLint          wrapS = GL_REPEAT,
                                                 SLint          wrapT = GL_REPEAT,
                                                 SLstring       name = "3D-Texture",
                                                 SLbool         loadGrayscaleIntoAlpha = false);
                  
                            //! ctor for cube mapping with internal image allocation
                            SLGLTexture         (SLstring       imageFilenameXPos,
                                                 SLstring       imageFilenameXNeg,
                                                 SLstring       imageFilenameYPos,
                                                 SLstring       imageFilenameYNeg,
                                                 SLstring       imageFilenameZPos,
                                                 SLstring       imageFilenameZNeg,
                                                 SLint          min_filter = GL_LINEAR,
                                                 SLint          mag_filter = GL_LINEAR,
                                                 SLTextureType  type = TT_unknown);

    virtual                ~SLGLTexture         ();

            void            clearData           ();
            void            build               (SLint texID=0);
            void            bindActive          (SLint texID=0);
            void            fullUpdate          ();
            void            drawSprite          (SLbool doUpdate = false);

            // Setters
            void            texType             (SLTextureType bt)  {_texType = bt;}
            void            bumpScale           (SLfloat bs) {_bumpScale = bs;}
            void            minFiler            (SLint minF) {_min_filter = minF;} // must be called befor build
            void            magFiler            (SLint magF) {_mag_filter = magF;} // must be called befor build

            // Getters
            SLCVVImage&     images              (){return _images;}
            SLenum          target              (){return _target;}
            SLuint          texName             (){return _texName;}
            SLTextureType   texType             (){return _texType;}
            SLfloat         bumpScale           (){return _bumpScale;}
            SLCol4f         getTexelf           (SLfloat s, SLfloat t);
            SLbool          hasAlpha            (){return (_images.size() &&
                                                          ((_images[0]->format()==PF_rgba  ||
                                                            _images[0]->format()==PF_bgra) ||
                                                           _texType==TT_font));}
            SLint           width               (){return _images[0]->width();}
            SLint           height              (){return _images[0]->height();}
            SLint           depth               (){return (SLint)_images.size();}
            SLMat4f         tm                  (){return _tm;}
            SLbool          autoCalcTM3D        (){return _autoCalcTM3D;}
            SLbool          needsUpdate         (){return _needsUpdate;}
            SLstring        typeName            ();

            // Misc
            SLTextureType   detectType          (SLstring filename);
            SLuint          closestPowerOf2     (SLuint num);
            SLuint          nextPowerOf2        (SLuint num);
            void            build2DMipmaps      (SLint target, SLuint index);
            void            setVideoImage       (SLstring videoImageFile);
            SLbool          copyVideoImage      (SLint camWidth,
                                                 SLint camHeight,
                                                 SLPixelFormat glFormat,
                                                 SLuchar* data,
                                                 SLbool isContinuous,
                                                 SLbool isTopLeft);
            void            calc3DGradients     (SLint sampleRadius);
            void            smooth3DGradients   (SLint smoothRadius);

            // Bumpmap methods
            SLVec2f         dsdt                (SLfloat s, SLfloat t); //! Returns the derivation as [s,t]
  
    // Statics
    static  SLstring        defaultPath;        //!< Default path for textures
    static  SLstring        defaultPathFonts;   //!< Default path for fonts images
    static  SLfloat         maxAnisotropy;      //!< max. anisotropy available
    static  SLuint          numBytesInTextures; //!< NO. of texture bytes on GPU

    protected:
            // loading the image files
            void            load            (SLstring filename,
                                             SLbool flipVertical = true,
                                             SLbool loadGrayscaleIntoAlpha = false);
            void            load            (const SLVCol4f& colors);
                               
            SLGLState*      _stateGL;        //!< Pointer to global SLGLState instance
            SLCVVImage      _images;         //!< vector of SLCVImage pointers
            SLuint          _texName;        //!< OpenGL texture "name" (= ID)
            SLTextureType   _texType;        //!< [unknown, ColorMap, NormalMap, HeightMap, GlossMap]
            SLint           _min_filter;     //!< Minification filter
            SLint           _mag_filter;     //!< Magnification filter
            SLint           _wrap_s;         //!< Wrapping in s direction
            SLint           _wrap_t;         //!< Wrapping in t direction
            SLenum          _target;         //!< texture target
            SLMat4f         _tm;             //!< texture matrix
            SLuint          _bytesOnGPU;     //!< NO. of bytes on GPU
            SLbool          _autoCalcTM3D;   //!< flag if texture matrix should be calculated from AABB for 3D mapping
            SLfloat         _bumpScale;      //!< Bump mapping scale factor
            SLbool          _resizeToPow2;   //!< Flag if image should be resized to n^2
            SLGLVertexArray _vaoSprite;      //!< Vertex array object for sprite rendering
            atomic<bool>    _needsUpdate;    //!< Flag if image needs an update
};
//-----------------------------------------------------------------------------
//! STL vector of SLGLTexture pointers
typedef std::vector<SLGLTexture*> SLVGLTexture;
//-----------------------------------------------------------------------------
#endif
