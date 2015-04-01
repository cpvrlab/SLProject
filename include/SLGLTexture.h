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
#include <SLImage.h>
#include <SLGLBuffer.h>

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
enum SLTexType 
{
    UnknownMap  // will be handeled as color maps
    ,ColorMap    //*_C.{ext}
    ,NormalMap   //*_N.{ext}
    ,HeightMap   //*_H.{ext}
    ,GlossMap    //*_G.{ext}
    ,FontMap     //*_F.glf
};
//-----------------------------------------------------------------------------
//! Texture object for OpenGL texturing
/*!      
The SLGLTexture class implements an OpenGL texture object that can is used by the 
SLMaterial class. A texture can have 1-n SLImages in the vector _images.
A simple 2D texture has just a single texture image (_images[0]). For cube maps
you will need 6 images (_images[0-5]). For 3D textures you can have as much
images of the same size than your GPU and/or CPU memory can hold.
The images are not released after the OpenGL texture creation. They may be needed
for ray tracing.
*/
class SLGLTexture : public SLObject
{
    public:        
                        //! Default contructor for fonts
                        SLGLTexture     ();
                     
                        //! ctor for 2D textures with internal image allocation
                        SLGLTexture     (SLstring   imageFilename,
                                         SLint      min_filte = GL_LINEAR_MIPMAP_LINEAR,
                                         SLint      mag_filte = GL_LINEAR,
                                         SLTexType  type = UnknownMap,
                                         SLint      wrapS = GL_REPEAT,
                                         SLint      wrapT = GL_REPEAT);

                        //! ctor for 3D texture with internal image allocation
                        SLGLTexture     (SLVstring  imageFilenames,
                                         SLint      min_filte = GL_LINEAR,
                                         SLint      mag_filte = GL_LINEAR);
                  
                        //! ctor for cube mapping with internal image allocation
                        SLGLTexture     (SLstring   imageFilenameXPos,
                                         SLstring   imageFilenameXNeg,
                                         SLstring   imageFilenameYPos,
                                         SLstring   imageFilenameYNeg,
                                         SLstring   imageFilenameZPos,
                                         SLstring   imageFilenameZNeg,
                                         SLint      min_filter = GL_LINEAR,
                                         SLint      mag_filter = GL_LINEAR,
                                         SLTexType  type = UnknownMap);      
                  
    virtual            ~SLGLTexture     ();

            void        clearData       ();
            void        build           (SLint texID=0);
            void        bindActive      (SLint texID=0);
            void        fullUpdate      ();
            void        drawSprite      (SLbool doUpdate = false);
      
            // Setters
            void        texType         (SLTexType bt)   {_texType = bt;}
            void        bumpScale       (SLfloat bs)     {_bumpScale = bs;}
      
            // Getters
            SLenum      target          (){return _target;}
            SLTexType   texType         (){return _texType;}
            SLfloat     bumpScale       (){return _bumpScale;}
            SLCol4f     getTexelf       (SLfloat s, SLfloat t);
            SLbool      hasAlpha        (){return (_images[0]->format()==GL_RGBA) || _texType==FontMap;}
            SLint       width           (){return _images[0]->width();}
            SLint       height          (){return _images[0]->height();}
            SLMat4f     tm              (){return _tm;}
            SLbool      autoCalcTM3D    (){return _autoCalcTM3D;}
      
            // Misc     
            SLTexType   detectType      (SLstring filename);  
            SLuint      closestPowerOf2 (SLuint num); 
            SLuint      nextPowerOf2    (SLuint num);
            void        build2DMipmaps  (SLint target, SLuint index);

        // Bumpmap methods
            SLVec2f     dsdt            (SLfloat s, SLfloat t); //! Returns the derivation as [s,t]
  
        // Statics
    static  SLstring    defaultPath;    //!< Default path for textures
    static  SLfloat     maxAnisotropy;  //!< max. anisotropy available

    protected:
            // loading the image files
            void        load            (SLstring filename);
                               
            SLGLState*  _stateGL;       //!< Pointer to global SLGLState instance
            SLVImage    _images;        //!< vector of SLImage pointers
            SLuint      _texName;       //!< OpenGL texture "name" (= ID)
            SLTexType   _texType;       //!< [unknown, ColorMap, NormalMap, HeightMap, GlossMap]
            SLint       _min_filter;    //!< Minification filter
            SLint       _mag_filter;    //!< Magnification filter
            SLint       _wrap_s;        //!< Wrapping in s direction
            SLint       _wrap_t;        //!< Wrapping in t direction
            SLenum      _target;        //!< texture target
            SLMat4f     _tm;            //!< texture matrix 
            SLbool      _autoCalcTM3D;  //!< flag if texture matrix should be calculated from AABB for 3D mapping     
            SLfloat     _bumpScale;     //!< Bump mapping scale factor
            SLbool      _resizeToPow2;  //!< Flag if image should be resized to n^2
            SLGLBuffer  _bufP;          //!< Sprite buffer for vertex positions
            SLGLBuffer  _bufT;          //!< Sprite buffer for vertex texcoords
            SLGLBuffer  _bufI;          //!< Sprite buffer for vertex indexes
};
//-----------------------------------------------------------------------------
//! STL vector of SLGLTexture pointers
typedef std::vector<SLGLTexture*> SLVGLTexture;
//-----------------------------------------------------------------------------
#endif
