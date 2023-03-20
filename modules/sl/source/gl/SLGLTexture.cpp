//#############################################################################
//   File:      SLGLTexture.cpp
//   Date:      July 2014
//   Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//   Authors:   Marcus Hudritsch
//   License:   This software is provided under the GNU General Public License
//              Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLGLState.h>
#include <SLGLTexture.h>
#include <SLScene.h>
#include <SLGLProgramManager.h>
#include <SLAssetManager.h>
#include "SLFileStorage.h"
#include <Utils.h>
#include <Profiler.h>

#ifdef SL_HAS_OPTIX
#    include <cuda.h>
#    include <cudaGL.h>
#    include <SLOptix.h>
#    include <SLOptixHelper.h>
#    include <SLOptixRaytracer.h>
#endif

//! maxAnisotropy=-1 show that GL_EXT_texture_filter_anisotropic is not checked
SLfloat SLGLTexture::maxAnisotropy = -1.0f;

//! NO. of texture byte allocated on GPU
SLuint SLGLTexture::totalNumBytesOnGPU = 0;
//-----------------------------------------------------------------------------
/*! Default ctor for all stack instances such as the video textures in SLScene
or the textures inherited by SLRaytracer. All other constructors add the this
pointer to the SLScene::_texture vector for global deallocation.
*/
SLGLTexture::SLGLTexture()
{
    _texID                 = 0;
    _texType               = TT_unknown;
    _width                 = 0;
    _height                = 0;
    _depth                 = 0;
    _uvIndex               = 0;
    _bytesPerPixel         = 0;
    _min_filter            = GL_NEAREST;
    _mag_filter            = GL_NEAREST;
    _wrap_s                = GL_REPEAT;
    _wrap_t                = GL_REPEAT;
    _target                = GL_TEXTURE_2D;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = false;
    _bytesOnGPU            = 0;
    _bytesInFile           = 0;
    _needsUpdate           = false;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif
}
//-----------------------------------------------------------------------------
/*!
 Constructor for empty 2D textures.
 Textures can be used in multiple materials. Textures can belong therefore
 to the global assets such as meshes (SLMesh), materials (SLMaterial),
 textures (SLGLTexture) and shader programs (SLGLProgram).
 @param assetMgr Pointer to a global asset manager. If passed the asset
 manager is the owner of the instance and will do the deallocation. If a
 nullptr is passed the creator is responsible for the deallocation.
 @param min_filter Minification filter constant from OpenGL
 @param mag_filter Magnification filter constant from OpenGL
 @param wrapS Texture wrapping in S direction (OpenGL constant)
 @param wrapT Texture wrapping in T direction (OpenGL constant)
 @param target Texture target GL_TEXTURE_1D, 2D, 3D
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLint           wrapS,
                         SLint           wrapT,
                         SLenum          target)
{
    _width                 = 0;
    _height                = 0;
    _depth                 = 0;
    _bytesPerPixel         = 0;
    _uvIndex               = 0;
    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = wrapS;
    _wrap_t                = wrapT;
    _target                = target;
    _texID                 = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = false;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _bytesInFile           = 0;
    _texType               = TT_unknown;
    _deleteImageAfterBuild = false;

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for 2D texture with a passed image data pointer.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param data Data pointer to the first top-left pixel
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param cvtype OpenCV image type
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param type Type of the texture
 * @param wrapS Texture wrapping in S direction (OpenGL constant)
 * @param wrapT Texture wrapping in T direction (OpenGL constant)
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         unsigned char*  data,
                         int             width,
                         int             height,
                         int             cvtype,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLTextureType   type,
                         SLint           wrapS,
                         SLint           wrapT)
{

    CVImage* image = new CVImage();
    image->load(width, height, PF_red, PF_red, data, true, false);
    _images.push_back(image);

    _width                 = (SLint)image->width();
    _height                = (SLint)image->height();
    _depth                 = (SLint)_images.size();
    _uvIndex               = 0;
    _bytesPerPixel         = (SLint)image->bytesPerPixel();
    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = wrapS;
    _wrap_t                = wrapT;
    _target                = GL_TEXTURE_2D;
    _texID                 = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = false;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _bytesInFile           = 0;
    _texType               = type;
    _deleteImageAfterBuild = false;

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for 2D textures from image file with internal image allocation.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param filename Name of the texture image file. If only a filename is
 * passed it will be search on the SLGLTexture::defaultPath.
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param type Type of the texture
 * @param wrapS Texture wrapping in S direction (OpenGL constant)
 * @param wrapT Texture wrapping in T direction (OpenGL constant)
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         const SLstring& filename,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLTextureType   type,
                         SLint           wrapS,
                         SLint           wrapT)
  : SLObject(Utils::getFileName(filename), filename)
{
    assert(!filename.empty());

    _texType = type == TT_unknown ? detectType(filename) : type;

    load(filename);

    if (!_images.empty())
    {
        _width         = _images[0]->width();
        _height        = _images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = _images[0]->bytesPerPixel();
        _bytesInFile   = _images[0]->bytesInFile();
    }
    else if (_compressedTexture)
    {
#ifdef SL_BUILD_WITH_KTX
        // todo ktx: get properties and extract necessary
        _width       = _ktxTexture->baseWidth;
        _height      = _ktxTexture->baseHeight;
        _depth       = _ktxTexture->numDimensions == 3 ? _ktxTexture->baseDepth : 1;
        _bytesInFile = Utils::getFileSize(filename);
#endif
    }

    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = wrapS;
    _wrap_t                = wrapT;
    _target                = GL_TEXTURE_2D;
    _texID                 = 0;
    _uvIndex               = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = false;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for 3D textures from image files with internal image allocation.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param files Vector of texture image files. If only filenames are
 * passed they will be searched on the SLGLTexture::defaultPath.
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param wrapS Texture wrapping in S direction (OpenGL constant)
 * @param wrapT Texture wrapping in T direction (OpenGL constant)
 * @param name Name of the 3D texture
 * @param loadGrayscaleIntoAlpha Flag if grayscale image should be loaded into
 * alpha channel.
 */
SLGLTexture::SLGLTexture(SLAssetManager*  assetMgr,
                         const SLVstring& files,
                         SLint            min_filter,
                         SLint            mag_filter,
                         SLint            wrapS,
                         SLint            wrapT,
                         const SLstring&  name,
                         SLbool           loadGrayscaleIntoAlpha) : SLObject(name)
{
    assert(files.size() > 1);

    _texType = TT_diffuse;

    for (const auto& filename : files)
        load(filename, true, loadGrayscaleIntoAlpha);

    if (!_images.empty())
    {
        _width         = _images[0]->width();
        _height        = _images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = _images[0]->bytesPerPixel();
    }

    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = wrapS;
    _wrap_t                = wrapT;
    _target                = GL_TEXTURE_3D;
    _texID                 = 0;
    _uvIndex               = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = true;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for 3D textures from single image file that is stacked depth times.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param depth Depth of 3D texture.
 * @param filename texture image file. If only filenames are
 * passed they will be searched on the SLGLTexture::defaultPath.
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param wrapS Texture wrapping in S direction (OpenGL constant)
 * @param wrapT Texture wrapping in T direction (OpenGL constant)
 * @param name Name of the 3D texture
 * @param loadGrayscaleIntoAlpha Flag if grayscale image should be loaded into
 * alpha channel.
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         SLint           depth,
                         const SLstring& filename,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLint           wrapS,
                         SLint           wrapT,
                         const SLstring& name,
                         SLbool          loadGrayscaleIntoAlpha) : SLObject(name)
{
    assert(depth > 1);

    _texType = TT_diffuse;

    for (SLint i = 0; i < depth; ++i)
        load(filename, true, loadGrayscaleIntoAlpha);

    if (!_images.empty())
    {
        _width         = (SLint)_images[0]->width();
        _height        = (SLint)_images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = (SLint)_images[0]->bytesPerPixel();
    }

    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = wrapS;
    _wrap_t                = wrapT;
    _target                = GL_TEXTURE_3D;
    _texID                 = 0;
    _uvIndex               = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = true;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for 1D texture from a color vector.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param colors Vector of colors
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param wrapS Texture wrapping in S direction (OpenGL constant)
 * @param name Name of the 1D texture
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         const SLVCol4f& colors,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLint           wrapS,
                         const SLstring& name) : SLObject(name)
{
    assert(colors.size() > 1);

    _texType = TT_diffuse;

    load(colors);

    if (!_images.empty())
    {
        _width         = (SLint)_images[0]->width();
        _height        = (SLint)_images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = (SLint)_images[0]->bytesPerPixel();
    }

    _min_filter = min_filter;
    _mag_filter = mag_filter;
    _wrap_s     = wrapS;
    _wrap_t     = wrapS;

    // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture
    _target = GL_TEXTURE_2D;

    _texID                 = 0;
    _uvIndex               = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = true;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif

    // Add pointer to the global resource vectors for deallocation
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * Constructor for a cubemap texture from 6 image files.
 * Textures can be used in multiple materials. Textures can belong therefore
 * to the global assets such as meshes (SLMesh), materials (SLMaterial),
 * textures (SLGLTexture) and shader programs (SLGLProgram).
 * @param assetMgr Pointer to a global asset manager. If passed the asset
 * manager is the owner of the instance and will do the deallocation. If a
 * nullptr is passed the creator is responsible for the deallocation.
 * @param filenameXPos Filename of the cubemap image in the pos. X direction.
 * @param filenameXNeg Filename of the cubemap image in the neg. X direction.
 * @param filenameYPos Filename of the cubemap image in the pos. Y direction.
 * @param filenameYNeg Filename of the cubemap image in the neg. Y direction.
 * @param filenameZPos Filename of the cubemap image in the pos. Z direction.
 * @param filenameZNeg Filename of the cubemap image in the neg. Z direction.
 * @param min_filter Minification filter constant from OpenGL
 * @param mag_filter Magnification filter constant from OpenGL
 * @param type Texture Type
 */
SLGLTexture::SLGLTexture(SLAssetManager* assetMgr,
                         const SLstring& filenameXPos,
                         const SLstring& filenameXNeg,
                         const SLstring& filenameYPos,
                         const SLstring& filenameYNeg,
                         const SLstring& filenameZPos,
                         const SLstring& filenameZNeg,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLTextureType   type) : SLObject(filenameXPos)
{
    _texType = type == TT_unknown ? detectType(filenameXPos) : type;

    assert(!filenameXPos.empty());
    assert(!filenameXNeg.empty());
    assert(!filenameYPos.empty());
    assert(!filenameYNeg.empty());
    assert(!filenameZPos.empty());
    assert(!filenameZNeg.empty());

    load(filenameXPos, false);
    load(filenameXNeg, false);
    load(filenameYPos, false);
    load(filenameYNeg, false);
    load(filenameZPos, false);
    load(filenameZNeg, false);

    if (!_images.empty())
    {
        _width         = (SLint)_images[0]->width();
        _height        = (SLint)_images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = (SLint)_images[0]->bytesPerPixel();
    }

    _min_filter            = min_filter;
    _mag_filter            = mag_filter;
    _wrap_s                = GL_CLAMP_TO_EDGE; // other you will see filter artefacts on the edges
    _wrap_t                = GL_CLAMP_TO_EDGE; // other you will see filter artefacts on the edges
    _target                = GL_TEXTURE_CUBE_MAP;
    _texID                 = 0;
    _uvIndex               = 0;
    _bumpScale             = 1.0f;
    _resizeToPow2          = false;
    _autoCalcTM3D          = false;
    _needsUpdate           = false;
    _bytesOnGPU            = 0;
    _deleteImageAfterBuild = false;

#ifdef SL_HAS_OPTIX
    _cudaGraphicsResource = nullptr;
    _cudaTextureObject    = 0;
#endif
    if (assetMgr)
        assetMgr->textures().push_back(this);
}
//-----------------------------------------------------------------------------
/*!
 * The destructor should be called by the owner of the texture. If an asset
 * manager was passed in the constructor it will do it after scene destruction.
 * The destructor deletes all images in the RAM as well as the texture objects
 * on the GPU.
 */
SLGLTexture::~SLGLTexture()
{
    // SL_LOG("~SLGLTexture(%s)", name().c_str());
    deleteData();
}
//-----------------------------------------------------------------------------
//! Delete all data (CVImages and GPU textures)
void SLGLTexture::deleteData()
{
    deleteImages();
    deleteDataGpu();

#ifdef SL_BUILD_WITH_KTX
    if (_ktxTexture)
        ktxTexture_Destroy((ktxTexture*)_ktxTexture);
#endif

    _texID                 = 0;
    _texType               = TT_unknown;
    _width                 = 0;
    _height                = 0;
    _depth                 = 0;
    _bytesPerPixel         = 0;
    _deleteImageAfterBuild = false;
}
//-----------------------------------------------------------------------------
//! Deletes the CVImages in _images. No more texture mapping in ray tracing.
void SLGLTexture::deleteImages()
{
    for (auto& img : _images)
    {
        delete img;
        img = nullptr;
    }
    _images.clear();
}
//-----------------------------------------------------------------------------
//! Deletes the OpenGL texture objects and releases the memory on the GPU
void SLGLTexture::deleteDataGpu()
{
    glDeleteTextures(1, &_texID);
    _texID = 0;
    totalNumBytesOnGPU -= _bytesOnGPU;
    _bytesOnGPU = 0;
    _vaoSprite.clearAttribs();

#ifdef SL_HAS_OPTIX
    if (_cudaGraphicsResource)
    {
        CUDA_CHECK(cuGraphicsUnregisterResource(_cudaGraphicsResource));
        _cudaGraphicsResource = nullptr;
    }
#endif
}
//-----------------------------------------------------------------------------
//! Loads the texture, converts color depth & applies vertical mirroring
void SLGLTexture::load(const SLstring& filename,
                       SLbool          flipVertical,
                       SLbool          loadGrayscaleIntoAlpha)
{
    if (!SLFileStorage::exists(filename, IOK_image))
    {
        SLstring msg = "SLGLTexture: File not found: " + filename;
        SL_EXIT_MSG(msg.c_str());
    }

    string ext = Utils::getFileExt(filename);

    // Check for compressed texture in KTX2 (Khronos Texture) format
    if (ext == "ktx2")
    {
#ifdef SL_BUILD_WITH_KTX
        _ktxFileName       = filename;
        _compressedTexture = true;

        SLIOBuffer     buffer = SLFileStorage::readIntoBuffer(filename, IOK_image);
        KTX_error_code error  = ktxTexture_CreateFromMemory(buffer.data,
                                                           buffer.size,
                                                           KTX_TEXTURE_CREATE_NO_FLAGS,
                                                           (ktxTexture**)&_ktxTexture);
        // KTX apparently takes ownership of the data, so no deallocation

        if (error != KTX_SUCCESS)
        {
            string errStr = "Error in SLGLTexture::load: " +
                            ktxErrorStr(error) + " in file: " + filename;
            SL_EXIT_MSG(errStr.c_str());
        }

#    ifdef SL_EMSCRIPTEN
        // WebGL doesn't support generating mipmaps for compressed textures
        _ktxTexture->generateMipmaps = false;
#    endif

        if (ktxTexture2_NeedsTranscoding(_ktxTexture))
        {
#    if 0
            // NOTE(dgj1): The following lines are for reading out supported compressed texture formats 
            // from the openGL driver. I have the suspicion that the enabled compression format (KTX_TTF_ETC2_RGBA)
            // is not supported on the tested device and that the reason the textures are not displayed on
            // these devices is this. However, this could not be verified.
            GLint numCompressedTextureFormats = 0;
            glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &numCompressedTextureFormats);
            std::vector<GLint> compressedTextureFormats(numCompressedTextureFormats, 0);
            glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, &compressedTextureFormats[0]);
#    endif

#    if defined(SL_OS_MACIOS)
            _compressionFormat = KTX_TTF_PVRTC1_4_RGB;
#    elif defined(SL_OS_ANDROID)
            _compressionFormat = KTX_TTF_ETC2_RGBA;
#    elif defined(SL_EMSCRIPTEN)
            // The executable might run on desktop or on mobile, so we have to check
            // the available compression extensions to pick the right format.
            if (SLGLState::instance()->hasExtension("WEBGL_compressed_texture_s3tc"))
                _compressionFormat = KTX_TTF_BC3_RGBA;
            if (SLGLState::instance()->hasExtension("WEBGL_compressed_texture_etc"))
                _compressionFormat = KTX_TTF_ETC2_RGBA;
            else
                SL_EXIT_MSG("No valid compression format found for this WebGL context");
#    else
            _compressionFormat = KTX_TTF_BC3_RGBA;
#    endif
            error = ktxTexture2_TranscodeBasis(_ktxTexture, _compressionFormat, 0);

            if (error != KTX_SUCCESS || _ktxTexture->pData == nullptr)
            {
                string errStr = "Error in SLGLTexture::load: " +
                                ktxErrorStr(error) +
                                "\nwhile transcoding file: " + filename +
                                "\nto format: " + compressionFormatStr(_compressionFormat);
                SL_EXIT_MSG(errStr.c_str());
            }
        }
#else
        SL_EXIT_MSG("Ktx files are not supported. You have to build with SL_BUILD_WITH_KTX flag enabled.");
#endif
    }
    else
    {
        CVImage* image = new CVImage(filename,
                                     flipVertical,
                                     loadGrayscaleIntoAlpha);
        _images.push_back(image);
    }
}
//-----------------------------------------------------------------------------
//! Loads the 1D color data into an image of height 1
void SLGLTexture::load(const SLVCol4f& colors)
{
    assert(colors.size() > 1);

    // convert to CV color vector
    CVVVec4f col4f;
    for (const auto& c : colors)
        col4f.push_back(CVVec4f(c.r, c.g, c.b, c.a));

    CVImage* image = new CVImage(col4f);
    _images.push_back(image);
}
//-----------------------------------------------------------------------------
//! Copies the image data from a video camera into the current video image
/*!
@brief SLGLTexture::copyVideoImage
@param camWidth Width in pixels of the camera image
@param camHeight Height in pixels of the camera image
@param srcFormat Pixel format according to the OpenGL pixel formats
@param data Pointer to the first byte of the first pixel
@param isContinuous Flag if the next line comes after the last byte of the prev. line
@param isTopLeft Flag if the data pointer points to the top left pixel
@return Returns true if the texture was rebuilt
It is important that passed pixel format is either PF_LUMINANCE, RGB or RGBA.
otherwise an expensive conversion must be done.
*/
SLbool SLGLTexture::copyVideoImage(SLint           camWidth,
                                   SLint           camHeight,
                                   CVPixelFormatGL srcFormat,
                                   SLuchar*        data,
                                   SLbool          isContinuous,
                                   SLbool          isTopLeft)
{
    PROFILE_FUNCTION();

    CVPixelFormatGL pixelFormat = PF_rgb;

    // Add image for the first time
    if (_images.empty())
        _images.push_back(new CVImage(camWidth,
                                      camHeight,
                                      pixelFormat,
                                      "LiveVideoImageFromMemory"));

    // load returns true if size or format changes
    bool needsBuild = _images[0]->load(camWidth,
                                       camHeight,
                                       srcFormat,
                                       pixelFormat,
                                       data,
                                       isContinuous,
                                       isTopLeft);

    if (!_images.empty())
    {
        _width         = (SLint)_images[0]->width();
        _height        = (SLint)_images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = (SLint)_images[0]->bytesPerPixel();
    }

    // OpenGL ES 2 only can resize non-power-of-two texture with clamp to edge
    _wrap_s = GL_CLAMP_TO_EDGE;
    _wrap_t = GL_CLAMP_TO_EDGE;

    if (needsBuild || _texID == 0)
    {
        SL_LOG("SLGLTexture::copyVideoImage: Rebuild: %d, %s",
               _texID,
               _images[0]->name().c_str());
        build(0);
    }

    _needsUpdate = true;
    return needsBuild;
}

SLbool SLGLTexture::copyVideoImage(SLint           camWidth,
                                   SLint           camHeight,
                                   CVPixelFormatGL srcFormat,
                                   CVPixelFormatGL dstFormat,
                                   SLuchar*        data,
                                   SLbool          isContinuous,
                                   SLbool          isTopLeft)
{
    PROFILE_FUNCTION();

    // Add image for the first time
    if (_images.empty())
        _images.push_back(new CVImage(camWidth,
                                      camHeight,
                                      dstFormat,
                                      "LiveVideoImageFromMemory"));

    // load returns true if size or format changes
    bool needsBuild = _images[0]->load(camWidth,
                                       camHeight,
                                       srcFormat,
                                       dstFormat,
                                       data,
                                       isContinuous,
                                       isTopLeft);
    if (!_images.empty())
    {
        _width         = (int)_images[0]->width();
        _height        = (int)_images[0]->height();
        _depth         = (SLint)_images.size();
        _bytesPerPixel = (int)_images[0]->bytesPerPixel();
    }

    // OpenGL ES 2 only can resize non-power-of-two texture with clamp to edge
    _wrap_s = GL_CLAMP_TO_EDGE;
    _wrap_t = GL_CLAMP_TO_EDGE;

    if (needsBuild || _texID == 0)
    {
        SL_LOG("SLGLTexture::copyVideoImage: Rebuild: %d, %s",
               _texID,
               _images[0]->name().c_str());
        build(0);
    }

    _needsUpdate = true;
    return needsBuild;
}

//-----------------------------------------------------------------------------
/*!
Builds an OpenGL texture object with the according OpenGL commands.
This texture creation must be done only once when a valid OpenGL rendering
context is present. This function is called the first time within the enable
method which is called by object that uses the texture.
*/
void SLGLTexture::build(SLint texUnit)
{
    PROFILE_FUNCTION();

    assert(texUnit >= 0 && texUnit < 16);

    if (_compressedTexture)
    {
#ifdef SL_BUILD_WITH_KTX
        if (_ktxTexture->pData == nullptr)
        {
            string errStr = "Error in SLGLTexture::build: texture " + _ktxFileName + " contains no data";
            SL_WARN_MSG(errStr.c_str());
        }

        // delete texture name if it already exits
        if (_texID)
        {
            glBindTexture(_target, _texID);
            glDeleteTextures(1, &_texID);
            glBindTexture(_target, 0);
            _texID = 0;
            totalNumBytesOnGPU -= _bytesOnGPU;
        }

        // get max texture size
        SLint texMaxSize = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texMaxSize);

        // todo ktx: compare with image size
        //  check 2D size
        if (_target == GL_TEXTURE_2D)
        {
            if (_width > texMaxSize)
                SL_EXIT_MSG("SLGLTexture::build: Texture width is too big.");
            if (_height > texMaxSize)
                SL_EXIT_MSG("SLGLTexture::build: Texture height is too big.");
        }

        // todo ktx: cubemaps and 3d textures

        // todo: upload in build process
        GLenum glerror = 0;
        glGenTextures(1, &_texID); // Optional. GLUpload can generate a texture.

#    if 0
        glBindTexture(_target, _texID);

        glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, _min_filter);
        GET_GL_ERROR;

        // apply magnification filter only GL_NEAREST & GL_LINEAR is allowed
        glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, _mag_filter);
        GET_GL_ERROR;

        // apply texture wrapping modes
        glTexParameteri(_target, GL_TEXTURE_WRAP_S, _wrap_s);
        glTexParameteri(_target, GL_TEXTURE_WRAP_T, _wrap_t);
        glTexParameteri(_target, GL_TEXTURE_WRAP_R, _wrap_t);
        GET_GL_ERROR;
#    endif

        KTX_error_code ktxErrorCode = ktxTexture_GLUpload((ktxTexture*)_ktxTexture, &_texID, &_target, &glerror);
        GET_GL_ERROR;

        _ktxTexture->baseHeight;

        _bytesOnGPU += (SLuint)_ktxTexture->dataSize;
        totalNumBytesOnGPU += _bytesOnGPU;

        // todo: destroy somewhere else
        if (_deleteImageAfterBuild)
        {
            ktxTexture_Destroy((ktxTexture*)_ktxTexture);
            _ktxTexture = nullptr;
        }
        GET_GL_ERROR;
#endif
    }
#ifdef SL_OS_ANDROID
    else if (_target == GL_TEXTURE_EXTERNAL_OES)
    {
        glGenTextures(1, &_texID);

        SLGLState* stateGL = SLGLState::instance();
        stateGL->activeTexture(GL_TEXTURE0 + (SLuint)texUnit);

        // create binding and apply texture properties
        stateGL->bindTexture(GL_TEXTURE_EXTERNAL_OES, _texID);

        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // apply texture wrapping modes
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, _wrap_s);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, _wrap_t);
        // ATTENTION: setting the color to black or white does not give correct results (dont know why)
        // colors different to black and white seem to work. Default value is {0, 0, 0, 0}.
        float color[] = {0.00001f, 0.00001f, 0.00001f, 1.0f};
        glTexParameterfv(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_BORDER_COLOR_OES, color);
        GET_GL_ERROR;
    }
#endif
    else
    {
        if (_images.empty())
            SL_EXIT_MSG("No images loaded in SLGLTexture::build");

        // delete texture name if it already exits
        if (_texID)
        {
            glBindTexture(_target, _texID);
            glDeleteTextures(1, &_texID);
            SL_LOG("SLGLTexture::build: Deleted: %d, %s",
                   _texID,
                   _images[0]->name().c_str());
            glBindTexture(_target, 0);
            _texID = 0;
            totalNumBytesOnGPU -= _bytesOnGPU;
            _bytesOnGPU = 0;
        }

        // get max texture size
        SLint texMaxSize = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texMaxSize);

        // check if texture has to be resized
        if (_resizeToPow2)
        {
            SLuint w2 = Utils::closestPowerOf2(_images[0]->width());
            SLuint h2 = Utils::closestPowerOf2(_images[0]->height());
            if (w2 == 0) SL_EXIT_MSG("Image can not be rescaled: width=0");
            if (h2 == 0) SL_EXIT_MSG("Image can not be rescaled: height=0");
            if (w2 != _images[0]->width() || h2 != _images[0]->height())
                _images[0]->resize((SLint)w2, (SLint)h2);
        }

        // check 2D size
        if (_target == GL_TEXTURE_2D)
        {
            if (_images[0]->width() > (SLuint)texMaxSize)
                SL_EXIT_MSG("SLGLTexture::build: Texture width is too big.");
            if (_images[0]->height() > (SLuint)texMaxSize)
                SL_EXIT_MSG("SLGLTexture::build: Texture height is too big.");
        }

        // check 3D size
        if (_target == GL_TEXTURE_3D)
        {
            SLint texMax3DSize = 0;
            glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &texMax3DSize);
            for (auto img : _images)
            {
                if (img->width() > (SLuint)texMax3DSize)
                    SL_EXIT_MSG("SLGLTexture::build: 3D Texture width is too big.");
                if (img->height() > (SLuint)texMax3DSize)
                    SL_EXIT_MSG("SLGLTexture::build: 3D Texture height is too big.");
                if (img->width() != _images[0]->width() ||
                    img->height() != _images[0]->height())
                    SL_EXIT_MSG("SLGLTexture::build: Not all images of the 3D texture have the same size.");
            }
        }

        // check cube mapping capability & max. cube map size
        if (_target == GL_TEXTURE_CUBE_MAP)
        {
            SLint texMaxCubeSize;
            glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &texMaxCubeSize);
            if (_images[0]->width() > (SLuint)texMaxCubeSize)
                SL_EXIT_MSG("SLGLTexture::build: Cube Texture width is too big.");
            if (_images.size() != 6)
                SL_EXIT_MSG("SLGLTexture::build: Not six images provided for cube map texture.");
        }

        // Generate texture names
        glGenTextures(1, &_texID);

        SLGLState* stateGL = SLGLState::instance();
        stateGL->activeTexture(GL_TEXTURE0 + (SLuint)texUnit);

        // create binding and apply texture properties
        stateGL->bindTexture(_target, _texID);

        // check if anisotropic texture filter extension is available
        if (maxAnisotropy < 0.0f)
        {
            if (stateGL->hasExtension("GL_EXT_texture_filter_anisotropic"))
                glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
            else
            {
                maxAnisotropy = 0.0f;
                SL_LOG("GL_EXT_texture_filter_anisotropic not available.\n");
            }
        }
        GET_GL_ERROR;

        // apply anisotropic or minification filter
        if (_min_filter > GL_LINEAR_MIPMAP_LINEAR)
        {
            SLfloat anisotropy; // = off
            if (_min_filter == SL_ANISOTROPY_MAX)
                anisotropy = maxAnisotropy;
            else
                anisotropy = std::min((SLfloat)(_min_filter - GL_LINEAR_MIPMAP_LINEAR),
                                      maxAnisotropy);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
        }
        else
            glTexParameteri(_target, GL_TEXTURE_MIN_FILTER, _min_filter);
        GET_GL_ERROR;

        // apply magnification filter only GL_NEAREST & GL_LINEAR is allowed
        glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, _mag_filter);
        GET_GL_ERROR;

        // apply texture wrapping modes
        glTexParameteri(_target, GL_TEXTURE_WRAP_S, _wrap_s);
        glTexParameteri(_target, GL_TEXTURE_WRAP_T, _wrap_t);
        glTexParameteri(_target, GL_TEXTURE_WRAP_R, _wrap_t);
        GET_GL_ERROR;

        // Handle special stupid case on iOS
        _internalFormat = _images[0]->format();
        if (_internalFormat == PF_red)
            _internalFormat = GL_R8;

        // Handle special case for realtime RT
        if (_images[0]->name() == "Optix Raytracer")
            _internalFormat = GL_RGB32F;

        // Handle special case for HDR textures
        if (_texType == TT_hdr)
            _internalFormat = SL_HDR_GL_INTERNAL_FORMAT;

        // Build textures
        if (_target == GL_TEXTURE_2D)
        {
            GLenum format = _images[0]->format();

            //////////////////////////////////////////////////////////////
            glTexImage2D(GL_TEXTURE_2D,
                         0,
                         _internalFormat,
                         (SLsizei)_images[0]->width(),
                         (SLsizei)_images[0]->height(),
                         0,
                         format,
                         _texType == TT_hdr ? SL_HDR_GL_TYPE : GL_UNSIGNED_BYTE,
                         (GLvoid*)_images[0]->data());
            /////////////////////////////////////////////////////////////

            GET_GL_ERROR;

            _bytesOnGPU = _images[0]->bytesPerImage();

            if (_min_filter >= GL_NEAREST_MIPMAP_NEAREST)
            {
                if (stateGL->glIsES2() ||
                    stateGL->glIsES3() ||
                    stateGL->glVersionNOf() >= 3.0)
                    glGenerateMipmap(GL_TEXTURE_2D);
                else
                    build2DMipmaps(GL_TEXTURE_2D, 0);

                // Mipmaps use 1/3 more memory on GPU
                _bytesOnGPU = (SLuint)((SLfloat)_bytesOnGPU * 1.333333333f);
                GET_GL_ERROR;
            }

            totalNumBytesOnGPU += _bytesOnGPU;
        }
        else if (_target == GL_TEXTURE_3D)
        {
            // temporary buffer for 3D image data
            SLVuchar buffer(_images[0]->bytesPerImage() * _images.size());
            SLuchar* imageData = &buffer[0];

            // copy each image data into temp. buffer
            for (CVImage* img : _images)
            {
                memcpy(imageData, img->data(), img->bytesPerImage());
                imageData += img->bytesPerImage();
                _bytesOnGPU += _images[0]->bytesPerImage();
            }

            /////////////////////////////////////////////////////
            glTexImage3D(GL_TEXTURE_3D,
                         0,               // Mipmap level,
                         _internalFormat, // Internal format
                         (SLsizei)_images[0]->width(),
                         (SLsizei)_images[0]->height(),
                         (SLsizei)_images.size(),
                         0,                    // Border
                         _images[0]->format(), // Format
                         GL_UNSIGNED_BYTE,     // Data type
                         &buffer[0]);
            /////////////////////////////////////////////////////

            totalNumBytesOnGPU += _bytesOnGPU;
            GET_GL_ERROR;
        }
        else if (_target == GL_TEXTURE_CUBE_MAP)
        {
            for (SLuint i = 0; i < 6; i++)
            {
                //////////////////////////////////////////////
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                             0,
                             _internalFormat,
                             (SLsizei)_images[i]->width(),
                             (SLsizei)_images[i]->height(),
                             0,
                             _images[i]->format(),
                             GL_UNSIGNED_BYTE,
                             (GLvoid*)_images[i]->data());
                //////////////////////////////////////////////

                _bytesOnGPU += _images[0]->bytesPerImage();
                GET_GL_ERROR;
            }

            totalNumBytesOnGPU += _bytesOnGPU;
            if (_min_filter >= GL_NEAREST_MIPMAP_NEAREST)
            {
                glGenerateMipmap(GL_TEXTURE_2D);

                // Mipmaps use 1/3 more memory on GPU
                _bytesOnGPU = (SLuint)((SLfloat)_bytesOnGPU * (4.0f / 3.0f));
            }
        }

        // If the images get deleted they only are on the GPU side
        if (_deleteImageAfterBuild)
            deleteImages();
    }

    // Check if texture name is valid only for debug purpose
    // if (glIsTexture(_texName))
    //     SL_LOG("SLGLTexture::build: name: %u, unit-id: %u, Filename: %s", _texName, texUnit, _images[0]->name().c_str());
    // else SL_LOG("SLGLTexture::build: invalid name: %u, unit-id: %u, Filename: %s", _texName, texUnit, _images[0]->name().c_str());

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
#ifdef SL_HAS_OPTIX
void SLGLTexture::buildCudaTexture()
{
    if (!_cudaTextureObject)
    {
        CUarray texture_ptr;

        CUDA_CHECK(cuGraphicsMapResources(1,
                                          &_cudaGraphicsResource,
                                          SLOptix::stream));
        CUDA_CHECK(cuGraphicsSubResourceGetMappedArray(&texture_ptr,
                                                       _cudaGraphicsResource,
                                                       0,
                                                       0));

        CUDA_RESOURCE_DESC res_desc = {};

        res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = texture_ptr;

        CUDA_TEXTURE_DESC tex_desc   = {};
        tex_desc.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
        tex_desc.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
        tex_desc.filterMode          = CU_TR_FILTER_MODE_LINEAR;
        tex_desc.flags               = CU_TRSF_NORMALIZED_COORDINATES;
        tex_desc.maxAnisotropy       = (SLuint)maxAnisotropy;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;

        CUDA_CHECK(cuTexObjectCreate(&_cudaTextureObject,
                                     &res_desc,
                                     &tex_desc,
                                     nullptr));
        CUDA_CHECK(cuGraphicsUnmapResources(1,
                                            &_cudaGraphicsResource,
                                            SLOptix::stream));
    }
}
#endif
//-----------------------------------------------------------------------------
/*!
SLGLTexture::bindActive binds the active texture. This method must be called
by the object that uses the texture every time BEFORE the its rendering.
The texUnit is only used for multi texturing. Before the first time the texture
is passed to OpenGL.
*/
void SLGLTexture::bindActive(SLuint texUnit)
{
    assert(texUnit >= 0 && texUnit < 16);

    // if texture not exists build it
    if (!_texID)
        build(texUnit);

    if (_texID)
    {
        SLGLState* stateGL = SLGLState::instance();
        // SL_LOG("SLGLTexture::bindActive: activeTexture: %d, bindTexture: %u, name: %s", texUnit, _texID, _name.c_str());
        stateGL->activeTexture(GL_TEXTURE0 + texUnit);
        stateGL->bindTexture(_target, _texID);

        if (_needsUpdate)
        {
            fullUpdate();
            _needsUpdate = false;
        }

#ifdef SL_HAS_OPTIX
        if (!_cudaGraphicsResource)
        {
            // Todo: Bugfix needed for Optix needs some work for newer shader models
            // CUDA_CHECK(
            cuGraphicsGLRegisterImage(&_cudaGraphicsResource,
                                      _texID,
                                      _target,
                                      CU_GRAPHICS_REGISTER_FLAGS_NONE);
            //);
        }
#endif
    }

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*!
Fully updates the OpenGL internal texture data by the image data
*/
void SLGLTexture::fullUpdate()
{
    PROFILE_FUNCTION();

    if (_texID &&
        !_images.empty() &&
        _images[0]->data() &&
        _target == GL_TEXTURE_2D)
    {
        // Do not allow MIP-Maps as they are to big
        if (_min_filter == GL_NEAREST || _min_filter == GL_LINEAR)
        {
            totalNumBytesOnGPU -= _bytesOnGPU;

            /////////////////////////////////////////////
            glTexSubImage2D(_target,
                            0,
                            0,
                            0,
                            (SLsizei)_images[0]->width(),
                            (SLsizei)_images[0]->height(),
                            _images[0]->format(),
                            GL_UNSIGNED_BYTE,
                            (GLvoid*)_images[0]->data());
            /////////////////////////////////////////////

            _bytesOnGPU = _images[0]->bytesPerImage();
            totalNumBytesOnGPU += _bytesOnGPU;
        }
        else
            SL_WARN_MSG("Filtering to expensive to full update in SLGLTexture::fullupdate!");
    }
    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
//! Draws the texture as 2D sprite with OpenGL buffers
/*! Draws the texture as a flat 2D sprite with a height and a width on two
triangles with zero in the bottom left corner: <br>
          w
       +-----+
       |    /|
       |   / |
    h  |  /  |
       | /   |
       |/    |
     0 +-----+
       0
*/
void SLGLTexture::drawSprite(SLbool doUpdate, SLfloat x, SLfloat y, SLfloat w, SLfloat h)
{
    // build buffer object once
    if (!_vaoSprite.vaoID())
    {
        // Vertex X & Y of corners
        SLVVec2f P = {{x, h},
                      {x, y},
                      {x + w, y + h},
                      {x + w, y}};

        // Texture coords of corners
        SLVVec2f T = {{0.0f, 1.0f},
                      {0.0f, 0.0f},
                      {1.0f, 1.0f},
                      {1.0f, 0.0f}};

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3};

        SLGLProgram* sp = SLGLProgramManager::get(SP_TextureOnly);
        sp->useProgram();
        _vaoSprite.setAttrib(AT_position, AT_position, &P);
        _vaoSprite.setAttrib(AT_uv1, AT_uv1, &T);
        _vaoSprite.setIndices(&I, nullptr);
        _vaoSprite.generate(4);
    }

    bindActive(0);              // Enable & build texture
    if (doUpdate) fullUpdate(); // Update the OpenGL texture on each draw

    // Draw the character triangles
    SLGLState*   stateGL = SLGLState::instance();
    SLGLProgram* sp      = SLGLProgramManager::get(SP_TextureOnly);
    sp->useProgram();
    sp->uniformMatrix4fv("u_mMatrix", 1, (SLfloat*)&stateGL->modelMatrix);
    sp->uniformMatrix4fv("u_vMatrix", 1, (SLfloat*)&stateGL->viewMatrix);
    sp->uniformMatrix4fv("u_pMatrix", 1, (SLfloat*)&stateGL->projectionMatrix);
    sp->uniform1i("u_matTexture0", 0);
    sp->uniform1f("u_oneOverGamma", 1.0f);

    ////////////////////////////////////////////
    _vaoSprite.drawElementsAs(PT_triangleStrip);
    ////////////////////////////////////////////
}
//-----------------------------------------------------------------------------
//! SLGLTexture::getTexelf returns a pixel color from u & v texture coordinates.
/*! If the OpenGL filtering is set to GL_LINEAR a bilinear interpolated color out
of four neighboring pixels is return. Otherwise the nearest pixel is returned.
*/
SLCol4f SLGLTexture::getTexelf(SLfloat u, SLfloat v, SLuint imgIndex)
{
    if (imgIndex < _images.size())
    {

        // transform tex coords with the texture matrix
        u = u * _tm.m(0) + _tm.m(12);
        v = v * _tm.m(5) + _tm.m(13);

        // Make sure the tex. coords are between 0.0 and 1.0
        if (u < 0.0f || u > 1.0f) u -= floor(u);
        if (v < 0.0f || v > 1.0f) v -= floor(v);

        // Bilinear interpolation
        if (_min_filter == GL_LINEAR || _mag_filter == GL_LINEAR)
        {
            CVVec4f c4f = _images[imgIndex]->getPixelf(u, v);
            return SLCol4f(c4f[0], c4f[1], c4f[2], c4f[3]);
        }
        else
        {
            CVVec4f c4f = _images[imgIndex]->getPixeli((SLint)(u * (float)_images[imgIndex]->width()),
                                                       (SLint)(v * (float)_images[imgIndex]->height()));
            return SLCol4f(c4f[0], c4f[1], c4f[2], c4f[3]);
        }
    }
    else
        return SLCol4f::BLACK;
}
//-----------------------------------------------------------------------------
//! SLGLTexture::getTexelf returns a pixel color at the specified cubemap direction
SLCol4f SLGLTexture::getTexelf(const SLVec3f& cubemapDir)
{
    assert(_images.size() == 6 &&
           _target == GL_TEXTURE_CUBE_MAP &&
           "SLGLTexture::getTexelf: Not a cubemap!");

    SLint   index = 0;
    SLfloat u, v;

    cubeXYZ2UV(cubemapDir.x, cubemapDir.y, cubemapDir.z, index, u, v);

    return getTexelf(u, v, (SLuint)index);
}
//-----------------------------------------------------------------------------
/*!
dudv calculates the partial derivation (gray value slope) at u,v for bump
mapping either from a height map or a normal map
*/
SLVec2f SLGLTexture::dudv(SLfloat u, SLfloat v)
{
    SLVec2f dudv(0, 0);
    SLfloat du = 1.0f / (SLfloat)_images[0]->width();
    SLfloat dv = 1.0f / (SLfloat)_images[0]->height();

    if (_texType == TT_height)
    {
        dudv.x = (getTexelf(u + du, v).x - getTexelf(u - du, v).x) * -_bumpScale;
        dudv.y = (getTexelf(u, v + dv).x - getTexelf(u, v - dv).x) * -_bumpScale;
    }
    else if (_texType == TT_normal)
    {
        SLVec4f texel = getTexelf(u, v);
        dudv.x        = texel.r * 2.0f - 1.0f;
        dudv.y        = texel.g * 2.0f - 1.0f;
    }
    return dudv;
}
//-----------------------------------------------------------------------------
//! Detects the texture type from the filename appendix (See SLTexType def.)
SLTextureType SLGLTexture::detectType(const SLstring& filename)
{
    // Check first our own texture name encoding
    SLstring name     = Utils::getFileNameWOExt(filename);
    SLstring ext      = Utils::getFileExt(filename);
    SLstring appendix = name.substr(name.length() - 2, 2);

    if (ext == "hdr") return TT_hdr;
    if (appendix == "_C") return TT_diffuse; // Color
    if (appendix == "_D") return TT_diffuse;
    if (appendix == "_N") return TT_normal;
    if (appendix == "_H") return TT_height;
    if (appendix == "_G") return TT_specular; // Gloss
    if (appendix == "_S") return TT_specular;
    if (appendix == "_R") return TT_roughness;
    if (appendix == "_M") return TT_metallic;
    if (appendix == "_O") return TT_occlusion; // Ambient Occlusion
    if (appendix == "_A") return TT_occlusion;
    if (appendix == "_F") return TT_font;

    // Now check various formats found in the past
    name = Utils::toUpperString(name);

    if (Utils::containsString(name, "COL") ||
        Utils::containsString(name, "COLOR") ||
        Utils::containsString(name, "BASECOLOR") ||
        Utils::containsString(name, "ALBEDO") ||
        Utils::containsString(name, "DIFFUSE") ||
        Utils::containsString(name, "DIFF") ||
        Utils::containsString(name, "DIF"))
        return TT_diffuse;

    if (Utils::containsString(name, "NRM") ||
        Utils::containsString(name, "NORM") ||
        Utils::containsString(name, "NORMAL"))
        return TT_normal;

    if (Utils::containsString(name, "DISP") ||
        Utils::containsString(name, "DISPL") ||
        Utils::containsString(name, "HEIGHT") ||
        Utils::containsString(name, "BUMP"))
        return TT_height;

    if (Utils::containsString(name, "GLOSS") ||
        Utils::containsString(name, "REFL") ||
        Utils::containsString(name, "SPECULAR") ||
        Utils::containsString(name, "SPEC"))
        return TT_specular;

    if (Utils::containsString(name, "OCCLUSIONROUGHNESSMETALLIC"))
        return TT_occluRoughMetal;

    if (Utils::containsString(name, "ROUGHNESSMETALLIC"))
        return TT_roughMetal;

    if (Utils::containsString(name, "ROUGHNESS") ||
        Utils::containsString(name, "RGH") ||
        Utils::containsString(name, "ROUGH"))
        return TT_roughness;

    if (Utils::containsString(name, "METAL") ||
        Utils::containsString(name, "METALLIC") ||
        Utils::containsString(name, "METALNESS"))
        return TT_metallic;

    if (Utils::containsString(name, "AO") ||
        Utils::containsString(name, "AMBIENT") ||
        Utils::containsString(name, "OCCLUSION") ||
        Utils::containsString(name, "OCCLU") ||
        Utils::containsString(name, "OCCL") ||
        Utils::containsString(name, "OCC"))
        return TT_occlusion;

    // if nothing was detected so far we interpret it as a color texture
    // SLstring msg = Utils::formatString("SLGLTexture::detectType: No type detected in file: %s", filename.c_str());
    // SL_WARN_MSG(msg.c_str());

    return TT_diffuse;
}
//-----------------------------------------------------------------------------
void SLGLTexture::build2DMipmaps(SLint target, SLuint index)
{
    // Create the base level mipmap
    SLint level = 0;
    glTexImage2D((SLuint)target,
                 level,
                 (SLint)_images[index]->bytesPerPixel(),
                 (SLsizei)_images[index]->width(),
                 (SLsizei)_images[index]->height(),
                 0,
                 _images[index]->format(),
                 GL_UNSIGNED_BYTE,
                 (GLvoid*)_images[index]->data());
    GET_GL_ERROR;

    // working copy of the base mipmap
    CVImage img2(*_images[index]);

    // create half sized sub level mipmaps
    while (img2.width() > 1 || img2.height() > 1)
    {
        level++;
        img2.resize((SLint)std::max(img2.width() >> 1, (SLuint)1),
                    (SLint)std::max(img2.height() >> 1, (SLuint)1));

        // SLfloat gauss[9] = {1.0f, 2.0f, 1.0f,
        //                     2.0f, 4.0f, 2.0f,
        //                     1.0f, 2.0f, 1.0f};

        // img2.convolve3x3(gauss);

        // Debug output
        // SLchar filename[255];
        // snprintf(filename,sizeof(filename),"%s_L%d_%dx%d.png", _name.c_str(), level, img2.width(), img2.height());
        // img2.savePNG(filename);

        glTexImage2D((SLuint)target,
                     level,
                     (SLint)img2.bytesPerPixel(),
                     (SLsizei)img2.width(),
                     (SLsizei)img2.height(),
                     0,
                     img2.format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)img2.data());
        GET_GL_ERROR;
    }
}
//-----------------------------------------------------------------------------
//! Returns the texture type as string
SLstring SLGLTexture::typeName()
{
    switch (_texType)
    {
        case TT_unknown: return "TT_unknown";
        case TT_diffuse: return "TT_diffuse";
        case TT_normal: return "TT_normal";
        case TT_height: return "TT_height";
        case TT_specular: return "TT_specular";
        case TT_emissive: return "TT_emissive";
        case TT_roughness: return "TT_roughness";
        case TT_metallic: return "TT_metallic";
        case TT_occluRoughMetal: return "TT_occluRoughMetal";
        case TT_roughMetal: return "TT_roughMetal";
        case TT_occlusion: return "TT_occlusion";
        case TT_font: return "TT_font";
        case TT_hdr: return "TT_hdr";
        case TT_environmentCubemap: return "TT_environmentCubemap";
        case TT_irradianceCubemap: return "TT_irradianceCubemap";
        case TT_roughnessCubemap:
        case TT_brdfLUT: return "TT_roughnessCubemap";
        case TT_videoBkgd: return "TT_videoBkgd";
        default: return "TT_unknown";
    }
}
//-----------------------------------------------------------------------------
//! Returns the texture type short
SLstring SLGLTexture::typeShortName()
{
    switch (_texType)
    {
        case TT_unknown: return "U";
        case TT_diffuse: return "D";
        case TT_normal: return "N";
        case TT_height: return "H";
        case TT_specular: return "S";
        case TT_emissive: return "E";
        case TT_roughness: return "R";
        case TT_metallic: return "M";
        case TT_occlusion: return "O";
        case TT_roughMetal: return "RM";
        case TT_occluRoughMetal: return "ORM";
        case TT_font: return "F";
        case TT_hdr: return "HDR";
        case TT_environmentCubemap: return "EnvE";
        case TT_irradianceCubemap: return "EnvI";
        case TT_roughnessCubemap: return "EnvR";
        case TT_brdfLUT: return "brdf";
        case TT_videoBkgd: return "VidBkgd";
        default: return "U";
    }
}
//-----------------------------------------------------------------------------
//! Returns OpenGL texture filter as string
SLstring SLGLTexture::filterString(SLint glFilter)
{
    switch (glFilter)
    {
        case GL_NEAREST: return "nearest";
        case GL_LINEAR: return "linear";
        case GL_NEAREST_MIPMAP_NEAREST: return "nearest mipmap nearest";
        case GL_LINEAR_MIPMAP_NEAREST: return "linear mipmap linear";
        case GL_NEAREST_MIPMAP_LINEAR: return "nearest mipmap linear";
        case GL_LINEAR_MIPMAP_LINEAR: return "linear mipmap linear";
        case SL_ANISOTROPY_MAX: return "anisotropic max.";
        default: return "unknown";
    }
}
//-----------------------------------------------------------------------------
/*! SLGLTexture::calc3DGradients calculates the normals based on the 3D
gradient of all images and stores them in the RGB components.
@param sampleRadius Distance from center to calculate the gradient
@param onUpdateProgress Callback function for progress display
*/
void SLGLTexture::calc3DGradients(SLint                      sampleRadius,
                                  const function<void(int)>& onUpdateProgress)
{
    SLint   r          = sampleRadius;
    SLint   volX       = (SLint)_images[0]->width();
    SLint   volY       = (SLint)_images[0]->height();
    SLint   volZ       = (SLint)_images.size();
    SLuint  numVoxels  = volX * volY * volZ;
    SLuint  cntVoxels  = 0;
    SLfloat oneOver255 = 1.0f / 255.0f;

    // check that all images in depth have the same size
    for (auto img : _images)
        if ((SLint)img->width() != volX ||
            (SLint)img->height() != volY || img->format() != PF_rgba)
            SL_EXIT_MSG("SLGLTexture::calc3DGradients: Not all images have the same size!");

    for (int z = r; z < volZ - r; ++z)
    {
        for (int y = r; y < volY - r; ++y)
        {
            for (int x = r; x < volX - r; ++x)
            {
                // Calculate the min & max vectors
                SLVec3f min, max;
                min.x = (SLfloat)_images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x - r)[3] * oneOver255;
                max.x = (SLfloat)_images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x + r)[3] * oneOver255;
                min.y = (SLfloat)_images[(SLuint)z]->cvMat().at<cv::Vec4b>(y - r, x)[3] * oneOver255;
                max.y = (SLfloat)_images[(SLuint)z]->cvMat().at<cv::Vec4b>(y + r, x)[3] * oneOver255;
                min.z = (SLfloat)_images[(SLuint)z - (SLuint)r]->cvMat().at<cv::Vec4b>(y, x)[3] * oneOver255;
                max.z = (SLfloat)_images[(SLuint)z + (SLuint)r]->cvMat().at<cv::Vec4b>(y, x)[3] * oneOver255;

                // Calculate normal as the difference between max & min
                SLVec3f normal = max - min;
                SLfloat length = normal.length();
                if (length > 0.0001f)
                    normal /= length;
                else
                    normal.set(0, 0, 0);

                // Store normal in the rgb channels. Scale range from -1 - 1 to 0 - 1 to 0 - 255
                normal += 1.0f;
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[0] = (SLuchar)(normal.x * 0.5f * 255.0f);
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[1] = (SLuchar)(normal.y * 0.5f * 255.0f);
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[2] = (SLuchar)(normal.z * 0.5f * 255.0f);

                // Calculate progress in percent
                cntVoxels++;
                SLint progress = (SLint)((SLfloat)cntVoxels / (SLfloat)numVoxels * 100.0f);
                onUpdateProgress(progress);
            }
        }
    }

    // Debug check
    // for (auto img : _images)
    //   img->savePNG(img->path() + "Normals_" + img->name());
}
//-----------------------------------------------------------------------------
/*! SLGLTexture::smooth3DGradients smooths the 3D gradients in the RGB channels
of all images.
@param smoothRadius Soothing radius
@param onUpdateProgress Callback function for progress display
*/
void SLGLTexture::smooth3DGradients(SLint               smoothRadius,
                                    function<void(int)> onUpdateProgress)
{
    SLint   r          = smoothRadius;
    SLint   volX       = (SLint)_images[0]->width();
    SLint   volY       = (SLint)_images[0]->height();
    SLint   volZ       = (SLint)_images.size();
    SLuint  numVoxels  = volX * volY * volZ;
    SLuint  cntVoxels  = 0;
    SLfloat oneOver255 = 1.0f / 255.0f;

    // check that all images in depth have the same size
    for (auto img : _images)
        if ((SLint)img->width() != volX ||
            (SLint)img->height() != volY || img->format() != PF_rgba)
            SL_EXIT_MSG("SLGLTexture::calc3DGradients: Not all images have the same size3@!");

    //@todo This is very slow and should be implemented as separable filter
    for (int z = r; z < volZ - r; ++z)
    {
        for (int y = r; y < volY - r; ++y)
        {
            for (int x = r; x < volX - r; ++x)
            {
                SLVec3f filtered(0, 0, 0);

                // box filter (= average)
                SLint num = 0;
                for (int fz = z - r; fz <= z + r; ++fz)
                {
                    for (int fy = y - r; fy <= y + r; ++fy)
                    {
                        for (int fx = x - r; fx <= x + r; ++fx)
                        {
                            filtered += SLVec3f((SLfloat)_images[(SLuint)fz]->cvMat().at<cv::Vec4b>(fy, fx)[0] * oneOver255 * 2.0f - 1.0f,
                                                (SLfloat)_images[(SLuint)fz]->cvMat().at<cv::Vec4b>(fy, fx)[1] * oneOver255 * 2.0f - 1.0f,
                                                (SLfloat)_images[(SLuint)fz]->cvMat().at<cv::Vec4b>(fy, fx)[2] * oneOver255 * 2.0f - 1.0f);
                            num++;
                        }
                    }
                }
                filtered /= (SLfloat)num;

                // Store normal in the rgb channels. Scale range from -1 - 1 to 0 - 1 to 0 - 255
                filtered += 1.0f;
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[0] = (SLuchar)(filtered.x * 0.5f * 255.0f);
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[1] = (SLuchar)(filtered.y * 0.5f * 255.0f);
                _images[(SLuint)z]->cvMat().at<cv::Vec4b>(y, x)[2] = (SLuchar)(filtered.z * 0.5f * 255.0f);

                // Calculate progress in percent
                cntVoxels++;
                SLint progress = (SLint)((SLfloat)cntVoxels / (SLfloat)numVoxels * 100.0f);
                onUpdateProgress(progress);
            }
        }
    }
}
//-----------------------------------------------------------------------------
//! Computes the unnormalised vector x,y,z from tex. coords. uv with cubemap index.
/*! A cube texture indexes six texture maps from 0 to 5 in order Positive X,
Negative X, Positive Y, Negative Y, Positive Z, Negative Z. The images are
stored with the origin at the lower left of the image. The Positive X and Y
faces must reverse the Z coordinate and the Negative Z face must negate the X
coordinate. If given the face, and texture coordinates (u,v), the unnormalized
vector (x,y,z) are computed. Source:\n
https://en.wikipedia.org/wiki/Cube_mapping
*/
void SLGLTexture::cubeUV2XYZ(SLint    index,
                             SLfloat  u,
                             SLfloat  v,
                             SLfloat& x,
                             SLfloat& y,
                             SLfloat& z)
{
    assert(_images.size() == 6 &&
           _target == GL_TEXTURE_CUBE_MAP &&
           "SLGLTexture::cubeUV2XYZ: Not a cubemap!");

    // convert range 0 to 1 to -1 to 1
    SLfloat uc = 2.0f * u - 1.0f;
    SLfloat vc = 2.0f * v - 1.0f;
    switch (index)
    {
        case 0:
            x = 1.0f;
            y = vc;
            z = -uc;
            break; // POSITIVE X
        case 1:
            x = -1.0f;
            y = vc;
            z = uc;
            break; // NEGATIVE X
        case 2:
            x = uc;
            y = 1.0f;
            z = -vc;
            break; // POSITIVE Y
        case 3:
            x = uc;
            y = -1.0f;
            z = vc;
            break; // NEGATIVE Y
        case 4:
            x = uc;
            y = vc;
            z = 1.0f;
            break; // POSITIVE Z
        case 5:
            x = -uc;
            y = vc;
            z = -1.0f;
            break; // NEGATIVE Z
        default:
            SL_EXIT_MSG("SLGLTexture::cubeUV2XYZ: Invalid index");
    }
}
//------------------------------------------------------------------------------
//! Computes the uv and cubemap image index from a unnormalised vector x,y,z.
/*! See also SLGLTexture::cubeUV2XYZ. Source:\n
https://en.wikipedia.org/wiki/Cube_mapping
*/
void SLGLTexture::cubeXYZ2UV(SLfloat  x,
                             SLfloat  y,
                             SLfloat  z,
                             SLint&   index,
                             SLfloat& u,
                             SLfloat& v)
{
    assert(_images.size() == 6 &&
           _target == GL_TEXTURE_CUBE_MAP &&
           "SLGLTexture::cubeXYZ2UV: Not a cubemap!");

    SLfloat absX = fabs(x);
    SLfloat absY = fabs(y);
    SLfloat absZ = fabs(z);

    SLint isXPositive = x > 0 ? 1 : 0;
    SLint isYPositive = y > 0 ? 1 : 0;
    SLint isZPositive = z > 0 ? 1 : 0;

    SLfloat maxAxis = 0.0f, uc = 0.0f, vc = 0.0f;

    // POSITIVE X
    if (isXPositive && absX >= absY && absX >= absZ)
    {
        // u (0 to 1) goes from +z to -z
        // v (0 to 1) goes from -y to +y
        maxAxis = absX;
        uc      = -z;
        vc      = y;
        index   = 0;
    }

    // NEGATIVE X
    if (!isXPositive && absX >= absY && absX >= absZ)
    {
        // u (0 to 1) goes from -z to +z
        // v (0 to 1) goes from -y to +y
        maxAxis = absX;
        uc      = z;
        vc      = y;
        index   = 1;
    }

    // POSITIVE Y
    if (isYPositive && absY >= absX && absY >= absZ)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +z to -z
        maxAxis = absY;
        uc      = x;
        vc      = -z;
        index   = 2;
    }

    // NEGATIVE Y
    if (!isYPositive && absY >= absX && absY >= absZ)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -z to +z
        maxAxis = absY;
        uc      = x;
        vc      = z;
        index   = 3;
    }

    // POSITIVE Z
    if (isZPositive && absZ >= absX && absZ >= absY)
    {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -y to +y
        maxAxis = absZ;
        uc      = x;
        vc      = y;
        index   = 4;
    }

    // NEGATIVE Z
    if (!isZPositive && absZ >= absX && absZ >= absY)
    {
        // u (0 to 1) goes from +x to -x
        // v (0 to 1) goes from -y to +y
        maxAxis = absZ;
        uc      = -x;
        vc      = y;
        index   = 5;
    }

    // Convert range from -1 to 1 to 0 to 1
    u = 0.5f * (uc / maxAxis + 1.0f);
    v = -0.5f * (vc / maxAxis + 1.0f);
}
#ifdef SL_BUILD_WITH_KTX
//------------------------------------------------------------------------------
//! Returns the KTX transcoding compression format as string
string SLGLTexture::compressionFormatStr(int compressionFormat)
{
    switch (compressionFormat)
    {
        case KTX_TTF_ETC1_RGB: return "ETC1_RGB";
        case KTX_TTF_ETC2_RGBA: return "ETC2_RGBA";
        case KTX_TTF_BC1_RGB: return "BC1_RGB";
        case KTX_TTF_BC3_RGBA: return "BC3_RGBA";
        case KTX_TTF_BC4_R: return "BC4_R";
        case KTX_TTF_BC5_RG: return "BC5_RG";
        case KTX_TTF_BC7_RGBA: return "BC7_RGBA";
        case KTX_TTF_PVRTC1_4_RGB: return "PVRTC1_4_RGB";
        case KTX_TTF_PVRTC1_4_RGBA: return "PVRTC1_4_RGBA";
        case KTX_TTF_ASTC_4x4_RGBA: return "ASTC_4x4_RGBA";
        case KTX_TTF_PVRTC2_4_RGB: return "PVRTC2_4_RGB";
        case KTX_TTF_PVRTC2_4_RGBA: return "PVRTC2_4_RGBA";
        case KTX_TTF_ETC2_EAC_R11: return "ETC2_EAC_R11";
        case KTX_TTF_ETC2_EAC_RG11: return "ETC2_EAC_RG11";
        case KTX_TTF_RGBA32: return "RGBA32";
        case KTX_TTF_RGB565: return "RGB565";
        case KTX_TTF_BGR565: return "BGR565";
        case KTX_TTF_RGBA4444: return "RGBA4444";
        case KTX_TTF_ETC: return "ETC";
        case KTX_TTF_BC1_OR_3: return "BC1_OR_3";
        default: return "NOT_COMPRESSED";
    }
}
//------------------------------------------------------------------------------
string SLGLTexture::ktxErrorStr(int ktxErrorCode)
{
    switch (ktxErrorCode)
    {
        case KTX_SUCCESS: return "KTX_SUCCESS";
        case KTX_FILE_DATA_ERROR: return "KTX_FILE_DATA_ERROR";
        case KTX_FILE_ISPIPE: return "KTX_FILE_ISPIPE";
        case KTX_FILE_OPEN_FAILED: return "KTX_FILE_OPEN_FAILED";
        case KTX_FILE_OVERFLOW: return "KTX_FILE_OVERFLOW";
        case KTX_FILE_READ_ERROR: return "KTX_FILE_READ_ERROR";
        case KTX_FILE_SEEK_ERROR: return "KTX_FILE_SEEK_ERROR";
        case KTX_FILE_UNEXPECTED_EOF: return "KTX_FILE_UNEXPECTED_EOF";
        case KTX_FILE_WRITE_ERROR: return "KTX_FILE_WRITE_ERROR";
        case KTX_GL_ERROR: return "KTX_GL_ERROR";
        case KTX_INVALID_OPERATION: return "KTX_INVALID_OPERATION";
        case KTX_INVALID_VALUE: return "KTX_INVALID_VALUE";
        case KTX_NOT_FOUND: return "KTX_NOT_FOUND";
        case KTX_OUT_OF_MEMORY: return "KTX_OUT_OF_MEMORY";
        case KTX_TRANSCODE_FAILED: return "KTX_TRANSCODE_FAILED";
        case KTX_UNKNOWN_FILE_FORMAT: return "KTX_UNKNOWN_FILE_FORMAT";
        case KTX_UNSUPPORTED_TEXTURE_TYPE: return "KTX_UNSUPPORTED_TEXTURE_TYPE";
        case KTX_UNSUPPORTED_FEATURE: return "KTX_UNSUPPORTED_FEATURE";
        case KTX_LIBRARY_NOT_LINKED: return "KTX_LIBRARY_NOT_LINKED";
        default: "Unknown KTX_ERROR";
    }
    return string("Unknown KTX_ERROR");
}
#endif
//------------------------------------------------------------------------------
//! Returns the internal pixel format from OpenGL
/*!
 * Taken from https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
 */
string SLGLTexture::internalFormatStr(int internalFormat)
{
    switch (internalFormat)
    {
        case GL_DEPTH_COMPONENT: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_STENCIL: return "GL_DEPTH_STENCIL";
        case GL_RED: return "GL_RED";
        case GL_RG: return "GL_RG";
        case GL_RGB: return "GL_RGB";
        case GL_RGBA: return "GL_RGBA";
        case GL_R8: return "GL_R8";
        case GL_R8_SNORM: return "GL_R8_SNORM";
        // case GL_R16: return "GL_R16"; // Not available on iOS
        // case GL_R16_SNORM: return "GL_R16_SNORM";
        case GL_RG8: return "GL_RG8";
        case GL_RG8_SNORM: return "GL_RG8_SNORM";
        // case GL_RG16: return "GL_RG16";
        // case GL_RG16_SNORM: return "GL_RG16_SNORM";
        // case GL_R3_G3_B2: return "GL_R3_G3_B2";
        // case GL_RGB4: return "GL_RGB4";
        // case GL_RGB5: return "GL_RGB5";
        // case GL_RGB8: return "GL_RGB8";
        case GL_RGB8_SNORM: return "GL_RGB8_SNORM";
        // case GL_RGB10: return "GL_RGB10";
        // case GL_RGB12: return "GL_RGB12";
        // case GL_RGB16_SNORM: return "GL_RGB16_SNORM";
        // case GL_RGBA2: return "GL_RGBA2";
        // case GL_RGBA4: return "GL_RGBA4";
        // case GL_RGB5_A1: return "GL_RGB5_A1";
        case GL_RGBA8: return "GL_RGBA8";
        case GL_RGBA8_SNORM: return "GL_RGBA8_SNORM";
        case GL_RGB10_A2: return "GL_RGB10_A2";
        case GL_RGB10_A2UI: return "GL_RGB10_A2UI";
        // case GL_RGBA12: return "GL_RGBA12";
        // case GL_RGBA16: return "GL_RGBA16";
        case GL_SRGB8: return "GL_SRGB8";
        case GL_SRGB8_ALPHA8: return "GL_SRGB8_ALPHA8";
        case GL_R16F: return "GL_R16F";
        case GL_RG16F: return "GL_RG16F";
        case GL_RGB16F: return "GL_RGB16F";
        case GL_RGBA16F: return "GL_RGBA16F";
        case GL_R32F: return "GL_R32F";
        case GL_RG32F: return "GL_RG32F";
        case GL_RGB32F: return "GL_RGB32F";
        case GL_RGBA32F: return "GL_RGBA32F";
        case GL_R11F_G11F_B10F: return "GL_R11F_G11F_B10F";
        case GL_RGB9_E5: return "GL_RGB9_E5";
        case GL_R8I: return "GL_R8I";
        case GL_R8UI: return "GL_R8UI";
        case GL_R16I: return "GL_R16I";
        case GL_R16UI: return "GL_R16UI";
        case GL_R32I: return "GL_R32I";
        case GL_R32UI: return "GL_R32UI";
        case GL_RG8I: return "GL_RG8I";
        case GL_RG8UI: return "GL_RG8UI";
        case GL_RG16I: return "GL_RG16I";
        case GL_RG16UI: return "GL_RG16UI";
        case GL_RG32I: return "GL_RG32I";
        case GL_RG32UI: return "GL_RG32UI";
        case GL_RGB8I: return "GL_RGB8I";
        case GL_RGB8UI: return "GL_RGB8UI";
        case GL_RGB16I: return "GL_RGB16I";
        case GL_RGB16UI: return "GL_RGB16UI";
        case GL_RGB32I: return "GL_RGB32I";
        case GL_RGB32UI: return "GL_RGB32UI";
        case GL_RGBA8I: return "GL_RGBA8I";
        case GL_RGBA8UI: return "GL_RGBA8UI";
        case GL_RGBA16I: return "GL_RGBA16I";
        case GL_RGBA16UI: return "GL_RGBA16UI";
        case GL_RGBA32I: return "GL_RGBA32I";
        case GL_RGBA32UI: return "GL_RGBA32UI";
        // case GL_COMPRESSED_RED: return "GL_COMPRESSED_RED";
        // case GL_COMPRESSED_RG: return "GL_COMPRESSED_RG";
        // case GL_COMPRESSED_RGB: return "GL_COMPRESSED_RGB";
        // case GL_COMPRESSED_RGBA: return "GL_COMPRESSED_RGBA";
        // case GL_COMPRESSED_SRGB: return "GL_COMPRESSED_SRGB";
        // case GL_COMPRESSED_SRGB_ALPHA: return "GL_COMPRESSED_SRGB_ALPHA";
        // case GL_COMPRESSED_RED_RGTC1: return "GL_COMPRESSED_RED_RGTC1";
        // case GL_COMPRESSED_SIGNED_RED_RGTC1: return "GL_COMPRESSED_SIGNED_RED_RGTC1";
        // case GL_COMPRESSED_RG_RGTC2: return "GL_COMPRESSED_RG_RGTC2";
        // case GL_COMPRESSED_SIGNED_RG_RGTC2: return "GL_COMPRESSED_SIGNED_RG_RGTC2";
        // case GL_COMPRESSED_RGBA_BPTC_UNORM: return "GL_COMPRESSED_RGBA_BPTC_UNORM";
        // case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: return "GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM";
        // case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT: return "GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT";
        // case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: return "GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT";
        default: return "Unknown format";
    }
}
//------------------------------------------------------------------------------
