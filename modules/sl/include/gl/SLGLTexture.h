//#############################################################################
//  File:      SLGLTexture.h
//  Author:    Marcus Hudritsch, Martin Christen
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLGLTEXTURE_H
#define SLGLTEXTURE_H

#include <SLObject.h>
#include <CVImage.h>
#include <SLGLVertexArray.h>
#include <SLMat4.h>
#include <atomic>
#include <mutex>

#ifdef SL_HAS_OPTIX
#    include <cuda.h>
#endif

class SLGLState;
class SLAssetManager;
class SLGLProgram;

//-----------------------------------------------------------------------------
// Special constants for anisotropic filtering
#define SL_ANISOTROPY_MAX (GL_LINEAR_MIPMAP_LINEAR + 1)
#define SL_ANISOTROPY_2 (GL_LINEAR_MIPMAP_LINEAR + 2)
#define SL_ANISOTROPY_4 (GL_LINEAR_MIPMAP_LINEAR + 4)
#define SL_ANISOTROPY_8 (GL_LINEAR_MIPMAP_LINEAR + 8)
#define SL_ANISOTROPY_16 (GL_LINEAR_MIPMAP_LINEAR + 16)
#define SL_ANISOTROPY_32 (GL_LINEAR_MIPMAP_LINEAR + 32)
//-----------------------------------------------------------------------------
// Extension constant for anisotropic filtering
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#    define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#    define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
//-----------------------------------------------------------------------------
//! Texture type enumeration & their filename appendix for auto type detection
enum SLTextureType
{
    TT_unknown,            // will be handled as color maps
    TT_diffuse,            // diffuse color map (aka albedo or just color map)
    TT_normal,             // normal map for normal bump mapping
    TT_height,             // height map for height map bump or parallax mapping
    TT_specular,           // specular gloss map
    TT_ambientOcclusion,   // ambient occlusion map
    TT_roughness,          // roughness map (PBR Cook-Torrance roughness 0-1)
    TT_metallic,           // metalness map (PBR Cook-Torrance metallic 0-1)
    TT_font,               // texture map for fonts
    TT_hdr,                // High Dynamic Range images
    TT_environmentCubemap, // environment cubemap generated from HDR Textures
    TT_irradianceCubemap,  // irradiance cubemap generated from HDR Textures
    TT_roughnessCubemap,   // prefilter roughness cubemap
    TT_brdfLUT             // BRDF 2D look up table Texture
};
//-----------------------------------------------------------------------------
//! Texture object for OpenGL texturing
/*!
The SLGLTexture class implements an OpenGL texture object that can be used by the
SLMaterial class. A texture can have 1-n CVImages in the vector _images.
A simple 2D texture has just a single texture image (_images[0]). For cube maps
you will need 6 images (_images[0-5]). For 3D textures you can have as much
images of the same size than your GPU and/or CPU memory can hold.
The images are not released after the OpenGL texture creation. They may be needed
for ray tracing.
*/
class SLGLTexture : public SLObject
{
public:
    //! Default ctor for all stack instances (not created with new)
    SLGLTexture();

    //! ctor for 1D texture with internal image allocation
    explicit SLGLTexture(SLAssetManager* assetMgr,
                         const SLVCol4f& colors,
                         SLint           min_filter = GL_LINEAR,
                         SLint           mag_filter = GL_LINEAR,
                         SLint           wrapS      = GL_REPEAT,
                         const SLstring& name       = "2D-Texture");

    //! ctor for empty 2D textures
    explicit SLGLTexture(SLAssetManager* assetMgr,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLint           wrapS,
                         SLint           wrapT);

    //! ctor for 2D textures from byte pointer
    explicit SLGLTexture(SLAssetManager* assetMgr,
                         unsigned char*  data,
                         int             width,
                         int             height,
                         int             cvtype,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLTextureType   type,
                         SLint           wrapS,
                         SLint           wrapT);

    //! ctor for 2D textures with internal image allocation
    explicit SLGLTexture(SLAssetManager* assetMgr,
                         const SLstring& imageFilename,
                         SLint           min_filter = GL_LINEAR_MIPMAP_LINEAR,
                         SLint           mag_filter = GL_LINEAR,
                         SLTextureType   type       = TT_unknown,
                         SLint           wrapS      = GL_REPEAT,
                         SLint           wrapT      = GL_REPEAT);

    //! ctor for 3D texture with internal image allocation
    explicit SLGLTexture(SLAssetManager*  assetMgr,
                         const SLVstring& imageFilenames,
                         SLint            min_filter             = GL_LINEAR,
                         SLint            mag_filter             = GL_LINEAR,
                         SLint            wrapS                  = GL_REPEAT,
                         SLint            wrapT                  = GL_REPEAT,
                         const SLstring&  name                   = "3D-Texture",
                         SLbool           loadGrayscaleIntoAlpha = false);

    //! ctor for cube mapping with internal image allocation
    SLGLTexture(SLAssetManager* assetMgr,
                const SLstring& imageFilenameXPos,
                const SLstring& imageFilenameXNeg,
                const SLstring& imageFilenameYPos,
                const SLstring& imageFilenameYNeg,
                const SLstring& imageFilenameZPos,
                const SLstring& imageFilenameZNeg,
                SLint           min_filter = GL_LINEAR,
                SLint           mag_filter = GL_LINEAR,
                SLTextureType   type       = TT_unknown);

    ~SLGLTexture() override;

    virtual void   clearData();
    virtual void   build(SLint texUnit = 0);
    virtual SLuint width() { return _images.empty() ? 0 : _images[0]->width(); }
    virtual SLuint height() { return _images.empty() ? 0 : _images[0]->height(); }
    virtual SLint  bytesPerPixel() { return (SLint)_images.empty() ? 0 : _images[0]->bytesPerPixel(); }

    void     bindActive(SLuint texUnit = 0);
    void     fullUpdate();
    void     drawSprite(SLbool doUpdate, SLfloat x, SLfloat y, SLfloat w, SLfloat h);
    void     cubeUV2XYZ(SLint index, SLfloat u, SLfloat v, SLfloat& x, SLfloat& y, SLfloat& z);
    void     cubeXYZ2UV(SLfloat x, SLfloat y, SLfloat z, SLint& index, SLfloat& u, SLfloat& v);
    SLstring filterString(SLint glFilter);

    // Setters
    void texType(SLTextureType bt) { _texType = bt; }
    void bumpScale(SLfloat bs) { _bumpScale = bs; }
    void minFiler(SLint minF) { _min_filter = minF; } // must be called befor build
    void magFiler(SLint magF) { _mag_filter = magF; } // must be called befor build
    void needsUpdate(SLbool update) { _needsUpdate = update; }

    // Getters
    CVVImage&     images() { return _images; }
    SLenum        target() const { return _target; }
    SLuint        texID() const { return _texID; }
    SLTextureType texType() { return _texType; }
    SLfloat       bumpScale() const { return _bumpScale; }
    SLCol4f       getTexelf(SLfloat u, SLfloat v, SLuint imgIndex = 0);
    SLCol4f       getTexelf(const SLVec3f& cubemapDir);
    SLbool        hasAlpha() { return (!_images.empty() &&
                                ((_images[0]->format() == PF_rgba ||
                                  _images[0]->format() == PF_bgra) ||
                                 _texType == TT_font)); }
    SLMat4f       tm() { return _tm; }
    SLbool        autoCalcTM3D() const { return _autoCalcTM3D; }
    SLbool        needsUpdate() { return _needsUpdate; }
    SLstring      typeName();
    SLstring      minificationFilterName() { return filterString(_min_filter); }
    SLstring      magnificationFilterName() { return filterString(_mag_filter); }

#ifdef SL_HAS_OPTIX
    void        buildCudaTexture();
    CUtexObject getCudaTextureObject()
    {
        buildCudaTexture();
        return _cudaTextureObject;
    }
#endif

    // Misc
    static SLTextureType detectType(const SLstring& filename);
    void                 build2DMipmaps(SLint target, SLuint index);
    SLbool               copyVideoImage(SLint       camWidth,
                                        SLint       camHeight,
                                        CVPixFormat glFormat,
                                        SLuchar*    data,
                                        SLbool      isContinuous,
                                        SLbool      isTopLeft);

    SLbool copyVideoImage(SLint       camWidth,
                          SLint       camHeight,
                          CVPixFormat srcFormat,
                          CVPixFormat dstFormat,
                          SLuchar*    data,
                          SLbool      isContinuous,
                          SLbool      isTopLeft);

    void calc3DGradients(SLint sampleRadius, function<void(int)> onUpdateProgress);
    void smooth3DGradients(SLint smoothRadius, function<void(int)> onUpdateProgress);

    // Bumpmap methods
    SLVec2f dudv(SLfloat u, SLfloat v); //! Returns the derivation as [s,t]

    // Statics
    static SLfloat maxAnisotropy;      //!< max. anisotropy available
    static SLuint  numBytesInTextures; //!< NO. of texture bytes on GPU

protected:
    // loading the image files
    void load(const SLstring& filename,
              SLbool          flipVertical           = true,
              SLbool          loadGrayscaleIntoAlpha = false);
    void load(const SLVCol4f& colors);

    CVVImage          _images;        //!< vector of CVImage pointers
    SLuint            _texID;         //!< OpenGL texture ID
    SLTextureType     _texType;       //!< [unknown, ColorMap, NormalMap, HeightMap, GlossMap]
    SLint             _min_filter;    //!< Minification filter
    SLint             _mag_filter;    //!< Magnification filter
    SLint             _wrap_s;        //!< Wrapping in s direction
    SLint             _wrap_t;        //!< Wrapping in t direction
    SLenum            _target;        //!< texture target
    SLMat4f           _tm;            //!< texture matrix
    SLuint            _bytesOnGPU;    //!< NO. of bytes on GPU
    SLbool            _autoCalcTM3D;  //!< flag if texture matrix should be calculated from AABB for 3D mapping
    SLfloat           _bumpScale;     //!< Bump mapping scale factor
    SLbool            _resizeToPow2;  //!< Flag if image should be resized to n^2
    SLGLVertexArray   _vaoSprite;     //!< Vertex array object for sprite rendering
    std::atomic<bool> _needsUpdate{}; //!< Flag if image needs an single update
    std::mutex        _mutex;         //!< Mutex to protect parallel access (used in ray tracing)

#ifdef SL_HAS_OPTIX
    CUgraphicsResource _cudaGraphicsResource; //!< Cuda Graphics object
    CUtexObject        _cudaTextureObject;
#endif
};
//-----------------------------------------------------------------------------
//! STL vector of SLGLTexture pointers
typedef vector<SLGLTexture*> SLVGLTexture;
//-----------------------------------------------------------------------------
#endif