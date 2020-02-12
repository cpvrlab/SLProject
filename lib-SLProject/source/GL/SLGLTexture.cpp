//#############################################################################
//  File:      SLGLTexture.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h> // Must be the 1st include followed by  an empty line

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

#include <SLApplication.h>
#include <SLGLTexture.h>
#include <SLScene.h>
#include <Utils.h>

//-----------------------------------------------------------------------------
//! Default path for texture files used when only filename is passed in load.
//! Is overwritten in slCreateAppAndScene.
SLstring SLGLTexture::defaultPath      = SLstring(SL_PROJECT_ROOT) + "/data/images/textures/";
SLstring SLGLTexture::defaultPathFonts = SLstring(SL_PROJECT_ROOT) + "/data/images/fonts/";

//! maxAnisotropy=-1 show that GL_EXT_texture_filter_anisotropic is not checked
SLfloat SLGLTexture::maxAnisotropy = -1.0f;

//! NO. of texture byte allocated on GPU
SLuint SLGLTexture::numBytesInTextures = 0;
//-----------------------------------------------------------------------------
//! Default ctor for all stack instances (not created with new)
/*! Default ctor for all stack instances such as the video textures in SLScene
or the textures inherited by SLRaytracer. All other contructors add the this
pointer to the SLScene::_texture vector for global deallocation.
*/
SLGLTexture::SLGLTexture()
{
    _texID        = 0;
    _texType      = TT_unknown;
    _min_filter   = GL_NEAREST;
    _mag_filter   = GL_NEAREST;
    _wrap_s       = GL_REPEAT;
    _wrap_t       = GL_REPEAT;
    _target       = GL_TEXTURE_2D;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = false;
    _bytesOnGPU   = 0;
    _needsUpdate  = false;
}

//-----------------------------------------------------------------------------
//! ctor for empty 2D textures
SLGLTexture::SLGLTexture(SLint min_filter,
                         SLint mag_filter,
                         SLint wrapS,
                         SLint wrapT)
{
    _min_filter   = min_filter;
    _mag_filter   = mag_filter;
    _wrap_s       = wrapS;
    _wrap_t       = wrapT;
    _target       = GL_TEXTURE_2D;
    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = false;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;
    _texType      = TT_unknown;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}

//-----------------------------------------------------------------------------
//! ctor for 2D textures from byte pointer
SLGLTexture::SLGLTexture(unsigned char* data,
                         int            width,
                         int            height,
                         int            cvtype,
                         SLint          min_filter,
                         SLint          mag_filter,
                         SLTextureType  type,
                         SLint          wrapS,
                         SLint          wrapT)
{

    CVImage* image = new CVImage();
    image->load(width, height, PF_red, PF_red, data, true, false);

    _min_filter   = min_filter;
    _mag_filter   = mag_filter;
    _wrap_s       = wrapS;
    _wrap_t       = wrapT;
    _target       = GL_TEXTURE_2D;
    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = false;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;
    _texType      = type;

    _images.push_back(image);
    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}

//-----------------------------------------------------------------------------
//! ctor 2D textures with internal image allocation
SLGLTexture::SLGLTexture(const SLstring& filename,
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

    _min_filter   = min_filter;
    _mag_filter   = mag_filter;
    _wrap_s       = wrapS;
    _wrap_t       = wrapT;
    _target       = GL_TEXTURE_2D;
    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = false;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}
//-----------------------------------------------------------------------------
//! ctor for 3D texture
SLGLTexture::SLGLTexture(const SLVstring& files,
                         SLint            min_filter,
                         SLint            mag_filter,
                         SLint            wrapS,
                         SLint            wrapT,
                         const SLstring&  name,
                         SLbool           loadGrayscaleIntoAlpha) : SLObject(name)
{
    assert(files.size() > 1);

    _texType = TT_color;

    for (const auto& filename : files)
        load(filename, true, loadGrayscaleIntoAlpha);

    _min_filter   = min_filter;
    _mag_filter   = mag_filter;
    _wrap_s       = wrapS;
    _wrap_t       = wrapT;
    _target       = GL_TEXTURE_3D;
    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = true;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}
//-----------------------------------------------------------------------------
//! ctor for 1D texture
SLGLTexture::SLGLTexture(const SLVCol4f& colors,
                         SLint           min_filter,
                         SLint           mag_filter,
                         SLint           wrapS,
                         const SLstring& name) : SLObject(name)
{
    assert(colors.size() > 1);

    _texType = TT_color;

    load(colors);

    _min_filter = min_filter;
    _mag_filter = mag_filter;
    _wrap_s     = wrapS;
    _wrap_t     = wrapS;

    // OpenGL ES doesn't define 1D textures. We just make a 1 pixel high 2D texture
    _target = GL_TEXTURE_2D;

    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = true;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;

    // Add pointer to the global resource vectors for deallocation
    SLApplication::scene->textures().push_back(this);
}
//-----------------------------------------------------------------------------
//! ctor for cube mapping with internal image allocation
SLGLTexture::SLGLTexture(const SLstring& filenameXPos,
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

    _min_filter   = min_filter;
    _mag_filter   = mag_filter;
    _wrap_s       = GL_CLAMP_TO_EDGE; // other you will see filter artefacts on the edges
    _wrap_t       = GL_CLAMP_TO_EDGE; // other you will see filter artefacts on the edges
    _target       = GL_TEXTURE_CUBE_MAP;
    _texID        = 0;
    _bumpScale    = 1.0f;
    _resizeToPow2 = false;
    _autoCalcTM3D = false;
    _needsUpdate  = false;
    _bytesOnGPU   = 0;

    SLApplication::scene->textures().push_back(this);
}
//-----------------------------------------------------------------------------
SLGLTexture::~SLGLTexture()
{
    //SL_LOG("~SLGLTexture(%s)", name().c_str());
    clearData();
}
//-----------------------------------------------------------------------------
void SLGLTexture::clearData()
{
    glDeleteTextures(1, &_texID);

    numBytesInTextures -= _bytesOnGPU;

    for (auto& _image : _images)
    {
        delete _image;
        _image = nullptr;
    }
    _images.clear();

    _texID      = 0;
    _bytesOnGPU = 0;
    _vaoSprite.clearAttribs();
}
//-----------------------------------------------------------------------------
//! Loads the texture, converts color depth & applies vertical mirroring
void SLGLTexture::load(SLstring filename,
                       SLbool   flipVertical,
                       SLbool   loadGrayscaleIntoAlpha)
{
    // Load the file directly
    if (!Utils::fileExists(filename))
    {
        filename = defaultPath + filename;
        if (!Utils::fileExists(filename))
        {
            SLstring msg = "SLGLTexture: File not found: " + filename;
            SL_EXIT_MSG(msg.c_str());
        }
    }

    _images.push_back(new CVImage(filename,
                                  flipVertical,
                                  loadGrayscaleIntoAlpha));
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

    _images.push_back(new CVImage(col4f));
}
//-----------------------------------------------------------------------------
//! Copies the image data from a video camera into the current video image
/*!
\brief SLGLTexture::copyVideoImage
\param camWidth Width in pixels of the camera image
\param camHeight Height in pixels of the camera image
\param srcFormat Pixel format according to the OpenGL pixel formats
\param data Pointer to the first byte of the first pixel
\param isContinuous Flag if the next line comes after the last byte of the prev. line
\param isTopLeft Flag if the data pointer points to the top left pixel
\return Returns true if the texture was rebuilt
It is important that passed pixel format is either PF_LUMINANCE, RGB or RGBA.
otherwise an expensive conversion must be done.
*/
SLbool SLGLTexture::copyVideoImage(SLint       camWidth,
                                   SLint       camHeight,
                                   CVPixFormat srcFormat,
                                   SLuchar*    data,
                                   SLbool      isContinuous,
                                   SLbool      isTopLeft)
{
    // Add image for the first time
    if (_images.empty())
        _images.push_back(new CVImage(camWidth,
                                      camHeight,
                                      PF_rgb,
                                      "LiveVideoImageFromMemory"));

    // load returns true if size or format changes
    bool needsBuild = _images[0]->load(camWidth,
                                       camHeight,
                                       srcFormat,
                                       PF_rgb,
                                       data,
                                       isContinuous,
                                       isTopLeft);

    // OpenGL ES 2 only can resize non-power-of-two texture with clamp to edge
    _wrap_s = GL_CLAMP_TO_EDGE;
    _wrap_t = GL_CLAMP_TO_EDGE;

    if (needsBuild || _texID == 0)
    {
        SL_LOG("SLGLTexture::copyVideoImage: Rebuild: %d, %s",
               _texID,
               _images[0]->name().c_str());
        build();
    }

    _needsUpdate = true;
    return needsBuild;
}

SLbool SLGLTexture::copyVideoImage(SLint       camWidth,
                                   SLint       camHeight,
                                   CVPixFormat srcFormat,
                                   CVPixFormat dstFormat,
                                   SLuchar*    data,
                                   SLbool      isContinuous,
                                   SLbool      isTopLeft)
{
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

    // OpenGL ES 2 only can resize non-power-of-two texture with clamp to edge
    _wrap_s = GL_CLAMP_TO_EDGE;
    _wrap_t = GL_CLAMP_TO_EDGE;

    if (needsBuild || _texID == 0)
    {
        SL_LOG("SLGLTexture::copyVideoImage: Rebuild: %d, %s",
               _texID,
               _images[0]->name().c_str());
        build();
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
void SLGLTexture::build(SLint texID)
{
    assert(texID >= 0 && texID < 32);

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
        numBytesInTextures -= _bytesOnGPU;
    }

    // get max texture size
    SLint texMaxSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texMaxSize);

    // check if texture has to be resized
    if (_resizeToPow2)
    {
        SLuint w2 = closestPowerOf2(_images[0]->width());
        SLuint h2 = closestPowerOf2(_images[0]->height());
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
    stateGL->activeTexture(GL_TEXTURE0 + (SLuint)texID);

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
            cout << "GL_EXT_texture_filter_anisotropic not available.\n";
        }
    }

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

    // apply magnification filter only GL_NEAREST & GL_LINEAR is allowed
    glTexParameteri(_target, GL_TEXTURE_MAG_FILTER, _mag_filter);

    // apply texture wrapping modes
    glTexParameteri(_target, GL_TEXTURE_WRAP_S, _wrap_s);
    glTexParameteri(_target, GL_TEXTURE_WRAP_T, _wrap_t);
    glTexParameteri(_target, GL_TEXTURE_WRAP_R, _wrap_t);

    // Handle special stupid case on iOS
    SLint internalFormat = _images[0]->format();
    if (internalFormat == PF_red)
        internalFormat = GL_R8;

    // Build textures
    if (_target == GL_TEXTURE_2D)
    {
        //////////////////////////////////////////
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     internalFormat,
                     (SLsizei)_images[0]->width(),
                     (SLsizei)_images[0]->height(),
                     0,
                     _images[0]->format(),
                     GL_UNSIGNED_BYTE,
                     (GLvoid*)_images[0]->data());
        //////////////////////////////////////////

        GET_GL_ERROR;

        _bytesOnGPU += _images[0]->bytesPerImage();

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
        }

        numBytesInTextures += _bytesOnGPU;
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
                     0,              //Mipmap level,
                     internalFormat, //Internal format
                     (SLsizei)_images[0]->width(),
                     (SLsizei)_images[0]->height(),
                     (SLsizei)_images.size(),
                     0,                    //Border
                     _images[0]->format(), //Format
                     GL_UNSIGNED_BYTE,     //Data type
                     &buffer[0]);
        /////////////////////////////////////////////////////

        numBytesInTextures += _bytesOnGPU;
    }
    else if (_target == GL_TEXTURE_CUBE_MAP)
    {
        for (SLuint i = 0; i < 6; i++)
        {
            //////////////////////////////////////////////
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                         0,
                         internalFormat,
                         (SLsizei)_images[i]->width(),
                         (SLsizei)_images[i]->height(),
                         0,
                         _images[i]->format(),
                         GL_UNSIGNED_BYTE,
                         (GLvoid*)_images[i]->data());
            //////////////////////////////////////////////

            _bytesOnGPU += _images[0]->bytesPerImage();
        }

        numBytesInTextures += _bytesOnGPU;
        if (_min_filter >= GL_NEAREST_MIPMAP_NEAREST)
        {
            glGenerateMipmap(GL_TEXTURE_2D);

            // Mipmaps use 1/3 more memory on GPU
            _bytesOnGPU = (SLuint)((SLfloat)_bytesOnGPU * 1.333333333f);
        }
    }

    // Check if texture name is valid only for debug purpose
    //if (glIsTexture(_texName))
    //     SL_LOG("SLGLTexture::build: name: %u, unit-id: %u, Filename: %s", _texName, texID, _images[0]->name().c_str());
    //else SL_LOG("SLGLTexture::build: invalid name: %u, unit-id: %u, Filename: %s", _texName, texID, _images[0]->name().c_str());

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*!
SLGLTexture::bindActive binds the active texture. This method must be called
by the object that uses the texture every time BEFORE the its rendering.
The texID is only used for multi texturing. Before the first time the texture
is passed to OpenGL.
*/
void SLGLTexture::bindActive(SLint texID)
{
    assert(texID >= 0 && texID < 32);

    // if texture not exists build it
    if (!_texID)
        build(texID);

    if (_texID)
    {
        SLGLState* stateGL = SLGLState::instance();
        stateGL->activeTexture(GL_TEXTURE0 + (SLuint)texID);
        stateGL->bindTexture(_target, _texID);

        // Check if texture name is valid only for debug purpose
        //if (!glIsTexture(_texName))
        //{   SL_LOG("\n\n****** SLGLTexture::bindActive: Invalid texName: %u, texID: %u, File: %s\n",
        //           _texName, texID, _images[0]->name().c_str());
        //}

        if (_needsUpdate)
        {
            fullUpdate();
            _needsUpdate = false;
        }
    }

    GET_GL_ERROR;
}
//-----------------------------------------------------------------------------
/*!
Fully updates the OpenGL internal texture data by the image data
*/
void SLGLTexture::fullUpdate()
{
    if (_texID &&
        _images.size() &&
        _images[0]->data() &&
        _target == GL_TEXTURE_2D)
    {
        // Do not allow MIP-Maps as they are to big
        if (_min_filter == GL_NEAREST || _min_filter == GL_LINEAR)
        {
            numBytesInTextures -= _bytesOnGPU;

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
            numBytesInTextures += _bytesOnGPU;
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
void SLGLTexture::drawSprite(SLbool doUpdate)
{
    SLfloat w = (SLfloat)_images[0]->width();
    SLfloat h = (SLfloat)_images[0]->height();

    // build buffer object once
    if (!_vaoSprite.vaoID())
    {
        // Vertex X & Y of corners
        SLVVec2f P = {{0.0f, h},
                      {0.0f, 0.0f},
                      {w, h},
                      {w, 0.0f}};

        // Texture coords of corners
        SLVVec2f T = {{0.0f, 1.0f},
                      {0.0f, 0.0f},
                      {1.0f, 1.0f},
                      {1.0f, 0.0f}};

        // Indexes for a triangle strip
        SLVushort I = {0, 1, 2, 3};

        SLGLProgram* sp = SLApplication::scene->programs(SP_TextureOnly);
        sp->useProgram();
        _vaoSprite.setAttrib(AT_position, sp->getAttribLocation("a_position"), &P);
        _vaoSprite.setAttrib(AT_texCoord, sp->getAttribLocation("a_texCoord"), &T);
        _vaoSprite.setIndices(&I);
        _vaoSprite.generate(4);
    }

    bindActive(0);              // Enable & build texture
    if (doUpdate) fullUpdate(); // Update the OpenGL texture on each draw

    // Draw the character triangles
    SLGLState*   stateGL = SLGLState::instance();
    SLMat4f      mvp(stateGL->projectionMatrix * stateGL->modelViewMatrix);
    SLGLProgram* sp = SLApplication::scene->programs(SP_TextureOnly);
    sp->useProgram();
    sp->uniformMatrix4fv("u_mvpMatrix", 1, (SLfloat*)&mvp);
    sp->uniform1i("u_texture0", 0);
    sp->uniform1f("u_oneOverGamma", 1.0f);

    ////////////////////////////////////////////
    _vaoSprite.drawElementsAs(PT_triangleStrip);
    ////////////////////////////////////////////
}
//-----------------------------------------------------------------------------
//! SLGLTexture::getTexelf returns a pixel color from s & t texture coordinates.
/*! If the OpenGL filtering is set to GL_LINEAR a bilinear interpolated color out
of four neighboring pixels is return. Otherwise the nearest pixel is returned.
*/
SLCol4f SLGLTexture::getTexelf(SLfloat s, SLfloat t, SLuint imgIndex)
{
    assert(imgIndex < _images.size() && "Image index to big!");

    // transform tex coords with the texture matrix
    s = s * _tm.m(0) + _tm.m(12);
    t = t * _tm.m(5) + _tm.m(13);

    // Bilinear interpolation
    if (_min_filter == GL_LINEAR || _mag_filter == GL_LINEAR)
    {
        CVVec4f c4f = _images[imgIndex]->getPixelf(s, t);
        return SLCol4f(c4f[0], c4f[1], c4f[2], c4f[3]);
    }
    else
    {
        CVVec4f c4f = _images[imgIndex]->getPixeli((SLint)(s * _images[imgIndex]->width()),
                                                   (SLint)(t * _images[imgIndex]->height()));
        return SLCol4f(c4f[0], c4f[1], c4f[2], c4f[3]);
    }
}
//-----------------------------------------------------------------------------
//! SLGLTexture::getTexelf returns a pixel color at the specified cubemap direction
SLCol4f SLGLTexture::getTexelf(const SLVec3f& cubemapDir)
{
    assert(_images.size() == 6 &&
           _target == GL_TEXTURE_CUBE_MAP &&
           "SLGLTexture::getTexelf: Not a cubemap!");

    SLint   index;
    SLfloat u, v;

    cubeXYZ2UV(cubemapDir.x, cubemapDir.y, cubemapDir.z, index, u, v);

    return getTexelf(u, v, (SLuint)index);
}
//-----------------------------------------------------------------------------
/*!
dsdt calculates the partial derivation (gray value slope) at s,t for bump
mapping either from a height map or a normal map
*/
SLVec2f SLGLTexture::dsdt(SLfloat s, SLfloat t)
{
    SLVec2f dsdt(0, 0);
    SLfloat ds = 1.0f / _images[0]->width();
    SLfloat dt = 1.0f / _images[0]->height();

    if (_texType == TT_height)
    {
        dsdt.x = (getTexelf(s + ds, t).x - getTexelf(s - ds, t).x) * -_bumpScale;
        dsdt.y = (getTexelf(s, t + dt).x - getTexelf(s, t - dt).x) * -_bumpScale;
    }
    else if (_texType == TT_normal)
    {
        SLVec4f texel = getTexelf(s, t);
        dsdt.x        = texel.r * 2.0f - 1.0f;
        dsdt.y        = texel.g * 2.0f - 1.0f;
    }
    return dsdt;
}
//-----------------------------------------------------------------------------
//! Detects the texture type from the filename appendix (See SLTexType def.)
SLTextureType SLGLTexture::detectType(const SLstring& filename)
{
    // Check first our own texture name encoding
    SLstring name     = Utils::getFileNameWOExt(filename);
    SLstring appendix = name.substr(name.length() - 2, 2);
    if (appendix == "_C") return TT_color;
    if (appendix == "_N") return TT_normal;
    if (appendix == "_H") return TT_height;
    if (appendix == "_G") return TT_gloss;
    if (appendix == "_R") return TT_roughness;
    if (appendix == "_M") return TT_metallic;
    if (appendix == "_F") return TT_font;

    // Now check various formats found in the past
    name = Utils::toUpperString(name);

    if (Utils::containsString(name, "COL") ||
        Utils::containsString(name, "COLOR") ||
        Utils::containsString(name, "ALBEDO") ||
        Utils::containsString(name, "DIFFUSE") ||
        Utils::containsString(name, "DIFF") ||
        Utils::containsString(name, "DIF"))
        return TT_color;

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
        return TT_gloss;

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
        Utils::containsString(name, "OCCL") ||
        Utils::containsString(name, "OCC"))
        return TT_ambientOcc;

    // if nothing was detected so far we interpret it as a color texture
    SLstring msg = Utils::formatString("SLGLTexture::detectType: No type detected in file: %s",
                                       filename.c_str());
    SL_WARN_MSG(msg.c_str());

    return TT_color;
}
//-----------------------------------------------------------------------------
//! Returns the closest power of 2 to a passed number.
SLuint SLGLTexture::closestPowerOf2(SLuint num)
{
    SLuint nextPow2 = 1;
    if (num <= 0) return 1;

    while (nextPow2 <= num)
        nextPow2 <<= 1;
    SLuint prevPow2 = nextPow2 >> 1;

    if (num - prevPow2 < nextPow2 - num)
        return prevPow2;
    else
        return nextPow2;
}
//-----------------------------------------------------------------------------
//! Returns the next power of 2 to a passed number.
SLuint SLGLTexture::nextPowerOf2(SLuint num)
{
    SLuint nextPow2 = 1;
    if (num <= 0) return 1;

    while (nextPow2 <= num)
        nextPow2 <<= 1;
    return nextPow2;
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

        //SLfloat gauss[9] = {1.0f, 2.0f, 1.0f,
        //                    2.0f, 4.0f, 2.0f,
        //                    1.0f, 2.0f, 1.0f};

        //img2.convolve3x3(gauss);

        // Debug output
        //SLchar filename[255];
        //sprintf(filename,"%s_L%d_%dx%d.png", _name.c_str(), level, img2.width(), img2.height());
        //img2.savePNG(filename);

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
        case TT_unknown: return "Unknown type";
        case TT_color: return "color map";
        case TT_normal: return "normal map";
        case TT_height: return "hight map";
        case TT_gloss: return "gloss map";
        case TT_roughness: return "roughness map";
        case TT_metallic: return "metalness map";
        case TT_font: return "font map"; ;
        default: return "Unknown type";
    }
}
//-----------------------------------------------------------------------------
/*! SLGLTexture::calc3DGradients calculates the normals based on the 3D
gradient of all images and stores them in the RGB components.
\param sampleRadius Distance from center to calculate the gradient
*/
void SLGLTexture::calc3DGradients(SLint sampleRadius)
{
    SLint   r          = sampleRadius;
    SLint   volX       = (SLint)_images[0]->width();
    SLint   volY       = (SLint)_images[0]->height();
    SLint   volZ       = (SLint)_images.size();
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
            }
        }
    }

    smooth3DGradients(1);

    // Debug check
    //for (auto img : _images)
    //   img->savePNG(img->path() + "Normals_" + img->name());
}
//-----------------------------------------------------------------------------
/*! SLGLTexture::smooth3DGradients smooths the 3D gradients in the RGB channels
of all images.
\param smoothRadius Soothing radius
*/
void SLGLTexture::smooth3DGradients(SLint smoothRadius)
{
    SLint   r          = smoothRadius;
    SLint   volX       = (SLint)_images[0]->width();
    SLint   volY       = (SLint)_images[0]->height();
    SLint   volZ       = (SLint)_images.size();
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
//! Computes the uv and cubemap image index from a unnormalized vector x,y,z.
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
//------------------------------------------------------------------------------
