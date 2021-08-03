#ifndef SENS_GL_TEXTURE_READER_H
#define SENS_GL_TEXTURE_READER_H

#include <string>
#include <opencv2/core/core.hpp>

//dont include opengl in the header so that opengl is not used if not needed
class SENSGLTextureReader
{
public:
    //! ATTENTION: make sure this constructor is called from gl thread
    SENSGLTextureReader(unsigned int textureId, bool isGlTextureExternal, int targetWidth, int targetHeight);
    ~SENSGLTextureReader();
    //! read gl texture from gpu. The transferred image will be filled. return false if it failed
    //! in this case one can retrieve the last error
    cv::Mat     readImageFromGpu();
    std::string getLastErrorMsg() const { return _lastErrorMsg; }

private:
    //! Checks if an OpenGL error occurred
    static void getGLError(const char* file, int line, bool quit);
    void        initGl();

    std::string _lastErrorMsg;
    //! id of externally generated texture
    unsigned int _extTextureId;
    bool         _isGlTextureExternal = false;
    int          _targetWidth;
    int          _targetHeight;

    unsigned int _fbo       = 0;
    unsigned int _prog      = 0;
    unsigned int _VBO       = 0;
    unsigned int _VAO       = 0;
    unsigned int _EBO       = 0;
    unsigned int _targetTex = 0;

    //! vector for errors collected in getGLError
    static std::vector<std::string> _errors;
};

#endif
