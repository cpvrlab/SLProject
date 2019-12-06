#ifndef GLSL_HESSIAN
#define GLSL_HESSIAN

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <CVCalibration.h>

class GLSLHessian
{
public:

    GLuint yuv422Converter;
    GLuint RGBTexture;
    GLuint patternTexture;

    GLuint d2Gdx2;
    GLuint d2Gdy2;
    GLuint dGdx;
    GLuint dGdy;
    GLuint Gx;
    GLuint Gy;
    GLuint detH;
    GLuint nmsx;
    GLuint nmsy;
    GLuint edge;
    GLuint extractor;

    GLint d2Gdx2WLoc;
    GLint d2Gdy2WLoc;
    GLint dGdxWLoc;
    GLint dGdyWLoc;
    GLint GxWLoc;
    GLint GyWLoc;
    GLint nmsxWLoc;
    GLint nmsyWLoc;
    GLint d2Gdx2TexLoc;
    GLint d2Gdy2TexLoc;
    GLint dGdxTexLoc;
    GLint dGdyTexLoc;
    GLint GxTexLoc;
    GLint GyTexLoc;
    GLint Gx1chTexLoc;
    GLint Gy1chTexLoc;
    GLint detHGxxLoc;
    GLint detHGyyLoc;
    GLint detHGxyLoc;
    GLint nmsxTexLoc;
    GLint nmsyTexLoc;
    GLint edgeWLoc;
    GLint edgeHLoc;
    GLint edgeTexLoc;
    GLint edgeDetLoc;
    GLint edgeGxxLoc;
    GLint edgeGyyLoc;

    GLint extractorWLoc;
    GLint extractorHLoc;
    GLint extractorTexLoc;
    GLint extractorPatternLoc;
    GLint extractorOffsetLoc;
    GLint extractorSizeLoc;
    GLint extractorIdxLoc;
    GLint extractorLowCountersLoc;
    GLint extractorMediumCountersLoc;
    GLint extractorHighCountersLoc;
    GLint extractorLowImageLoc;
    GLint extractorMediumImageLoc;
    GLint extractorHighImageLoc;

    bool externalTexture;
    GLuint renderTextures[12];
    GLuint renderFBO[12];

    GLuint atomicCounter;
    GLuint highImagesFB[2];
    GLuint mediumImagesFB[2];
    GLuint lowImagesFB[2];

    GLuint highImages[2];
    GLuint mediumImages[2];
    GLuint lowImages[2];

    GLuint highImagePBOs[2];
    GLuint mediumImagePBOs[2];
    GLuint lowImagePBOs[2];

    GLuint vao;
    GLuint vbo;
    GLuint vboi;
    int curr;
    int ready;
    int m_w, m_h;
    int mNbKeypointsLow;
    int mNbKeypointsMedium;
    int mNbKeypointsHigh;
    string thresholdStr;

    std::string gaussianKernelStr;
    std::string gaussianD1KernelStr;
    std::string gaussianD2KernelStr;
    std::string kernelSizeStr;

    std::string nbKeypointsLowStr;
    std::string nbKeypointsMediumStr;
    std::string nbKeypointsHighStr;

    ~GLSLHessian();
    GLSLHessian();
    GLSLHessian(int w, int h, int nbKeypointsLow, int nbKeypointsMedium, int nbKeypointsHigh, float thrs, float lowSigma, float mediumSigma, float highSigma);

    string gaussian(int size, int halfSize, float sigma);
    string gaussianD1(int size, int halfSize, float sigma);
    string gaussianD2(int size, int halfSize, float sigma);

    void init(int w, int h, int nbKeypointsLow, int nbKeypointsMedium, int nbKeypointsHigh, float thrs, float lowSigma, float mediumSigma, float highSigma);
    void initShaders();
    void initVBO();
    void initTextureBuffers(int width, int height);

    void clearCounterBuffer();
    void initKeypointBuffers();
    void initFBO();
    void initPattern();
    void setTextureParameters();

    void textureRGBAF(int w, int h);
    void textureRGBA(int w, int h);
    void textureRGBF(int w, int h);
    void textureRF(int w, int h);
    void textureR(int w, int h);
    GLuint buildShaderFromSource(string source, GLenum shaderType);

    void gxx(int w, int h);
    void gyy(int w, int h);
    void gxy(int w, int h);
    void det(int w, int h);
    void nms(int w, int h);
    void extract(int w, int h, int curr);

    void setInputTexture(SLGLTexture &tex);
    void setInputTexture(cv::Mat &image);
    void gpu_kp();
    void readResult(std::vector<cv::KeyPoint> &kps);

    void initComputeShader();
    void computeBRIEF();
};

#endif
