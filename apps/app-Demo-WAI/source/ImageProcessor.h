#ifndef IMAGE_PROCESSOR
#define IMAGE_PROCESSOR

#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <CVCalibration.h>

class ImageProcessor
{
public:

    GLuint yuv422Converter;
    GLuint RGBTexture;

    GLuint d2Gdx2;
    GLuint d2Gdy2;
    GLuint dGdx;
    GLuint dGdy;
    GLuint Gx;
    GLuint Gy;
    GLuint detH;
    GLuint nmsx;
    GLuint nmsy;
    GLuint nmsz;
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
    GLint nmszTexLoc;
    GLint nmszGxxLoc;
    GLint nmszGyyLoc;
    GLint nmszGxyLoc;
    GLint extractorWLoc;
    GLint extractorHLoc;
    GLint extractorTexLoc;
    GLint extractorAtomicCounterLoc;
    GLint extractorKpBufferLoc;

    GLuint renderTextures[12];
    GLuint renderFBO[12];
    GLuint outFBO[2];
    GLuint atomicCounters[2];
    GLuint kpBuffer;

    GLuint vao;
    GLuint vbo;
    GLuint vboi;
    GLuint outTexture[2];
    GLuint pbo[2];
    int curr;
    int ready;
    int m_w, m_h;

    ~ImageProcessor();
    ImageProcessor();
    ImageProcessor(int w, int h);

    void resetAtomicCounter();
    void init(int w, int h);
    void initAtomicCounters();
    void initShaders();
    void initVBO();
    void initTextureBuffers(int width, int height);
    void initKeypointBuffers(int nb_elements);
    void initFBO();
    void setTextureParameters();

    void textureRGBAI(int w, int h);
    void textureRGBF(int w, int h);
    void textureRF(int w, int h);
    void textureRB(int w, int h);
    GLuint buildShaderFromSource(string source, GLenum shaderType);

    void gxx(int w, int h);
    void gyy(int w, int h);
    void gxy(int w, int h);
    void det(int w, int h);
    void nms(int w, int h);
    void extract(int w, int h, int curr);
    void gpu_kp();
    //void readResult(cv::Mat * tex);
    void readResult(std::vector<cv::KeyPoint> &kps);

    void initComputeShader();
    void computeBRIEF();
};

#endif
