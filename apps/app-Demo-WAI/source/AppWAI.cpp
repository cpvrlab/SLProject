#include <atomic>
#include <SLApplication.h>
#include <SLInterface.h>
#include <SLKeyframeCamera.h>
#include <CVCapture.h>
#include <Utils.h>
#include <AverageTiming.h>

#include <WAIModeOrbSlam2.h>

#include <WAIMapStorage.h>

#include <WAICalibration.h>
#include <AppWAIScene.h>
#include <AppDemoGui.h>
#include <AppDemoGuiMenu.h>
#include <AppDemoGuiInfosDialog.h>
#include <AppDemoGuiAbout.h>
#include <AppDemoGuiInfosFrameworks.h>
#include <AppDemoGuiInfosMapNodeTransform.h>
#include <AppDemoGuiInfosScene.h>
#include <AppDemoGuiInfosSensors.h>
#include <AppDemoGuiInfosTracking.h>
#include <AppDemoGuiMapStorage.h>
#include <AppDemoGuiProperties.h>
#include <AppDemoGuiSceneGraph.h>
#include <AppDemoGuiStatsDebugTiming.h>
#include <AppDemoGuiStatsTiming.h>
#include <AppDemoGuiStatsVideo.h>
#include <AppDemoGuiTrackedMapping.h>
#include <AppDemoGuiTransform.h>
#include <AppDemoGuiUIPrefs.h>
#include <AppDemoGuiVideoControls.h>
#include <AppDemoGuiVideoStorage.h>
#include <AppDemoGuiSlamLoad.h>
#include <AppDemoGuiTestOpen.h>
#include <AppDemoGuiTestWrite.h>
#include <AppDemoGuiSlamParam.h>
#include <AppWAI.h>
#include <AppDirectories.h>

AppDemoGuiAbout* WAIApp::aboutDial = nullptr;
AppDemoGuiError* WAIApp::errorDial = nullptr;

GUIPreferences     WAIApp::uiPrefs;
SLGLTexture*       WAIApp::cpvrLogo   = nullptr;
SLGLTexture*       WAIApp::videoImage = nullptr;
AppWAIDirectories* WAIApp::dirs       = nullptr;
AppWAIScene*       WAIApp::waiScene   = nullptr;
WAICalibration*    WAIApp::wc         = nullptr;
int                WAIApp::scrWidth;
int                WAIApp::scrHeight;
int                WAIApp::defaultScrWidth;
int                WAIApp::defaultScrHeight;
float              WAIApp::scrWdivH;
cv::VideoWriter*   WAIApp::videoWriter     = nullptr;
cv::VideoWriter*   WAIApp::videoWriterInfo = nullptr;
WAI::ModeOrbSlam2* WAIApp::mode            = nullptr;
bool               WAIApp::loaded          = false;
ofstream           WAIApp::gpsDataStream;
WAIApp::GLSLKP     WAIApp::glslKP;
SLGLTexture*       WAIApp::testTexture;
unsigned char * WAIApp::outputTexture;

std::string WAIApp::videoDir       = "";
std::string WAIApp::calibDir       = "";
std::string WAIApp::mapDir         = "";
std::string WAIApp::vocDir         = "";
std::string WAIApp::experimentsDir = "";

bool WAIApp::resizeWindow = false;

bool WAIApp::pauseVideo           = false;
int  WAIApp::videoCursorMoveIndex = 0;


GLuint WAIApp::buildShaderFromSource(string source,
                                     GLenum shaderType)
{
    // Compile Shader code
    GLuint      shaderHandle = glCreateShader(shaderType);
    const char* src          = source.c_str();
    glShaderSource(shaderHandle, 1, &src, nullptr);
    glCompileShader(shaderHandle);

    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle,
                  GL_COMPILE_STATUS,
                  &compileSuccess);

    GLint logSize = 0;
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &logSize);

    GLchar * log = new GLchar[logSize];

    glGetShaderInfoLog(shaderHandle, logSize, nullptr, log);

    if (!compileSuccess)
    {
        Utils::log("Cannot compile shader %s\n", log);

        std::cout << source << std::endl;
        exit(1);
    }
    return shaderHandle;
}

void WAIApp::initTestProgram()
{
    glslKP.curr = 1;
    glslKP.ready = 0;
    glGenFramebuffers(8, glslKP.renderFBO);
    glGenFramebuffers(2, glslKP.framebuffers);
    glGenBuffers(2, glslKP.pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, glslKP.pbo[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, scrWidth * scrHeight, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, glslKP.pbo[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, scrWidth * scrHeight, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glslKP.d2Gdx2 = glCreateProgram();
    glslKP.d2Gdy2 = glCreateProgram();
    glslKP.dGdx   = glCreateProgram();
    glslKP.dGdy   = glCreateProgram();
    glslKP.Gx     = glCreateProgram();
    glslKP.Gy     = glCreateProgram();
    glslKP.detH   = glCreateProgram();
    glslKP.nmsx   = glCreateProgram();
    glslKP.nmsy   = glCreateProgram();
    glslKP.nmsz   = glCreateProgram();

#ifdef ANDROID
    std::string screeQuadVs = "#version 320 es\n"
                              "in vec3 vcoords;\n"
                              "out mediump vec2 texcoords;\n"
                              "\n"
                              "void main()\n"
                              "{\n"
                              "    texcoords = 0.5 * (vcoords.xy + vec2(1.0));\n"
                              "    gl_Position = vec4(vcoords, 1.0);\n"
                              "}\n" ;

    std::string hLaplacianFs = "#version 320 es\n"
                               "precision mediump float;\n"
                               "out mediump float pixel;\n"
                               "in mediump vec2 texcoords;\n"
                               "uniform mediump float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel[11] = float[11](1.37959207955, "
                               "3.07380807807, 3.88509751298, 0.92611021675, "
                               "-5.34775050133, -8.83371477203, -5.34775050133, "
                               "0.92611021675, 3.88509751298, 3.07380807807, 1.37959207955);\n"
                               "\n"
                               "    \n"
                               "    float val = 0.0;\n"
                               "    for (int i = 0; i < 11; i++)\n"
                               "    {\n"
                               "        float p = float(i);\n"
                               "\n"
                               "\n"
                               "\n"
                               "\n"
                               "        vec2 offset = vec2(float(p - 5.0f) / w, 0.0f);\n"
                               "\n"
                               "\n"
                               "\n"
                               "        val += kernel[i] * float(texture(tex, texcoords + offset).r);\n"
                               "    }\n"
                               "    pixel = val;\n"
                               "}\n";

    std::string vLaplacianFs = "#version 320 es\n"
                               "precision mediump float;\n"
                               "out mediump float pixel;\n"
                               "in mediump vec2 texcoords;\n"
                               "uniform mediump float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "const float kernel[11] = float[11](1.37959207955, "
                               "3.07380807807, 3.88509751298, 0.92611021675, "
                               "-5.34775050133, -8.83371477203, -5.34775050133, "
                               "0.92611021675, 3.88509751298, 3.07380807807, 1.37959207955);\n"
                               "\n"
                               "    \n"
                               "    float val = 0.0;\n"
                               "\n"
                               "\n"
                               "\n"
                               "    for (int i = 0; i < 11; i++)\n"
                               "    {\n"
                               "        float p = float(i);\n"
                               "        vec2 offset = vec2(0.0f, float(p - 5.0f) / w);\n"
                               "\n"
                               "\n"
                               "\n"
                               "        val += kernel[i] * float(texture(tex, texcoords + offset).r);\n"
                               "    }\n"
                               "    pixel = val;\n"
                               "}\n";
#else
    std::string screeQuadVs = "#version 330\n"
                              "layout (location = 1) in vec3 vcoords;\n"
                              "out vec2 texcoords;\n"
                              "\n"
                              "void main()\n"
                              "{\n"
                              "    texcoords = 0.5 * (vcoords.xy + vec2(1.0));\n"
                              "    gl_Position = vec4(vcoords, 1.0);\n"
                              "}\n" ;


    std::string hGaussianFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "    0.007614419169296346, 0.03607496968918392, 0.10958608179781393, "
                               "    0.2134445419434044, 0.26655997480060273, 0.21344454194340445, "
                               "    0.109586081797814, 0.036074969689183944, 0.007614419169296356);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "    0.0031742033144480037, 0.008980510024247402, 0.02165110898093487, "
                               "    0.04448075733770272, 0.07787123866346017, 0.11617023707406768, "
                               "    0.1476813151730447, 0.15998125886418896, 0.14768131517304472, "
                               "    0.11617023707406769, 0.07787123866346018, 0.04448075733770272, "
                               "    0.02165110898093487, 0.008980510024247402, 0.0031742033144480037);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "    0.0019290645132252363, 0.004189349123089384, 0.008384820035189703, "
                               "    0.015466367540072906, 0.02629240397422038, 0.041192642776781994, "
                               "    0.05947800651444567, 0.07914810874862578, 0.09706710312973113, "
                               "    0.10971120494447856, 0.11428185740027867, 0.10971120494447856, "
                               "    0.09706710312973113, 0.07914810874862578, 0.05947800651444567, "
                               "    0.041192642776781994, 0.02629240397422038, 0.015466367540072906, "
                               "    0.008384820035189703, 0.004189349123089384, 0.0019290645132252363);\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).b;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).b;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(ofst, 0.0)).g;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(ofst, 0.0)).b;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).g;\n"
                               "        response.r += kernel12[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(ofst, 0.0)).g;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).b;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).b;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";

    std::string vGaussianFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "    0.007614419169296346, 0.03607496968918392, 0.10958608179781393, "
                               "    0.2134445419434044, 0.26655997480060273, 0.21344454194340445, "
                               "    0.109586081797814, 0.036074969689183944, 0.007614419169296356);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "    0.0031742033144480037, 0.008980510024247402, 0.02165110898093487, "
                               "    0.04448075733770272, 0.07787123866346017, 0.11617023707406768, "
                               "    0.1476813151730447, 0.15998125886418896, 0.14768131517304472, "
                               "    0.11617023707406769, 0.07787123866346018, 0.04448075733770272, "
                               "    0.02165110898093487, 0.008980510024247402, 0.0031742033144480037);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "    0.0019290645132252363, 0.004189349123089384, 0.008384820035189703, "
                               "    0.015466367540072906, 0.02629240397422038, 0.041192642776781994, "
                               "    0.05947800651444567, 0.07914810874862578, 0.09706710312973113, "
                               "    0.10971120494447856, 0.11428185740027867, 0.10971120494447856, "
                               "    0.09706710312973113, 0.07914810874862578, 0.05947800651444567, "
                               "    0.041192642776781994, 0.02629240397422038, 0.015466367540072906, "
                               "    0.008384820035189703, 0.004189349123089384, 0.0019290645132252363);\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "        response.r += kernel12[i] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";

    std::string hGaussianDxFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.02030511778479025, 0.07214993937836783, 0.14611477573041856, 0.14229636129560294, 4.9323503577855103e-17, -0.1422963612956029, -0.14611477573041862, -0.07214993937836786, -0.02030511778479028);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.00888776928045441, 0.021553224058193758, 0.04330221796186974, 0.07116921174032437, 0.09344548639615223, 0.09293618965925417, 0.0590725260692179, 1.7761487709954e-17, -0.059072526069217875, -0.09293618965925415, -0.09344548639615223, -0.07116921174032437, -0.04330221796186974, -0.021553224058193758, -0.00888776928045441);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.005511612894929245, 0.01077261203080127, 0.01916530293757646, 0.03093273508014581, 0.04507269252723493, 0.058846632538259974, 0.06797486458793789, 0.06784123607025065, 0.05546691607413207, 0.0313460585555653, -0.0, -0.0313460585555653, -0.05546691607413207, -0.06784123607025065, -0.06797486458793789, -0.058846632538259974, -0.04507269252723493, -0.03093273508014581, -0.01916530293757646, -0.01077261203080127, -0.005511612894929245);\n"
                               "\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.r += kernel12[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";


    std::string vGaussianDyFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.02030511778479025, 0.07214993937836783, 0.14611477573041856, 0.14229636129560294, 4.9323503577855103e-17, -0.1422963612956029, -0.14611477573041862, -0.07214993937836786, -0.02030511778479028);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.00888776928045441, 0.021553224058193758, 0.04330221796186974, 0.07116921174032437, 0.09344548639615223, 0.09293618965925417, 0.0590725260692179, 1.7761487709954e-17, -0.059072526069217875, -0.09293618965925415, -0.09344548639615223, -0.07116921174032437, -0.04330221796186974, -0.021553224058193758, -0.00888776928045441);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.005511612894929245, 0.01077261203080127, 0.01916530293757646, 0.03093273508014581, 0.04507269252723493, 0.058846632538259974, 0.06797486458793789, 0.06784123607025065, 0.05546691607413207, 0.0313460585555653, -0.0, -0.0313460585555653, -0.05546691607413207, -0.06784123607025065, -0.06797486458793789, -0.058846632538259974, -0.04507269252723493, -0.03093273508014581, -0.01916530293757646, -0.01077261203080127, -0.005511612894929245);\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "        response.g += kernel20[i  ] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "        response.r += kernel12[i  ] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i  ] * texture(tex, texcoords + vec2(0.0, ofst)).g;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).b;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";


    std::string hGaussianDx2Fs  = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.04653256159014432, 0.10822490906755174, 0.08523361917607754, -0.11858030107966908, -0.2665599748006027, -0.1185803010796692, 0.08523361917607745, 0.10822490906755174, 0.046532561590144364);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.021711550670824344, 0.04274722771541763, 0.0649533269428046, 0.06938998144681627, 0.034263345011922505, -0.04182128534666434, -0.12405230474535753, -0.15998125886418896, -0.12405230474535757, -0.041821285346664384, 0.03426334501192248, 0.06938998144681627, 0.0649533269428046, 0.04274722771541763, 0.021711550670824344);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.013818400900858318, 0.02351165324182817, 0.035421586679270776, 0.046399102620218693, 0.05097506892961091, 0.04287397513501796, 0.018207553014626197, -0.020998477831268087, -0.06537172251594138, -0.10075518821431705, -0.11428185740027866, -0.10075518821431705, -0.06537172251594138, -0.020998477831268087, 0.018207553014626197, 0.04287397513501796, 0.05097506892961091, 0.046399102620218693, 0.035421586679270776, 0.02351165324182817, 0.013818400900858318);\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.g += kernel20[i  ] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.r += kernel12[i  ] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i  ] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";

    std::string vGaussianDy2Fs  = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.04653256159014432, 0.10822490906755174, 0.08523361917607754, -0.11858030107966908, -0.2665599748006027, -0.1185803010796692, 0.08523361917607745, 0.10822490906755174, 0.046532561590144364);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.021711550670824344, 0.04274722771541763, 0.0649533269428046, 0.06938998144681627, 0.034263345011922505, -0.04182128534666434, -0.12405230474535753, -0.15998125886418896, -0.12405230474535757, -0.041821285346664384, 0.03426334501192248, 0.06938998144681627, 0.0649533269428046, 0.04274722771541763, 0.021711550670824344);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.013818400900858318, 0.02351165324182817, 0.035421586679270776, 0.046399102620218693, 0.05097506892961091, 0.04287397513501796, 0.018207553014626197, -0.020998477831268087, -0.06537172251594138, -0.10075518821431705, -0.11428185740027866, -0.10075518821431705, -0.06537172251594138, -0.020998477831268087, 0.018207553014626197, 0.04287397513501796, 0.05097506892961091, 0.046399102620218693, 0.035421586679270776, 0.02351165324182817, 0.013818400900858318);\n"
                               "\n"
                               "    \n"
                               "    vec3 response = vec3(0.0);\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 3; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "        response.g += kernel20[i  ] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 0; i < 9; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 4.0) / w;\n"
                               "        response.b += kernel28[i+6] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "        response.g += kernel20[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "        response.r += kernel12[i  ] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 12; i < 15; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 7.0) / w;\n"
                               "        response.g += kernel20[i] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "        response.b += kernel28[i+3] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    for (int i = 18; i < 21; i++)\n"
                               "    {\n"
                               "        float ofst = (float(i) - 10.0) / w;\n"
                               "        response.b += kernel28[i] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                               "    }\n"
                               "    pixel = response;\n"
                               "}\n";

    std::string detHFs = "#version 330\n"
                        "out float pixel;\n"
                        "in vec2 texcoords;\n"
                        "uniform sampler2D tgxx;\n"
                        "uniform sampler2D tgyy;\n"
                        "uniform sampler2D tgxy;\n"
                        "\n"
                        "void main()\n"
                        "{\n"
                        "    \n"
                        "    vec3 gxx = texture(tgxx, texcoords).rgb;\n"
                        "    vec3 gyy = texture(tgyy, texcoords).rgb;\n"
                        "    vec3 gxy = texture(tgxy, texcoords).rgb;\n"
                        "    vec3 det = abs(gxx * gyy - gxy * gxy);\n"
                        "    vec3 trace = gxx + gyy;\n"
                        "    vec3 r = trace*trace / det;\n"
                        "    float maxd = max(det.x, max(det.y, det.z));\n"
                        "    float maxr = max(r.x, max(r.y, r.z));\n"
                        "    float minr = min(r.x, min(r.y, r.z));\n"
                        "    pixel = 0.0;\n"
                        "    if (maxr < 1.05 && minr > 0.95) { pixel = 100.0 * maxd; }\n"
                        "}\n";

    std::string nmsxFs = "#version 330\n"
                        "out float pixel;\n"
                        "in vec2 texcoords;\n"
                        "uniform sampler2D tex;\n"
                        "uniform float w;\n"
                        "\n"
                        "void main()\n"
                        "{\n"
                        "    float o = texture(tex, texcoords).r;\n"
                        "    float px = texture(tex, texcoords + vec2(1.0/w, 0.0f)).r;\n"
                        "    float nx = texture(tex, texcoords - vec2(1.0/w, 0.0f)).r;\n"
                        "    pixel = 1.0;\n"
                        "    if (o <= nx || o <= px)\n"
                        "    {\n"
                        "       pixel = 0.0;\n"
                        "    }\n"
                        "}\n";

    std::string nmsyFs = "#version 330\n"
                        "out float pixel;\n"
                        "in vec2 texcoords;\n"
                        "uniform sampler2D tex;\n"
                        "uniform float w;\n"
                        "\n"
                        "void main()\n"
                        "{\n"
                        "    \n"
                        "    float o = texture(tex, texcoords).r;\n"
                        "    float py = texture(tex, texcoords + vec2(0.0f, 1.0/w)).r;\n"
                        "    float ny = texture(tex, texcoords - vec2(0.0f, 1.0/w)).r;\n"
                        "    pixel = 1.0;\n"
                        "    if (o <= ny || o <= py)\n"
                        "    {\n"
                        "       pixel = 0.0;\n"
                        "    }\n"
                        "}\n";

    std::string nmszFs = "#version 330\n"
                         "out float pixel;\n"
                         "in vec2 texcoords;\n"
                         "uniform sampler2D tex;\n"
                         "\n"
                         "void main()\n"
                         "{\n"
                         "    \n"
                         "    float o = texture(tex, texcoords).g;\n"
                         "    float pz = texture(tex, texcoords).r;\n"
                         "    float nz = texture(tex, texcoords).b;\n"
                         "    pixel = max(nz, max(o, pz));\n"
                         "}\n";

#endif

    GLuint vscreenQuad = buildShaderFromSource(screeQuadVs, GL_VERTEX_SHADER);
    GLuint fd2Gdx2 = buildShaderFromSource(hGaussianDx2Fs, GL_FRAGMENT_SHADER);
    GLuint fd2Gdy2 = buildShaderFromSource(vGaussianDy2Fs, GL_FRAGMENT_SHADER);
    GLuint fdGdx = buildShaderFromSource(hGaussianDxFs, GL_FRAGMENT_SHADER);
    GLuint fdGdy = buildShaderFromSource(vGaussianDyFs, GL_FRAGMENT_SHADER);
    GLuint fGx = buildShaderFromSource(hGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fGy = buildShaderFromSource(vGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fdetH = buildShaderFromSource(detHFs, GL_FRAGMENT_SHADER);
    GLuint fnmsx = buildShaderFromSource(nmsxFs, GL_FRAGMENT_SHADER);
    GLuint fnmsy = buildShaderFromSource(nmsyFs, GL_FRAGMENT_SHADER);
    GLuint fnmsz = buildShaderFromSource(nmszFs, GL_FRAGMENT_SHADER);

    glAttachShader(glslKP.d2Gdx2, vscreenQuad);
    glAttachShader(glslKP.d2Gdx2, fd2Gdx2);
    glLinkProgram(glslKP.d2Gdx2);

    glAttachShader(glslKP.d2Gdy2, vscreenQuad);
    glAttachShader(glslKP.d2Gdy2, fd2Gdy2);
    glLinkProgram(glslKP.d2Gdy2);

    glAttachShader(glslKP.dGdx, vscreenQuad);
    glAttachShader(glslKP.dGdx, fdGdx);
    glLinkProgram(glslKP.dGdx);

    glAttachShader(glslKP.dGdy, vscreenQuad);
    glAttachShader(glslKP.dGdy, fdGdy);
    glLinkProgram(glslKP.dGdy);

    glAttachShader(glslKP.Gx, vscreenQuad);
    glAttachShader(glslKP.Gx, fGx);
    glLinkProgram(glslKP.Gx);

    glAttachShader(glslKP.Gy, vscreenQuad);
    glAttachShader(glslKP.Gy, fGy);
    glLinkProgram(glslKP.Gy);

    glAttachShader(glslKP.detH, vscreenQuad);
    glAttachShader(glslKP.detH, fdetH);
    glLinkProgram(glslKP.detH);

    glAttachShader(glslKP.nmsx, vscreenQuad);
    glAttachShader(glslKP.nmsx, fnmsx);
    glLinkProgram(glslKP.nmsx);

    glAttachShader(glslKP.nmsy, vscreenQuad);
    glAttachShader(glslKP.nmsy, fnmsy);
    glLinkProgram(glslKP.nmsy);

    glAttachShader(glslKP.nmsz, vscreenQuad);
    glAttachShader(glslKP.nmsz, fnmsz);
    glLinkProgram(glslKP.nmsz);

    GLuint vertexLoc = glGetAttribLocation(glslKP.d2Gdx2, "vcoords");
    glslKP.d2Gdx2TexLoc = glGetUniformLocation(glslKP.d2Gdx2, "tex");
    glslKP.d2Gdx2WLoc = glGetUniformLocation(glslKP.d2Gdx2, "w");
    glslKP.d2Gdy2TexLoc = glGetUniformLocation(glslKP.d2Gdy2, "tex");
    glslKP.d2Gdy2WLoc = glGetUniformLocation(glslKP.d2Gdy2, "w");
    glslKP.dGdxTexLoc = glGetUniformLocation(glslKP.dGdx, "tex");
    glslKP.dGdxWLoc = glGetUniformLocation(glslKP.dGdx, "w");
    glslKP.dGdyTexLoc = glGetUniformLocation(glslKP.dGdy, "tex");
    glslKP.dGdyWLoc = glGetUniformLocation(glslKP.dGdy, "w");
    glslKP.GxTexLoc = glGetUniformLocation(glslKP.Gx, "tex");
    glslKP.GxWLoc = glGetUniformLocation(glslKP.Gx, "w");
    glslKP.GyTexLoc = glGetUniformLocation(glslKP.Gy, "tex");
    glslKP.GyWLoc = glGetUniformLocation(glslKP.Gy, "w");
    glslKP.detHGxxLoc = glGetUniformLocation(glslKP.detH, "tgxx");
    glslKP.detHGyyLoc = glGetUniformLocation(glslKP.detH, "tgyy");
    glslKP.detHGxyLoc = glGetUniformLocation(glslKP.detH, "tgxy");
    glslKP.nmsxTexLoc = glGetUniformLocation(glslKP.nmsx, "tex");
    glslKP.nmsxWLoc = glGetUniformLocation(glslKP.nmsx, "w");
    glslKP.nmsyTexLoc = glGetUniformLocation(glslKP.nmsy, "tex");
    glslKP.nmsyWLoc = glGetUniformLocation(glslKP.nmsy, "w");
    glslKP.nmszTexLoc = glGetUniformLocation(glslKP.nmsz, "tex");

    float vertices[12] = {-1, -1,  0,
                           1, -1,  0,
                           1,  1,  0,
                          -1,  1,  0 };

    GLuint indices[6] = {0, 1, 2, 2, 3, 0};

    //Gen VBO
    glGenBuffers(1, &glslKP.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, glslKP.vbo);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &glslKP.vboi);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(float), indices, GL_STATIC_DRAW);

    // Gen VAO
    glGenVertexArrays(1, &glslKP.vao);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ARRAY_BUFFER, glslKP.vbo);
    glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(vertexLoc);
    glBindVertexArray(0);

    glGenTextures(1, &glslKP.grayTexture);
    glGenTextures(8, glslKP.renderTextures);
    glGenTextures(2, glslKP.outTextures);

    glBindTexture(GL_TEXTURE_2D, glslKP.grayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexImage2D(GL_TEXTURE_2D,    // target texture type 1D, 2D or 3D
                 0,                // Base level for mipmapped textures
                 GL_RED,           // internal format: e.g. GL_RGBA, see spec.
                 scrWidth,         // image width
                 scrHeight,        // image height
                 0,                // border pixels: must be 0
                 GL_RED,           // data format: e.g. GL_RGBA, see spec.
                 GL_UNSIGNED_BYTE, // data type
                 nullptr);         // image data pointer

    for (int i = 0; i < 6; i++)
    {
        glBindTexture(GL_TEXTURE_2D, glslKP.renderTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        glTexImage2D(GL_TEXTURE_2D,    // target texture type 1D, 2D or 3D
                     0,                // Base level for mipmapped textures
                     GL_RGB,           // internal format: e.g. GL_RGBA, see spec.
                     scrWidth,         // image width
                     scrHeight,        // image height
                     0,                // border pixels: must be 0
                     GL_RGB,           // data format: e.g. GL_RGBA, see spec.
                     GL_UNSIGNED_BYTE, // data type
                     nullptr);         // image data pointer
    }

    for (int i = 6; i < 8; i++)
    {
        glBindTexture(GL_TEXTURE_2D, glslKP.renderTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        glTexImage2D(GL_TEXTURE_2D,    // target texture type 1D, 2D or 3D
                     0,                // Base level for mipmapped textures
                     GL_RED,           // internal format: e.g. GL_RGBA, see spec.
                     scrWidth,         // image width
                     scrHeight,        // image height
                     0,                // border pixels: must be 0
                     GL_RED,           // data format: e.g. GL_RGBA, see spec.
                     GL_UNSIGNED_BYTE, // data type
                     nullptr);         // image data pointer
    }
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, glslKP.outTextures[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        glTexImage2D(GL_TEXTURE_2D,    // target texture type 1D, 2D or 3D
                     0,                // Base level for mipmapped textures
                     GL_RED,           // internal format: e.g. GL_RGBA, see spec.
                     scrWidth,         // image width
                     scrHeight,        // image height
                     0,                // border pixels: must be 0
                     GL_RED,           // data format: e.g. GL_RGBA, see spec.
                     GL_UNSIGNED_BYTE, // data type
                     nullptr);         // image data pointer
    }

    for (int i = 0; i < 8; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glslKP.renderTextures[i], 0);
    }

    for (int i = 0; i < 2; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, glslKP.framebuffers[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glslKP.outTextures[i], 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void WAIApp::gxx()
{
    glUseProgram(glslKP.d2Gdx2);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[D2GDX2]);

    glUniform1i(glslKP.d2Gdx2TexLoc, INPUT_TEXTURE);
    glUniform1f(glslKP.d2Gdx2WLoc, scrWidth);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(glslKP.Gy);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[GXX]);

    glUniform1i(glslKP.GyTexLoc, D2GDX2);
    glUniform1f(glslKP.GyWLoc, scrHeight);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void WAIApp::gyy()
{
    glUseProgram(glslKP.d2Gdy2);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[D2GDY2]);

    glUniform1i(glslKP.d2Gdy2TexLoc, INPUT_TEXTURE);
    glUniform1f(glslKP.d2Gdy2WLoc, scrHeight);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(glslKP.Gx);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[GYY]);

    glUniform1i(glslKP.GxTexLoc, D2GDY2);
    glUniform1f(glslKP.GxWLoc, scrWidth);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void WAIApp::gxy()
{
    glUseProgram(glslKP.dGdx);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[DGDX]);

    glUniform1i(glslKP.dGdxTexLoc, INPUT_TEXTURE);
    glUniform1f(glslKP.dGdxWLoc, scrWidth);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(glslKP.dGdy);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[GXY]);

    glUniform1i(glslKP.GyTexLoc, DGDX);
    glUniform1f(glslKP.GyWLoc, scrHeight);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void WAIApp::detH()
{
    glUseProgram(glslKP.detH);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[DETH]);

    glUniform1i(glslKP.detHGxxLoc, GXX);
    glUniform1i(glslKP.detHGyyLoc, GYY);
    glUniform1i(glslKP.detHGxyLoc, GXY);

    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void WAIApp::nms(int id)
{
    glUseProgram(glslKP.nmsx);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.renderFBO[NMSX]);

    glUniform1i(glslKP.nmsxTexLoc, DETH);
    glUniform1f(glslKP.nmsxWLoc, scrWidth);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(glslKP.nmsy);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.framebuffers[id]);

    glUniform1i(glslKP.nmsyTexLoc, NMSX);
    glUniform1f(glslKP.nmsyWLoc, scrHeight);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    /*
    glUseProgram(glslKP.nmsz);
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.framebuffers[id]);

    glUniform1i(glslKP.nmszTexLoc, NMSY);
    glBindVertexArray(glslKP.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glslKP.vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    */

}

void WAIApp::gpu_kp()
{
    glDisable(GL_DEPTH_TEST);

    glslKP.ready = glslKP.curr;
    glslKP.curr = (glslKP.curr+1) % 2; //Set rendering F

    glActiveTexture(GL_TEXTURE0 + INPUT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, glslKP.grayTexture);

    for (int i = 0; i < 8; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, glslKP.renderTextures[i]);
    }

    gxx();
    gyy();
    gxy();
    detH();
    nms(glslKP.curr);

    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
}

void WAIApp::readResult()
{
    HighResTimer t;
    t.start();
    AVERAGE_TIMING_START("PBO");

    /* Copy pixel to curr pbo */
    glBindFramebuffer(GL_FRAMEBUFFER, glslKP.framebuffers[glslKP.ready]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, glslKP.pbo[glslKP.curr]);
    glReadPixels(0, 0, scrWidth, scrHeight, GL_RED, GL_UNSIGNED_BYTE, 0);
    /* Continue processing without stall */

    /* Read pixels from ready pbo */
    glBindBuffer(GL_PIXEL_PACK_BUFFER, glslKP.pbo[glslKP.ready]); //Read pixel from ready pbo
    unsigned char * data = (unsigned char*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, scrWidth * scrHeight, GL_MAP_READ_BIT);

    Utils::log("timing %f\n", (float)t.elapsedTimeInMicroSec());

    AVERAGE_TIMING_STOP("PBO");

    if (data)
    {
        testTexture->copyVideoImage(scrWidth,
                                    scrHeight,
                                    PF_red,
                                    PF_red,
                                    data,
                                    true,
                                    true);

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return;
    }

    Utils::log("Texture empty\n");
    Utils::log("Data not ready\n");
}

int WAIApp::load(int width, int height, float scr2fbX, float scr2fbY, int dpi, AppWAIDirectories* directories)
{
    defaultScrWidth  = width;
    defaultScrHeight = height;

    dirs = directories;
    videoDir       = dirs->writableDir + "videos/";
    calibDir       = dirs->writableDir + "calibrations/";
    mapDir         = dirs->writableDir + "maps/";
    vocDir         = dirs->writableDir + "voc/";
    experimentsDir = dirs->writableDir + "experiments/";

    wc              = new WAICalibration();
    waiScene        = new AppWAIScene();
    videoWriter     = new cv::VideoWriter();
    videoWriterInfo = new cv::VideoWriter();

    outputTexture = new unsigned char[width * height];
    for (int i = 0; i < scrWidth * scrHeight; i++)
        outputTexture[i] = 0;

    SLVstring empty;
    empty.push_back("WAI APP");
    slCreateAppAndScene(empty,
                        dirs->slDataRoot + "/shaders/",
                        dirs->slDataRoot + "/models/",
                        dirs->slDataRoot + "/images/textures/",
                        dirs->slDataRoot + "/images/fonts/",
                        dirs->writableDir,
                        "WAI Demo App",
                        (void*)WAIApp::onLoadWAISceneView);

    // This load the GUI configs that are locally stored
    uiPrefs.setDPI(dpi);
    uiPrefs.load();

    int svIndex = slCreateSceneView((int)(width * scr2fbX),
                                    (int)(height * scr2fbY),
                                    dpi,
                                    (SLSceneID)0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    (void*)buildGUI);

    loaded = true;
    SLApplication::devRot.isUsed(true);
    SLApplication::devLoc.isUsed(true);

    initTestProgram();

    return svIndex;
}

void WAIApp::close()
{
    uiPrefs.save();
    //ATTENTION: Other imgui stuff is automatically saved every 5 seconds
}

/*
videoFile: path to a video or empty if live video should be used
calibrationFile: path to a calibration or empty if calibration should be searched automatically
mapFile: path to a map or empty if no map should be used
*/
OrbSlamStartResult WAIApp::startOrbSlam(std::string videoFileName,
                                        std::string calibrationFileName,
                                        std::string mapFileName,
                                        std::string vocFileName,
                                        bool        saveVideoFrames)
{
    OrbSlamStartResult result = {};
    uiPrefs.showError         = false;

    bool useVideoFile             = !videoFileName.empty();
    bool detectCalibAutomatically = calibrationFileName.empty();
    bool useMapFile               = !mapFileName.empty();

    // reset stuff
    if (mode)
    {
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        delete mode;
        mode = nullptr;
    }

    // Check that files exist
    std::string videoFile = videoDir + videoFileName;
    if (useVideoFile && !Utils::fileExists(videoFile))
    {
        result.errorString = "Video file " + videoFile + " does not exist.";
        return result;
    }

    // determine correct calibration file
    if (detectCalibAutomatically)
    {
        std::string computerInfo;

        if (useVideoFile)
        {
            // get calibration file name from video file name
            std::vector<std::string> stringParts;
            Utils::splitString(videoFileName, '_', stringParts);

            if (stringParts.size() < 3)
            {
                result.errorString = "Could not extract computer infos from video filename.";
                return result;
            }

            computerInfo = stringParts[1];
        }
        else
        {
            computerInfo = SLApplication::getComputerInfos();
        }

        calibrationFileName = "camCalib_" + computerInfo + "_main.xml";
    }
    std::string calibrationFile = calibDir + calibrationFileName;

    if (!Utils::fileExists(calibrationFile))
    {
        result.errorString = "Calibration file " + calibrationFile + " does not exist.";
        return result;
    }

    std::string vocFile = vocDir + vocFileName;
    if (!vocFileName.empty() && !Utils::fileExists(vocFile))
    {
        result.errorString = "Vocabulary file does not exist: " + vocFile;
        return result;
    }

    std::string mapFile = mapDir + mapFileName;
    if (useMapFile && !Utils::fileExists(mapFile))
    {
        result.errorString = "Map file " + mapFile + " does not exist.";
        return result;
    }

    // 1. Initialize CVCapture with either video file or live video
    cv::Size2i videoFrameSize;
    if (useVideoFile)
    {
        CVCapture::instance()->videoType(VT_FILE);
        CVCapture::instance()->videoFilename = videoFile;
        CVCapture::instance()->videoLoops    = true;
        videoFrameSize                       = CVCapture::instance()->openFile();
    }
    else
    {
        CVCapture::instance()->videoType(VT_MAIN);
        CVCapture::instance()->open(0);

        videoFrameSize = cv::Size2i(defaultScrWidth, defaultScrHeight);
    }

    // 2. Load Calibration
    if (!wc->loadFromFile(calibrationFile))
    {
        result.errorString = "Error when loading calibration from file: " +
                             calibrationFile;
        return result;
    }

    float videoAspectRatio = (float)videoFrameSize.width / (float)videoFrameSize.height;
    float epsilon          = 0.01f;
    if (wc->aspectRatio() > videoAspectRatio + epsilon ||
        wc->aspectRatio() < videoAspectRatio - epsilon)
    {
        result.errorString = "Calibration aspect ratio does not fit video aspect ratio.\nCalib file: " +
                             calibrationFile + "\nVideo file: " +
                             (!videoFile.empty() ? videoFile : "Live Video");
        return result;
    }

    CVCapture::instance()->activeCalib->load(calibDir, calibrationFileName, 0, 0);

    // 3. Adjust FOV of camera node according to new calibration
    waiScene->cameraNode->fov(wc->calcCameraVerticalFOV());

    // 4. Create new mode ORBSlam
    mode = new WAI::ModeOrbSlam2(wc->cameraMat(),
                                 wc->distortion(),
                                 false,
                                 saveVideoFrames,
                                 false,
                                 false,
                                 vocFile);

    // 5. Load map data
    if (useMapFile)
    {
        mode->requestStateIdle();
        while (!mode->hasStateIdle())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        mode->reset();

        bool mapLoadingSuccess = WAIMapStorage::loadMap(mode->getMap(),
                                                        mode->getKfDB(),
                                                        waiScene->mapNode,
                                                        mapFile);

        if (!mapLoadingSuccess)
        {
            delete mode;
            mode = nullptr;

            result.errorString = "Could not load map from file " + mapFile;
            return result;
        }

        mode->resume();
        mode->setInitialized(true);
    }

    // 6. resize window
    scrWidth     = videoFrameSize.width;
    scrHeight    = videoFrameSize.height;
    scrWdivH     = (float)scrWidth / (float)scrHeight;
    resizeWindow = true;

    result.wasSuccessful = true;
    return result;
}

void WAIApp::setupGUI()
{
    aboutDial = new AppDemoGuiAbout("about", cpvrLogo, &uiPrefs.showAbout);
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosFrameworks("frameworks", &uiPrefs.showInfosFrameworks));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosMapNodeTransform("map node",
                                                                  waiScene->mapNode,
                                                                  &uiPrefs.showInfosMapNodeTransform));

    AppDemoGui::addInfoDialog(new AppDemoGuiInfosScene("scene", &uiPrefs.showInfosScene));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosSensors("sensors", &uiPrefs.showInfosSensors));
    AppDemoGui::addInfoDialog(new AppDemoGuiInfosTracking("tracking", uiPrefs));
    AppDemoGui::addInfoDialog(new AppDemoGuiSlamLoad("slam load", wc, &uiPrefs.showSlamLoad));

    AppDemoGui::addInfoDialog(new AppDemoGuiProperties("properties", &uiPrefs.showProperties));
    AppDemoGui::addInfoDialog(new AppDemoGuiSceneGraph("scene graph", &uiPrefs.showSceneGraph));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsDebugTiming("debug timing", &uiPrefs.showStatsDebugTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsTiming("timing", &uiPrefs.showStatsTiming));
    AppDemoGui::addInfoDialog(new AppDemoGuiStatsVideo("video", wc, &uiPrefs.showStatsVideo));
    AppDemoGui::addInfoDialog(new AppDemoGuiTrackedMapping("tracked mapping", &uiPrefs.showTrackedMapping));

    AppDemoGui::addInfoDialog(new AppDemoGuiTransform("transform", &uiPrefs.showTransform));
    AppDemoGui::addInfoDialog(new AppDemoGuiUIPrefs("prefs", &uiPrefs, &uiPrefs.showUIPrefs));

    AppDemoGui::addInfoDialog(new AppDemoGuiVideoStorage("video storage", videoWriter, videoWriterInfo, &gpsDataStream, &uiPrefs.showVideoStorage));
    AppDemoGui::addInfoDialog(new AppDemoGuiVideoControls("video load", &uiPrefs.showVideoControls));

    AppDemoGui::addInfoDialog(new AppDemoGuiMapStorage("Map storage", waiScene->mapNode, &uiPrefs.showMapStorage));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestOpen("Tests Settings",
                                                     wc,
                                                     waiScene->mapNode,
                                                     &uiPrefs.showTestSettings));

    AppDemoGui::addInfoDialog(new AppDemoGuiTestWrite("Test Writer",
                                                      wc,
                                                      waiScene->mapNode,
                                                      videoWriter,
                                                      videoWriterInfo,
                                                      &gpsDataStream,
                                                      &uiPrefs.showTestWriter));

    AppDemoGui::addInfoDialog(new AppDemoGuiSlamParam("Slam Param", &uiPrefs.showSlamParam));
    errorDial = new AppDemoGuiError("Error", &uiPrefs.showError);

    AppDemoGui::addInfoDialog(errorDial);

    //TODO: AppDemoGuiInfosDialog are never deleted. Why not use smart pointer when the reponsibility for an object is not clear?
}

void WAIApp::buildGUI(SLScene* s, SLSceneView* sv)
{
    if (uiPrefs.showAbout)
    {
        aboutDial->buildInfos(s, sv);
    }
    else
    {
        AppDemoGui::buildInfosDialogs(s, sv);
        AppDemoGuiMenu::build(&uiPrefs, s, sv);
    }
}

void WAIApp::refreshTexture(cv::Mat* image)
{
    if (image == nullptr)
        return;


    videoImage->copyVideoImage(image->cols, image->rows, CVCapture::instance()->format, image->data, image->isContinuous(), true);
}

//-----------------------------------------------------------------------------
void WAIApp::onLoadWAISceneView(SLScene* s, SLSceneView* sv, SLSceneID sid)
{
    s->init();
    waiScene->rebuild();
    //setup gui at last because ui elements depend on other instances
    setupGUI();

    // Set scene name and info string
    s->name("Track Keyframe based Features");
    s->info("Example for loading an existing pose graph with map points.");

    // Save no energy
    sv->doWaitOnIdle(false); //for constant video feed
    sv->camera(waiScene->cameraNode);

    videoImage = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    //waiScene->cameraNode->background().texture(videoImage);

    testTexture = new SLGLTexture("LiveVideoError.png", GL_LINEAR, GL_LINEAR);
    waiScene->cameraNode->background().texture(testTexture);

    //waiScene->cameraNode->fov(wc->calcCameraVerticalFOV());

    s->root3D(waiScene->rootNode);

    sv->onInitialize();
    sv->doWaitOnIdle(false);

    OrbSlamStartResult orbSlamStartResult = startOrbSlam();

    if (!orbSlamStartResult.wasSuccessful)
    {
        errorDial->setErrorMsg(orbSlamStartResult.errorString);
        uiPrefs.showError = true;
    }

    ////setup gui at last because ui elements depend on other instances
    //setupGUI();
}

//-----------------------------------------------------------------------------
bool WAIApp::update()
{
    AVERAGE_TIMING_START("WAIAppUpdate");
    //resetAllTexture();
    if (!mode)
        return false;

    if (!loaded)
        return false;


    glBindTexture(GL_TEXTURE_2D, glslKP.grayTexture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RED,
                 CVCapture::instance()->lastFrameGray.cols,
                 CVCapture::instance()->lastFrameGray.rows,
                 0,
                 GL_RED,
                 GL_UNSIGNED_BYTE,
                 CVCapture::instance()->lastFrameGray.data);
    glBindTexture(GL_TEXTURE_2D, 0);


    gpu_kp();
    readResult();



    bool iKnowWhereIAm = (mode->getTrackingState() == WAI::TrackingState_TrackingOK);
    while (videoCursorMoveIndex < 0)
    {
        CVCapture::instance()->moveCapturePosition(-2);
        CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex++;
    }

    while (videoCursorMoveIndex > 0)
    {
        CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
        iKnowWhereIAm = updateTracking();

        videoCursorMoveIndex--;
    }

    if (CVCapture::instance()->videoType() != VT_NONE)
    {
        if (CVCapture::instance()->videoType() != VT_FILE || !pauseVideo)
        {
            CVCapture::instance()->grabAndAdjustForSL(scrWdivH);
            iKnowWhereIAm = updateTracking();
        }
    }

    //update tracking infos visualization
    updateTrackingVisualization(iKnowWhereIAm);

    if (iKnowWhereIAm)
    {
        // TODO(dgj1): maybe make this API cleaner
        cv::Mat pose = cv::Mat(4, 4, CV_32F);
        if (!mode->getPose(&pose))
        {
            return false;
        }

        // update camera node position
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);

        Rwc = (pose.rowRange(0, 3).colRange(0, 3)).t();
        twc = -Rwc * pose.rowRange(0, 3).col(3);

        cv::Mat PoseInv = cv::Mat::eye(4, 4, CV_32F);

        Rwc.copyTo(PoseInv.colRange(0, 3).rowRange(0, 3));
        twc.copyTo(PoseInv.rowRange(0, 3).col(3));
        SLMat4f om;

        om.setMatrix(PoseInv.at<float>(0, 0),
                     -PoseInv.at<float>(0, 1),
                     -PoseInv.at<float>(0, 2),
                     PoseInv.at<float>(0, 3),
                     PoseInv.at<float>(1, 0),
                     -PoseInv.at<float>(1, 1),
                     -PoseInv.at<float>(1, 2),
                     PoseInv.at<float>(1, 3),
                     PoseInv.at<float>(2, 0),
                     -PoseInv.at<float>(2, 1),
                     -PoseInv.at<float>(2, 2),
                     PoseInv.at<float>(2, 3),
                     PoseInv.at<float>(3, 0),
                     -PoseInv.at<float>(3, 1),
                     -PoseInv.at<float>(3, 2),
                     PoseInv.at<float>(3, 3));

        waiScene->cameraNode->om(om);
    }

    AVERAGE_TIMING_STOP("WAIAppUpdate");

    return true;
}
//-----------------------------------------------------------------------------
bool WAIApp::updateTracking()
{
    bool iKnowWhereIAm = false;

    if (CVCapture::instance()->videoType() != VT_NONE && !CVCapture::instance()->lastFrame.empty())
    {
        if (videoWriter->isOpened())
        {
            videoWriter->write(CVCapture::instance()->lastFrame);
        }

        iKnowWhereIAm = mode->update(CVCapture::instance()->lastFrameGray,
                                     CVCapture::instance()->lastFrame);

        videoImage->copyVideoImage(CVCapture::instance()->lastFrame.cols,
                                   CVCapture::instance()->lastFrame.rows,
                                   CVCapture::instance()->format,
                                   CVCapture::instance()->lastFrame.data,
                                   CVCapture::instance()->lastFrame.isContinuous(),
                                   true);

        if (videoWriterInfo->isOpened())
        {
            videoWriterInfo->write(CVCapture::instance()->lastFrame);
        }

        if (gpsDataStream.is_open())
        {
            if (SLApplication::devLoc.isUsed())
            {
                SLVec3d v = SLApplication::devLoc.locLLA();
                gpsDataStream << SLApplication::devLoc.locAccuracyM();
                gpsDataStream << std::to_string(v.x) + " " + std::to_string(v.y) + " " + std::to_string(v.z);
                gpsDataStream << std::to_string(SLApplication::devRot.yawRAD());
                gpsDataStream << std::to_string(SLApplication::devRot.pitchRAD());
                gpsDataStream << std::to_string(SLApplication::devRot.rollRAD());
            }
        }
    }

    return iKnowWhereIAm;
}
//-----------------------------------------------------------------------------
void WAIApp::updateTrackingVisualization(const bool iKnowWhereIAm)
{
    //update keypoints visualization (2d image points):
    //TODO: 2d visualization is still done in mode... do we want to keep it there?
    mode->showKeyPoints(uiPrefs.showKeyPoints);
    mode->showKeyPointsMatched(uiPrefs.showKeyPointsMatched);

    //update map point visualization:
    //if we still want to visualize the point cloud
    if (uiPrefs.showMapPC)
    {
        //get new points and add them
        renderMapPoints("MapPoints",
                        mode->getMapPoints(),
                        waiScene->mapPC,
                        waiScene->mappointsMesh,
                        waiScene->redMat);
    }
    else if (waiScene->mappointsMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapPC->deleteMesh(waiScene->mappointsMesh);
    }

    //update visualization of local map points:
    //only update them with a valid pose from WAI
    if (uiPrefs.showLocalMapPC && iKnowWhereIAm)
    {
        renderMapPoints("LocalMapPoints",
                        mode->getLocalMapPoints(),
                        waiScene->mapLocalPC,
                        waiScene->mappointsLocalMesh,
                        waiScene->blueMat);
    }
    else if (waiScene->mappointsLocalMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapLocalPC->deleteMesh(waiScene->mappointsLocalMesh);
    }

    //update visualization of matched map points
    //only update them with a valid pose from WAI
    if (uiPrefs.showMatchesPC && iKnowWhereIAm)
    {
        renderMapPoints("MatchedMapPoints",
                        mode->getMatchedMapPoints(),
                        waiScene->mapMatchedPC,
                        waiScene->mappointsMatchedMesh,
                        waiScene->greenMat);
    }
    else if (waiScene->mappointsMatchedMesh)
    {
        //delete mesh if we do not want do visualize it anymore
        waiScene->mapMatchedPC->deleteMesh(waiScene->mappointsMatchedMesh);
    }

    //update keyframe visualization
    waiScene->keyFrameNode->deleteChildren();
    if (uiPrefs.showKeyFrames)
    {
        renderKeyframes();
    }

    //update pose graph visualization
    renderGraphs();
}

//-----------------------------------------------------------------------------
void WAIApp::renderMapPoints(std::string                      name,
                             const std::vector<WAIMapPoint*>& pts,
                             SLNode*&                         node,
                             SLPoints*&                       mesh,
                             SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    if (mesh)
        node->deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(points, normals, name, material);
        node->addMesh(mesh);
        node->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void WAIApp::renderKeyframes()
{
    std::vector<WAIKeyFrame*> keyframes = mode->getKeyFrames();

    // TODO(jan): delete keyframe textures
    for (WAIKeyFrame* kf : keyframes)
    {
        if (kf->isBad())
            continue;

        SLKeyframeCamera* cam = new SLKeyframeCamera("KeyFrame " + std::to_string(kf->mnId));
        //set background
        if (kf->getTexturePath().size())
        {
            // TODO(jan): textures are saved in a global textures vector (scene->textures)
            // and should be deleted from there. Otherwise we have a yuuuuge memory leak.
#if 0
        SLGLTexture* texture = new SLGLTexture(kf->getTexturePath());
        _kfTextures.push_back(texture);
        cam->background().texture(texture);
#endif
        }

        cv::Mat Twc = kf->getObjectMatrix();
        SLMat4f om;
        om.setMatrix(Twc.at<float>(0, 0),
                     -Twc.at<float>(0, 1),
                     -Twc.at<float>(0, 2),
                     Twc.at<float>(0, 3),
                     Twc.at<float>(1, 0),
                     -Twc.at<float>(1, 1),
                     -Twc.at<float>(1, 2),
                     Twc.at<float>(1, 3),
                     Twc.at<float>(2, 0),
                     -Twc.at<float>(2, 1),
                     -Twc.at<float>(2, 2),
                     Twc.at<float>(2, 3),
                     Twc.at<float>(3, 0),
                     -Twc.at<float>(3, 1),
                     -Twc.at<float>(3, 2),
                     Twc.at<float>(3, 3));

        cam->om(om);

        //calculate vertical field of view
        SLfloat fy     = (SLfloat)kf->fy;
        SLfloat cy     = (SLfloat)kf->cy;
        SLfloat fovDeg = 2 * (SLfloat)atan2(cy, fy) * Utils::RAD2DEG;
        cam->fov(fovDeg);
        cam->focalDist(0.11f);
        cam->clipNear(0.1f);
        cam->clipFar(1000.0f);

        waiScene->keyFrameNode->addChild(cam);
    }
}
//-----------------------------------------------------------------------------
void WAIApp::renderGraphs()
{
    std::vector<WAIKeyFrame*> kfs = mode->getKeyFrames();

    SLVVec3f covisGraphPts;
    SLVVec3f spanningTreePts;
    SLVVec3f loopEdgesPts;
    for (auto* kf : kfs)
    {
        cv::Mat Ow = kf->GetCameraCenter();

        //covisibility graph
        const std::vector<WAIKeyFrame*> vCovKFs = kf->GetBestCovisibilityKeyFrames(uiPrefs.minNumOfCovisibles);

        if (!vCovKFs.empty())
        {
            for (vector<WAIKeyFrame*>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
            {
                if ((*vit)->mnId < kf->mnId)
                    continue;
                cv::Mat Ow2 = (*vit)->GetCameraCenter();

                covisGraphPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
                covisGraphPts.push_back(SLVec3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2)));
            }
        }

        //spanning tree
        WAIKeyFrame* parent = kf->GetParent();
        if (parent)
        {
            cv::Mat Owp = parent->GetCameraCenter();
            spanningTreePts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            spanningTreePts.push_back(SLVec3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2)));
        }

        //loop edges
        std::set<WAIKeyFrame*> loopKFs = kf->GetLoopEdges();
        for (set<WAIKeyFrame*>::iterator sit = loopKFs.begin(), send = loopKFs.end(); sit != send; sit++)
        {
            if ((*sit)->mnId < kf->mnId)
                continue;
            cv::Mat Owl = (*sit)->GetCameraCenter();
            loopEdgesPts.push_back(SLVec3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2)));
            loopEdgesPts.push_back(SLVec3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2)));
        }
    }

    if (waiScene->covisibilityGraphMesh)
        waiScene->covisibilityGraph->deleteMesh(waiScene->covisibilityGraphMesh);

    if (covisGraphPts.size() && uiPrefs.showCovisibilityGraph)
    {
        waiScene->covisibilityGraphMesh = new SLPolyline(covisGraphPts, false, "CovisibilityGraph", waiScene->covisibilityGraphMat);
        waiScene->covisibilityGraph->addMesh(waiScene->covisibilityGraphMesh);
        waiScene->covisibilityGraph->updateAABBRec();
    }

    if (waiScene->spanningTreeMesh)
        waiScene->spanningTree->deleteMesh(waiScene->spanningTreeMesh);

    if (spanningTreePts.size() && uiPrefs.showSpanningTree)
    {
        waiScene->spanningTreeMesh = new SLPolyline(spanningTreePts, false, "SpanningTree", waiScene->spanningTreeMat);
        waiScene->spanningTree->addMesh(waiScene->spanningTreeMesh);
        waiScene->spanningTree->updateAABBRec();
    }

    if (waiScene->loopEdgesMesh)
        waiScene->loopEdges->deleteMesh(waiScene->loopEdgesMesh);

    if (loopEdgesPts.size() && uiPrefs.showLoopEdges)
    {
        waiScene->loopEdgesMesh = new SLPolyline(loopEdgesPts, false, "LoopEdges", waiScene->loopEdgesMat);
        waiScene->loopEdges->addMesh(waiScene->loopEdgesMesh);
        waiScene->loopEdges->updateAABBRec();
    }
}
