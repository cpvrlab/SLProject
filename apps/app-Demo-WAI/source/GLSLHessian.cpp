#include <AverageTiming.h>
#include <GLSLHessian.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <CVCalibration.h>
//#include <BRIEFpattern.h>

#define INPUT 0
#define D2GDX2 1
#define D2GDY2 2
#define DGDX 3
#define GXX 4
#define GYY 5
#define GXY 6
#define DETH 7
#define NMSX 8
#define NMSY 9
#define REMOVEEDGE 10
#define EXTRACTOR 11

static std::string textureOfstFct = "\n"
                                    "float Ix(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                                    "}\n"
                                    "\n"
                                    "float Iy(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                                    "}\n"
                                    "\n";

//#version 320 es
static std::string screenQuadVs = "layout (location = 0) in vec3 vcoords;\n"
                                  "out vec2 texcoords;\n"
                                  "\n"
                                  "void main()\n"
                                  "{\n"
                                  "    texcoords = 0.5 * (vcoords.xy + vec2(1.0));\n"
                                  "    gl_Position = vec4(vcoords, 1.0);\n"
                                  "}\n";


static std::string hGaussianFs = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "#endif\n"
                                 "out float pixel;\n"
                                 "in vec2 texcoords;\n"
                                 "uniform float w;\n"
                                 "uniform sampler2D tex;\n"
                                 "\n"
                                 "#include texOffsets\n"
                                 "\n"
                                 "const float kernel[15] = float[15]("
                                 "0.0031742033144480037, 0.008980510024247402, 0.02165110898093487, 0.04448075733770272, 0.07787123866346017, 0.11617023707406768, 0.1476813151730447, 0.15998125886418896, 0.14768131517304472, 0.11617023707406769, 0.07787123866346018, 0.04448075733770272, 0.02165110898093487, 0.008980510024247402, 0.0031742033144480037);\n"
                                 "void main()\n"
                                 "{\n"
                                 "    \n"
                                 "    float response = 0.0;\n"
                                 "    for (int i = 0; i < 15; i++)\n"
                                 "    {\n"
                                 "        float v = Ix((float(i) - 7.0) / w);\n"
                                 "        response += kernel[i] * v;\n"
                                 "    }\n"

                                 "    pixel = response;\n"
                                 "}\n";

static std::string vGaussianFs = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "#endif\n"
                                 "out float pixel;\n"
                                 "in vec2 texcoords;\n"
                                 "uniform float w;\n"
                                 "uniform sampler2D tex;\n"
                                 "\n"
                                 "#include texOffsets\n"
                                 "\n"
                                 "const float kernel[15] = float[15]("
                                 "0.0031742033144480037, 0.008980510024247402, 0.02165110898093487, 0.04448075733770272, 0.07787123866346017, 0.11617023707406768, 0.1476813151730447, 0.15998125886418896, 0.14768131517304472, 0.11617023707406769, 0.07787123866346018, 0.04448075733770272, 0.02165110898093487, 0.008980510024247402, 0.0031742033144480037);\n"
                                 "\n"
                                 "void main()\n"
                                 "{\n"
                                 "\n"
                                 "    \n"
                                 "    float response = 0.0;\n"
                                 "    for (int i = 0; i < 15; i++)\n"
                                 "    {\n"
                                 "        float v = Iy((float(i) - 7.0) / w);\n"
                                 "        response += kernel[i] * v;\n"
                                 "    }\n"
                                 "    pixel = response;\n"
                                 "}\n";

static std::string hGaussianDxFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out float pixel;\n"
                                   "in vec2 texcoords;\n"
                                   "uniform float w;\n"
                                   "uniform sampler2D tex;\n"
                                   "\n"
                                   "#include texOffsets\n"
                                   "\n"
                                   "const float kernel[15] = float[15]("
                                   "0.00888776928045441, 0.021553224058193758, 0.04330221796186974, 0.07116921174032437, 0.09344548639615223, 0.09293618965925417, 0.0590725260692179, 0.0, -0.059072526069217875, -0.09293618965925415, -0.09344548639615223, -0.07116921174032437, -0.04330221796186974, -0.021553224058193758, -0.00888776928045441);\n"
                                   "\n"
                                   "void main()\n"
                                   "{\n"
                                   "    \n"
                                   "    float response = 0.0;\n"
                                   "    for (int i = 0; i < 15; i++)\n"
                                   "    {\n"
                                   "        float v = Ix((float(i) - 7.0) / w);\n"
                                   "        response += kernel[i] * v;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";

static std::string vGaussianDyFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out float pixel;\n"
                                   "in vec2 texcoords;\n"
                                   "uniform float w;\n"
                                   "uniform sampler2D tex;\n"
                                   "\n"
                                   "#include texOffsets\n"
                                   "\n"
                                   "const float kernel[15] = float[15]("
                                   "0.00888776928045441, 0.021553224058193758, 0.04330221796186974, 0.07116921174032437, 0.09344548639615223, 0.09293618965925417, 0.0590725260692179, 0.0, -0.059072526069217875, -0.09293618965925415, -0.09344548639615223, -0.07116921174032437, -0.04330221796186974, -0.021553224058193758, -0.00888776928045441);\n"
                                   "\n"
                                   "void main()\n"
                                   "{\n"
                                   "    \n"
                                   "    float response = 0.0;\n"
                                   "    for (int i = 0; i < 15; i++)\n"
                                   "    {\n"
                                   "        float v = Iy((float(i) - 7.0) / w);\n"
                                   "        response += kernel[i] * v;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";


static std::string hGaussianDx2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out float pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "#include texOffsets\n"
                                    "\n"
                                    "const float kernel[15] = float[15]("
                                    "0.021711550670824344, 0.04274722771541763, 0.0649533269428046, 0.06938998144681627, 0.034263345011922505, -0.04182128534666434, -0.12405230474535753, -0.15998125886418896, -0.12405230474535757, -0.041821285346664384, 0.03426334501192248, 0.06938998144681627, 0.0649533269428046, 0.04274722771541763, 0.021711550670824344);\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "    \n"
                                    "    float response = 0.0;\n"
                                    "    for (int i = 0; i < 15; i++)\n"
                                    "    {\n"
                                    "        float v = Ix((float(i) - 7.0) / w);\n"
                                    "        response += kernel[i] * v;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string vGaussianDy2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out float pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "#include texOffsets\n"
                                    "\n"
                                    "const float kernel[15] = float[15]("
                                    "0.021711550670824344, 0.04274722771541763, 0.0649533269428046, 0.06938998144681627, 0.034263345011922505, -0.04182128534666434, -0.12405230474535753, -0.15998125886418896, -0.12405230474535757, -0.041821285346664384, 0.03426334501192248, 0.06938998144681627, 0.0649533269428046, 0.04274722771541763, 0.021711550670824344);\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "    \n"
                                    "    float response = 0.0;\n"
                                    "    for (int i = 0; i < 15; i++)\n"
                                    "    {\n"
                                    "        float v = Iy((float(i) - 7.0) / w);\n"
                                    "        response += kernel[i] * v;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string detHFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out float pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tgxx;\n"
                            "uniform sampler2D tgyy;\n"
                            "uniform sampler2D tgxy;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    \n"
                            "    float gxx = texture(tgxx, texcoords).r;\n"
                            "    float gyy = texture(tgyy, texcoords).r;\n"
                            "    float gxy = texture(tgxy, texcoords).r;\n"
                            "    float det = gxx * gyy - gxy * gxy;\n"
                            "    pixel = det;\n"
                            "}\n";

static std::string nmsxFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
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
                            "    pixel = o;\n"
                            "    if (o <= nx || o <= px)\n"
                            "    {\n"
                            "        pixel = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string nmsyFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
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
                            "    pixel = o;"
                            "    if (o <= ny || o <= py)\n"
                            "    {\n"
                            "        pixel = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string removeEdge = "#ifdef GL_ES\n"
                                "precision highp float;\n"
                                "#endif\n"
                                "out float pixel;\n"
                                "in vec2 texcoords;\n"
                                "uniform float w;\n"
                                "uniform float h;\n"
                                "uniform sampler2D gray;\n"
                                "uniform sampler2D det;\n"
                                "uniform sampler2D tgxx;\n"
                                "uniform sampler2D tgyy;\n"
                                "uniform sampler2D tgxy;\n"
                                "\n"
                                "\n"
                                "bool fast(float t)\n"
                                "{\n"
                                "    float p[16];\n"
                                "    float o = texture(gray, texcoords).r;\n"
                                "    p[0 ] = texture(gray, texcoords + vec2( 0.0 / w, 3.0 / h)).r - o;\n"
                                "    p[4 ] = texture(gray, texcoords + vec2( 3.0 / w, 0.0 / h)).r - o;\n"
                                "    p[8 ] = texture(gray, texcoords + vec2( 0.0 / w,-3.0 / h)).r - o;\n"
                                "    p[12] = texture(gray, texcoords + vec2(-3.0 / w, 0.0 / h)).r - o;\n"
                                "\n"
                                "    int n;\n"
                                "    if (p[0] > t || p[1] > t)\n"
                                "    {\n"
                                "        n = 0;\n"
                                "        if (p[0] > t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[4] > t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[8] > t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[12] > t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "    }\n"
                                "\n"
                                "    if (n < 3)\n"
                                "        return false;\n"
                                "\n"
                                "    if (p[0] < -t || p[1] < -t)\n"
                                "    {\n"
                                "        n = 0;\n"
                                "        if (p[0] < -t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[4] < -t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[8] < -t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "        if (p[12] < -t)\n"
                                "        {\n"
                                "            n++;\n"
                                "        }\n"
                                "    }\n"
                                "\n"
                                "    if (n < 3)\n"
                                "        return false;\n"
                                "\n"
                                "    p[1 ] = texture(gray, texcoords + vec2( 1.0 / w, 3.0 / h)).r - o;\n"
                                "    p[2 ] = texture(gray, texcoords + vec2( 2.0 / w, 2.0 / h)).r - o;\n"
                                "    p[3 ] = texture(gray, texcoords + vec2( 3.0 / w, 1.0 / h)).r - o;\n"
                                "    p[5 ] = texture(gray, texcoords + vec2( 3.0 / w,-1.0 / h)).r - o;\n"
                                "    p[6 ] = texture(gray, texcoords + vec2( 2.0 / w,-2.0 / h)).r - o;\n"
                                "    p[7 ] = texture(gray, texcoords + vec2( 1.0 / w,-3.0 / h)).r - o;\n"
                                "    p[9 ] = texture(gray, texcoords + vec2(-1.0 / w,-3.0 / h)).r - o;\n"
                                "    p[10] = texture(gray, texcoords + vec2(-2.0 / w,-2.0 / h)).r - o;\n"
                                "    p[11] = texture(gray, texcoords + vec2(-3.0 / w,-1.0 / h)).r - o;\n"
                                "    p[13] = texture(gray, texcoords + vec2(-3.0 / w, 1.0 / h)).r - o;\n"
                                "    p[14] = texture(gray, texcoords + vec2(-2.0 / w, 2.0 / h)).r - o;\n"
                                "    p[15] = texture(gray, texcoords + vec2(-1.0 / w, 3.0 / h)).r - o;\n"
                                "    \n"
                                "    n = 0;\n"
                                "    bool sup = true;\n"
                                "    for (int i = 0; i < 24; i++)\n\n"
                                "    {\n"
                                "         int idx = i % 16;\n"
                                "         if (sup)\n"
                                "         {\n"
                                "              if (p[idx] > t)\n"
                                "              {\n"
                                "                   n++;\n"
                                "              }\n"
                                "              else if(p[idx] < -t)\n"
                                "              {\n"
                                "                   n = 1;\n"
                                "                   sup = false;\n"
                                "              }\n"
                                "              else\n"
                                "              {\n"
                                "                   n = 0;\n"
                                "                   sup = false;\n"
                                "              }\n"
                                "         }\n"
                                "         else\n"
                                "         {\n"
                                "              if (p[idx] < -t)\n"
                                "              {\n"
                                "                   n++;\n"
                                "              }\n"
                                "              else if(p[idx] > t)\n"
                                "              {\n"
                                "                   n = 1;\n"
                                "                   sup = true;\n"
                                "              }\n"
                                "              else\n"
                                "              {\n"
                                "                   n = 0;\n"
                                "                   sup = true;\n"
                                "              }\n"
                                "         }\n"
                                "         if (n >= 8)\n"
                                "         {\n"
                                "               return true;\n"
                                "         }\n"
                                "    }\n"
                                "    return false;\n"
                                "    \n"
                                "}\n"
                                "\n"
                                "\n"
                                "void main()\n"
                                "{\n"
                                "    \n"
                                "    float nms_det = texture(det, texcoords).r;\n"
                                "    float gxx = texture(tgxx, texcoords).r;\n"
                                "    float gyy = texture(tgyy, texcoords).r;\n"
                                "    float gxy = texture(tgxy, texcoords).r;\n"
                                "    float det = gxx * gyy - gxy * gxy;\n"
                                "    float tr = gxx + gyy;\n"
                                "    float r = tr * tr / det;\n"
                                "    pixel = 0.0;\n"
                                "    if (nms_det > 0.0)\n"
                                "    {\n"
                                "        if (r < 6.0)\n"
                                "        {\n"
                                "            pixel = det;\n"
                                "        }\n"
                                "    }\n"
                                "    else\n"
                                "    {\n"
                                "        if (fast(0.2))\n"
                                "            pixel = 1.0;\n"
                                "    }\n"
                                "}\n";

static std::string screenQuadOffsetVs = "layout (location = 0) in vec3 vcoords;\n"
                                       "out vec2 texcoords;\n"
                                       "uniform vec2 ofst;\n"
                                       "uniform vec2 s;\n"
                                       "\n"
                                       "void main()\n"
                                       "{\n"
                                       "    vec2 coords = 0.5 * (vcoords.xy + vec2(1.0));\n" //[0, 1]
                                       "    texcoords = ofst + coords * s;\n" // offset + [0, s]
                                       "    gl_Position = vec4((2.0 * texcoords) - vec2(1.0), 0.0, 1.0);\n"
                                       "}\n"
                                       ;

static std::string extractorFS = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "precision highp iimage2D;\n"
                                 "#endif\n"
                                 "layout (binding = 0, offset = 0) uniform atomic_uint highCounter;\n"
                                 "layout (binding = 0, offset = 4) uniform atomic_uint lowCounter;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D lowImage;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D highImage;\n"
                                 "uniform sampler2D tex;\n"
                                 "uniform float w;\n"
                                 "uniform float h;\n"
                                 "uniform int idx;\n"
                                 "in vec2 texcoords;\n"
                                 "\n"
                                 "void main()\n"
                                 "{\n"
                                 "    ivec4 pos = ivec4(int(w * texcoords.x), int(h * texcoords.y), 0, 0);\n"
                                 "\n"
                                 "    float r = texture(tex, texcoords).r;\n"
                                 "    if (r > $HIGH_THRESHOLD)\n"
                                 "    {\n"
                                 "         int ih = int(atomicCounterIncrement(highCounter));\n"
                                 "         int il = int(atomicCounterIncrement(lowCounter));\n"
                                 "         if (il < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(highImage, ivec2(ih, idx), pos);\n"
                                 "             imageStore(lowImage, ivec2(il, idx), pos);\n"
                                 "         }\n"
                                 "         if (ih < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(highImage, ivec2(ih, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (r > $LOW_THRESHOLD)\n"
                                 "    {\n"
                                 "         int il = int(atomicCounterIncrement(lowCounter));\n"
                                 "         if (il < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(lowImage, ivec2(il, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "}\n"
                                 ;

GLuint GLSLHessian::buildShaderFromSource(string source, GLenum shaderType)
{
    // Compile Shader code
    GLuint      shaderHandle = glCreateShader(shaderType);
    string version;
    SLGLState* state = SLGLState::instance();

    if (state->glIsES3())
    {
        version = "#version 320 es\n";
    }
    else
    {
        version = "#version 450\n";
    }

    string completeSrc = version + source;

    Utils::replaceString(completeSrc, "#include texOffsets", textureOfstFct);
    Utils::replaceString(completeSrc, "$NB_KEYPOINTS", nbKeypointsStr);
    Utils::replaceString(completeSrc, "$HIGH_THRESHOLD", highThresholdStr);
    Utils::replaceString(completeSrc, "$LOW_THRESHOLD", lowThresholdStr);

    const char* src         = completeSrc.c_str();

    glShaderSource(shaderHandle, 1, &src, nullptr);
    glCompileShader(shaderHandle);

    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compileSuccess);

    GLint logSize = 0;
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &logSize);

    GLchar * log = new GLchar[logSize];

    glGetShaderInfoLog(shaderHandle, logSize, nullptr, log);

    if (!compileSuccess)
    {
        Utils::log("AAAA Cannot compile shader %s\n", log);
        Utils::log("AAAA %s\n", completeSrc);
        exit(1);
    }
    return shaderHandle;
}

void GLSLHessian::initShaders()
{
    GLuint vscreenQuad       = buildShaderFromSource(screenQuadVs, GL_VERTEX_SHADER);
    GLuint vscreenQuadOffset = buildShaderFromSource(screenQuadOffsetVs, GL_VERTEX_SHADER);
    GLuint fd2Gdx2           = buildShaderFromSource(hGaussianDx2Fs, GL_FRAGMENT_SHADER);
    GLuint fd2Gdy2           = buildShaderFromSource(vGaussianDy2Fs, GL_FRAGMENT_SHADER);
    GLuint fdGdx             = buildShaderFromSource(hGaussianDxFs, GL_FRAGMENT_SHADER);
    GLuint fdGdy             = buildShaderFromSource(vGaussianDyFs, GL_FRAGMENT_SHADER);
    GLuint fGx               = buildShaderFromSource(hGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fGy               = buildShaderFromSource(vGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fdetH             = buildShaderFromSource(detHFs, GL_FRAGMENT_SHADER);
    GLuint fnmsx             = buildShaderFromSource(nmsxFs, GL_FRAGMENT_SHADER);
    GLuint fnmsy             = buildShaderFromSource(nmsyFs, GL_FRAGMENT_SHADER);
    GLuint fedge             = buildShaderFromSource(removeEdge, GL_FRAGMENT_SHADER);
    GLuint fextractor        = buildShaderFromSource(extractorFS, GL_FRAGMENT_SHADER);

    d2Gdx2    = glCreateProgram();
    d2Gdy2    = glCreateProgram();
    dGdx      = glCreateProgram();
    dGdy      = glCreateProgram();
    Gx        = glCreateProgram();
    Gy        = glCreateProgram();
    detH      = glCreateProgram();
    nmsx      = glCreateProgram();
    nmsy      = glCreateProgram();
    edge      = glCreateProgram();
    extractor = glCreateProgram();

    glAttachShader(d2Gdx2, vscreenQuad);
    glAttachShader(d2Gdx2, fd2Gdx2);
    glLinkProgram(d2Gdx2);

    glAttachShader(d2Gdy2, vscreenQuad);
    glAttachShader(d2Gdy2, fd2Gdy2);
    glLinkProgram(d2Gdy2);

    glAttachShader(dGdx, vscreenQuad);
    glAttachShader(dGdx, fdGdx);
    glLinkProgram(dGdx);

    glAttachShader(dGdy, vscreenQuad);
    glAttachShader(dGdy, fdGdy);
    glLinkProgram(dGdy);

    glAttachShader(Gx, vscreenQuad);
    glAttachShader(Gx, fGx);
    glLinkProgram(Gx);

    glAttachShader(Gy, vscreenQuad);
    glAttachShader(Gy, fGy);
    glLinkProgram(Gy);

    glAttachShader(detH, vscreenQuad);
    glAttachShader(detH, fdetH);
    glLinkProgram(detH);

    glAttachShader(nmsx, vscreenQuad);
    glAttachShader(nmsx, fnmsx);
    glLinkProgram(nmsx);

    glAttachShader(nmsy, vscreenQuad);
    glAttachShader(nmsy, fnmsy);
    glLinkProgram(nmsy);

    glAttachShader(edge, vscreenQuad);
    glAttachShader(edge, fedge);
    glLinkProgram(edge);

    glAttachShader(extractor, vscreenQuadOffset);
    glAttachShader(extractor, fextractor);
    glLinkProgram(extractor);

    glDeleteShader(vscreenQuad);
    glDeleteShader(fd2Gdx2);
    glDeleteShader(fd2Gdy2);
    glDeleteShader(fdGdx);
    glDeleteShader(fdGdy);
    glDeleteShader(fGx);
    glDeleteShader(fGy);
    glDeleteShader(fdetH);
    glDeleteShader(fnmsx);
    glDeleteShader(fnmsy);
    glDeleteShader(fedge);
    glDeleteShader(fextractor);

    d2Gdx2TexLoc = glGetUniformLocation(d2Gdx2, "tex");
    d2Gdx2WLoc   = glGetUniformLocation(d2Gdx2, "w");
    d2Gdy2TexLoc = glGetUniformLocation(d2Gdy2, "tex");
    d2Gdy2WLoc   = glGetUniformLocation(d2Gdy2, "w");
    dGdxTexLoc   = glGetUniformLocation(dGdx, "tex");
    dGdxWLoc     = glGetUniformLocation(dGdx, "w");
    dGdyTexLoc   = glGetUniformLocation(dGdy, "tex");
    dGdyWLoc     = glGetUniformLocation(dGdy, "w");
    GxTexLoc     = glGetUniformLocation(Gx, "tex");
    GxWLoc       = glGetUniformLocation(Gx, "w");
    GyTexLoc     = glGetUniformLocation(Gy, "tex");
    GyWLoc       = glGetUniformLocation(Gy, "w");
    detHGxxLoc   = glGetUniformLocation(detH, "tgxx");
    detHGyyLoc   = glGetUniformLocation(detH, "tgyy");
    detHGxyLoc   = glGetUniformLocation(detH, "tgxy");
    nmsxTexLoc   = glGetUniformLocation(nmsx, "tex");
    nmsxWLoc     = glGetUniformLocation(nmsx, "w");
    nmsyTexLoc   = glGetUniformLocation(nmsy, "tex");
    nmsyWLoc     = glGetUniformLocation(nmsy, "w");
    edgeWLoc     = glGetUniformLocation(edge, "w");
    edgeHLoc     = glGetUniformLocation(edge, "h");
    edgeTexLoc   = glGetUniformLocation(edge, "gray");
    edgeDetLoc   = glGetUniformLocation(edge, "det");
    edgeGxxLoc   = glGetUniformLocation(edge, "tgxx");
    edgeGyyLoc   = glGetUniformLocation(edge, "tgyy");
    edgeGxyLoc   = glGetUniformLocation(edge, "tgxy");

    extractorTexLoc          = glGetUniformLocation(extractor, "tex");
    extractorOffsetLoc       = glGetUniformLocation(extractor, "ofst");
    extractorSizeLoc         = glGetUniformLocation(extractor, "s");
    extractorIdxLoc          = glGetUniformLocation(extractor, "idx");
    extractorWLoc            = glGetUniformLocation(extractor, "w");
    extractorHLoc            = glGetUniformLocation(extractor, "h");
    extractorHighCountersLoc = glGetUniformLocation(extractor, "highCounters");
    extractorLowCountersLoc  = glGetUniformLocation(extractor, "lowCounters");
    extractorLowImageLoc     = glGetUniformLocation(extractor, "lowImage");
    extractorHighImageLoc    = glGetUniformLocation(extractor, "highImage");
}

void GLSLHessian::initVBO()
{
    float vertices[12] = {-1, -1,  0,
                           1, -1,  0,
                           1,  1,  0,
                          -1,  1,  0 };

    GLuint indices[6] = {0, 1, 2, 2, 3, 0};

    //Gen VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &vboi);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(float), indices, GL_STATIC_DRAW);

    // Gen VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void GLSLHessian::setTextureParameters()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
}

void GLSLHessian::textureRGBF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_HALF_FLOAT, nullptr);
}

void GLSLHessian::textureRF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_HALF_FLOAT, nullptr);
}

void GLSLHessian::textureRB(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
}

void GLSLHessian::initTextureBuffers(int width, int height)
{
    glGenTextures(12, renderTextures);

    glBindTexture(GL_TEXTURE_2D, renderTextures[0]);
    setTextureParameters();
    textureRB(width, height);

    int i = 1;

    for (; i < 12; i++)
    {
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
        setTextureParameters();
        textureRF(width, height);
    }
}

void GLSLHessian::clearCounterBuffer()
{
    int i[2] = {0};
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 8, i, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

void GLSLHessian::initKeypointBuffers()
{
    /* Buffers to store keypoints */
    glGenTextures(2, highImages);
    glGenTextures(2, lowImages);

    glGenFramebuffers(2, highImagesFB);
    glGenFramebuffers(2, lowImagesFB);

    glGenBuffers(2, highImagePBOs);
    glGenBuffers(2, lowImagePBOs);

    int i[2] = {0};
    glGenBuffers(1, &atomicCounter);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 8, i, GL_DYNAMIC_DRAW);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glClearColor(0, 0, 0, 0);

    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, highImages[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypoints, 16);
        glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, highImages[i], 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, lowImages[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypoints, 16);
        glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lowImages[i], 0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypoints * 16 * 4 * 4, 0, GL_DYNAMIC_READ);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypoints * 16 * 4 * 4, 0, GL_DYNAMIC_READ);
    }
}

void GLSLHessian::initFBO()
{
    glGenFramebuffers(12, renderFBO);

    for (int i = 0; i < 12; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTextures[i], 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLSLHessian::gxx(int w, int h)
{
    glUseProgram(d2Gdx2);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[D2GDX2]);

    glUniform1i(d2Gdx2TexLoc, INPUT);
    glUniform1f(d2Gdx2WLoc, (float)w);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(Gy);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[GXX]);

    glUniform1i(GyTexLoc, D2GDX2);
    glUniform1f(GyWLoc, (float)h);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void GLSLHessian::gyy(int w, int h)
{
    glUseProgram(d2Gdy2);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[D2GDY2]);

    glUniform1i(d2Gdy2TexLoc, INPUT);
    glUniform1f(d2Gdy2WLoc, (float)h);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(Gx);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[GYY]);

    glUniform1i(GxTexLoc, D2GDY2);
    glUniform1f(GxWLoc, (float)w);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void GLSLHessian::gxy(int w, int h)
{
    glUseProgram(dGdx);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[DGDX]);

    glUniform1i(dGdxTexLoc, INPUT);
    glUniform1f(dGdxWLoc, (float)w);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(dGdy);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[GXY]);

    glUniform1i(GyTexLoc, DGDX);
    glUniform1f(GyWLoc, (float)h);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void GLSLHessian::det(int w, int h)
{
    glUseProgram(detH);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[DETH]);

    glUniform1i(detHGxxLoc, GXX);
    glUniform1i(detHGyyLoc, GYY);
    glUniform1i(detHGxyLoc, GXY);

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void GLSLHessian::nms(int w, int h)
{
    glUseProgram(nmsx);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[NMSX]);
    glUniform1i(nmsxTexLoc, DETH);
    glUniform1f(nmsxWLoc, (float)w);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(nmsy);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[NMSY]);
    glUniform1i(nmsyTexLoc, NMSX);
    glUniform1f(nmsyWLoc, (float)h);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(edge);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[REMOVEEDGE]);

    glUniform1f(edgeWLoc, m_w);
    glUniform1f(edgeHLoc, m_h);
    glUniform1i(edgeTexLoc, INPUT);
    glUniform1i(edgeDetLoc, NMSY);
    glUniform1i(edgeGxxLoc, GXX);
    glUniform1i(edgeGyyLoc, GYY);
    glUniform1i(edgeGxyLoc, GXY);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void GLSLHessian::extract(int w, int h, int curr)
{
    glUseProgram(extractor);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[EXTRACTOR]);
    glUniform1f(extractorWLoc, (float)w);
    glUniform1f(extractorHLoc, (float)h);
    glUniform1i(extractorTexLoc, REMOVEEDGE);
    glBindImageTexture(0, highImages[curr], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glUniform1i(extractorHighImageLoc, 0);
    glBindImageTexture(1, lowImages[curr], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glUniform1i(extractorLowImageLoc, 1);

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    float offsetx = 15.0 / (float)m_w;
    float offsety = 15.0 / (float)m_h;
    float sizex = ((float)m_w - 30.0) / (4.0 * float(m_w));
    float sizey = ((float)m_h - 30.0) / (4.0 * float(m_h));

    glUniform2f(extractorSizeLoc, sizex, sizey);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            clearCounterBuffer();
            glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounter);
            glUniform1i(extractorIdxLoc, i * 4 + j);
            glUniform2f(extractorOffsetLoc, offsetx + j * sizex, offsety + i * sizey);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        }
    }
}

void GLSLHessian::init(int w, int h, int nbKeypointsPerArea, float lowThrs, float highThrs)
{
    m_w = w;
    m_h = h;
    curr = 1;
    ready = 0;
    mNbKeypoints = nbKeypointsPerArea;
    nbKeypointsStr = std::to_string(nbKeypointsPerArea);
    lowThresholdStr = std::to_string(lowThrs);
    highThresholdStr = std::to_string(highThrs);
    initShaders();
    initVBO();
    initTextureBuffers(w, h);
    initKeypointBuffers();
    initFBO();
}

GLSLHessian::GLSLHessian() { }

GLSLHessian::GLSLHessian(int w, int h, int nbKeypointsPerArea, float lowThrs, float highThrs)
{
    init(w, h, nbKeypointsPerArea, lowThrs, highThrs);
}

GLSLHessian::~GLSLHessian()
{
    glDeleteTextures(12, renderTextures);
    glDeleteFramebuffers(12, renderFBO);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &vboi);

    glDeleteProgram(d2Gdx2);
    glDeleteProgram(d2Gdy2);
    glDeleteProgram(dGdx);
    glDeleteProgram(dGdy);
    glDeleteProgram(Gx);
    glDeleteProgram(Gy);
    glDeleteProgram(detH);
    glDeleteProgram(nmsx);
    glDeleteProgram(nmsy);
    glDeleteProgram(edge);
    glDeleteProgram(extractor);
}

void GLSLHessian::gpu_kp()
{
    glDisable(GL_DEPTH_TEST);

    ready = curr;
    curr = (curr+1) % 2; //Set rendering buffers

    SLVec4i wp = SLGLState::instance()->getViewport();
    SLGLState::instance()->viewport(0, 0, m_w, m_h);

    for (int i = 0; i < 12; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
    }

    glClearColor(0, 0, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB[curr]);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB[curr]);
    glClear(GL_COLOR_BUFFER_BIT);

    gxx(m_w, m_h);
    gyy(m_w, m_h);
    gxy(m_w, m_h);
    det(m_w, m_h);
    nms(m_w, m_h);
    extract(m_w, m_h, curr);

    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    SLGLState::instance()->viewport(wp.x, wp.y, wp.z, wp.w);
}

void GLSLHessian::readResult(std::vector<cv::KeyPoint> &kps)
{
    glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB[ready]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypoints, 16, GL_RGBA_INTEGER, GL_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB[ready]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypoints, 16, GL_RGBA_INTEGER, GL_INT, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[ready]);
    unsigned int * hData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypoints * 16 * 4 * 4, GL_MAP_READ_BIT);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[ready]);
    unsigned int * lData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypoints * 16 * 4 * 4, GL_MAP_READ_BIT);

    if (hData)
    {
        for (int i = 0; i < 16; i++)
        {
            if (hData[(mNbKeypoints * i + 5)*4] > 0) //if there is keypoint in the subimage
            {
                for (int j = 0; j < mNbKeypoints; j++)
                {
                    int idx = (i * mNbKeypoints + j) * 4;
                    int x = hData[idx];
                    int y = hData[idx+1];
                    if (x == 0)
                        break;

                    if (x < 15 || x > m_w-15)
                    {
                        Utils::log("AAAA Error reading the high thres texture\n");
                        break;
                    }

                    kps.push_back(cv::KeyPoint(cv::Point2f(x, y), 1));
                }
            }
            else if (lData)
            {
                for (int j = 0; j < mNbKeypoints; j++)
                {
                    int idx = (i * mNbKeypoints + j) * 4;
                    int x = lData[idx];
                    int y = lData[idx+1];
                    if (x == 0)
                        break;
                    if (x < 15 || x > m_w-15)
                    {
                        Utils::log("AAAA Error reading low thres texture\n");
                        break;
                    }

                    kps.push_back(cv::KeyPoint(cv::Point2f(x, y), 1));
                }
            }
        }
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[ready]);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

