#include <AverageTiming.h>
#include <GLSLHessian.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <CVCalibration.h>
#include <BRIEFPattern.h>

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
#define PATTERN 12

static std::string textureOfstFct = "\n"
                                    "vec3 Ix(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(ofst, 0.0)).rgb;\n"
                                    "}\n"
                                    "\n"
                                    "vec3 Iy(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(0.0, ofst)).rgb;\n"
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
                                 "out vec3 pixel;\n"
                                 "in vec2 texcoords;\n"
                                 "uniform float w;\n"
                                 "uniform sampler2D tex;\n"
                                 "\n"
                                 "#include texOffsets\n"
                                 "#include kernelSize\n"
                                 "#include gaussianKernel\n"
                                 ""
                                 "void main()\n"
                                 "{\n"
                                 "    \n"
                                 "    vec3 response = vec3(0.0);\n"
                                 "    for (int i = 0; i < kSize; i++)\n"
                                 "    {\n"
                                 "        vec3 v = Ix((float(i) - kHalfSize) / w);\n"
                                 "        response.r += lowKernel[i] * v.r;\n"
                                 "        response.g += mediumKernel[i] * v.g;\n"
                                 "        response.b += highKernel[i] * v.b;\n"
                                 "    }\n"
                                 "    pixel = response;\n"
                                 "}\n";

static std::string vGaussianFs = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "#endif\n"
                                 "out vec3 pixel;\n"
                                 "in vec2 texcoords;\n"
                                 "uniform float w;\n"
                                 "uniform sampler2D tex;\n"
                                 "\n"
                                 "#include texOffsets\n"
                                 "#include kernelSize\n"
                                 "#include gaussianKernel\n"
                                 "\n"
                                 "void main()\n"
                                 "{\n"
                                 "\n"
                                 "    \n"
                                 "    vec3 response = vec3(0.0);\n"
                                 "    for (int i = 0; i < kSize; i++)\n"
                                 "    {\n"
                                 "        vec3 v = Iy((float(i) - kHalfSize) / w);\n"
                                 "        response.r += lowKernel[i] * v.r;\n"
                                 "        response.g += mediumKernel[i] * v.g;\n"
                                 "        response.b += highKernel[i] * v.b;\n"
                                 "    }\n"
                                 "    pixel = response;\n"
                                 "}\n";

static std::string hGaussianDxFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out vec3 pixel;\n"
                                   "in vec2 texcoords;\n"
                                   "uniform float w;\n"
                                   "uniform sampler2D tex;\n"
                                   "\n"
                                   "#include texOffsets\n"
                                   "#include kernelSize\n"
                                   "#include gaussianD1Kernel\n"
                                   "\n"
                                   "\n"
                                   "void main()\n"
                                   "{\n"
                                   "    \n"
                                   "    vec3 response = vec3(0.0);\n"
                                   "    for (int i = 0; i < kSize; i++)\n"
                                   "    {\n"
                                   "        vec3 v = Ix((float(i) - kHalfSize) / w);\n"
                                   "        response.r += lowKernel[i] * v.r;\n"
                                   "        response.g += mediumKernel[i] * v.r;\n"
                                   "        response.b += highKernel[i] * v.r;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";

static std::string vGaussianDyFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out vec3 pixel;\n"
                                   "in vec2 texcoords;\n"
                                   "uniform float w;\n"
                                   "uniform sampler2D tex;\n"
                                   "\n"
                                   "#include texOffsets\n"
                                   "#include kernelSize\n"
                                   "#include gaussianD1Kernel\n"
                                   "\n"
                                   "void main()\n"
                                   "{\n"
                                   "    \n"
                                   "    vec3 response = vec3(0.0);\n"
                                   "    for (int i = 0; i < kSize; i++)\n"
                                   "    {\n"
                                   "        vec3 v = Iy((float(i) - kHalfSize) / w);\n"
                                   "        response.r += lowKernel[i] * v.r;\n"
                                   "        response.g += mediumKernel[i] * v.g;\n"
                                   "        response.b += highKernel[i] * v.b;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";


static std::string hGaussianDx2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out vec3 pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "#include texOffsets\n"
                                    "#include kernelSize\n"
                                    "#include gaussianD2Kernel\n"
                                    "\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "    \n"
                                    "    vec3 response = vec3(0.0);\n"
                                    "    for (int i = 0; i < kSize; i++)\n"
                                    "    {\n"
                                    "        vec3 v = Ix((float(i) - kHalfSize) / w);\n"
                                    "        response.r += lowKernel[i] * v.r;\n"
                                    "        response.g += mediumKernel[i] * v.r;\n"
                                    "        response.b += highKernel[i] * v.r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string vGaussianDy2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out vec3 pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "#include texOffsets\n"
                                    "#include kernelSize\n"
                                    "#include gaussianD2Kernel\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "    \n"
                                    "    vec3 response = vec3(0.0);\n"
                                    "    for (int i = 0; i < kSize; i++)\n"
                                    "    {\n"
                                    "        vec3 v = Iy((float(i) - kHalfSize) / w);\n"
                                    "        response.r += lowKernel[i] * v.r;\n"
                                    "        response.g += mediumKernel[i] * v.r;\n"
                                    "        response.b += highKernel[i] * v.r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string detHFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec3 pixel;\n"
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
                            "    pixel = gxx*gyy - gxy*gxy;\n"
                            "}\n";

static std::string nmsxFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec3 pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tex;\n"
                            "uniform float w;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    vec3 o = texture(tex, texcoords).rgb;\n"
                            "    vec3 px = texture(tex, texcoords + vec2(1.0/w, 0.0f)).rgb;\n"
                            "    vec3 nx = texture(tex, texcoords - vec2(1.0/w, 0.0f)).rgb;\n"
                            "    pixel = o;\n"
                            "    if (o.r <= nx.r || o.r <= px.r)\n"
                            "    {\n"
                            "        pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= nx.g || o.g <= px.g)\n"
                            "    {\n"
                            "        pixel.g = 0.0;\n"
                            "    }\n"
                            "    if (o.b <= nx.b || o.b <= px.b)\n"
                            "    {\n"
                            "        pixel.b = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string nmsyFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec3 pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tex;\n"
                            "uniform float w;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    \n"
                            "    vec3 o = texture(tex, texcoords).rgb;\n"
                            "    vec3 py = texture(tex, texcoords + vec2(0.0f, 1.0/w)).rgb;\n"
                            "    vec3 ny = texture(tex, texcoords - vec2(0.0f, 1.0/w)).rgb;\n"
                            "    pixel = o;"
                            "    if (o.r <= ny.r || o.r <= py.r)\n"
                            "    {\n"
                            "        pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= ny.g || o.g <= py.g)\n"
                            "    {\n"
                            "        pixel.g = 0.0;\n"
                            "    }\n"
                            "    if (o.b <= ny.b || o.b <= py.g)\n"
                            "    {\n"
                            "        pixel.b = 0.0;\n"
                            "    }\n"
                            "}\n";


static std::string fast =  "bool fast(float t)\n"
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
                           "\n";

static std::string removeEdge = "#ifdef GL_ES\n"
                                "precision highp float;\n"
                                "#endif\n"
                                "out vec3 pixel;\n"
                                "in vec2 texcoords;\n"
                                "uniform float w;\n"
                                "uniform float h;\n"
                                "uniform sampler2D gray;\n"
                                "uniform sampler2D det;\n"
                                "uniform sampler2D tgxx;\n"
                                "uniform sampler2D tgyy;\n"
                                "\n"
                                "void main()\n"
                                "{\n"
                                "    \n"
                                "    vec3 nms_det = texture(det, texcoords).rgb;\n"
                                "    vec3 gxx = texture(tgxx, texcoords).rgb;\n"
                                "    vec3 gyy = texture(tgyy, texcoords).rgb;\n"
                                "    vec3 tr = gxx + gyy;\n"
                                "    vec3 r = tr*tr / nms_det;\n"
                                "    pixel = vec3(0.0);\n"
                                "    if (r.r < 5.0)\n"
                                "    {\n"
                                "        pixel.r = nms_det.r;\n"
                                "    }\n"
                                "    if (r.g < 5.0)\n"
                                "    {\n"
                                "        pixel.g = nms_det.g;\n"
                                "    }\n"
                                "    if (r.b < 5.0)\n"
                                "    {\n"
                                "        pixel.b = nms_det.b;\n"
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
                                 "layout (binding = 0, offset = 0) uniform atomic_uint lowCounter;\n"
                                 "layout (binding = 0, offset = 4) uniform atomic_uint mediumCounter;\n"
                                 "layout (binding = 0, offset = 8) uniform atomic_uint highCounter;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D lowImage;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D mediumImage;\n"
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
                                 "    vec3 p = texture(tex, texcoords).rgb;\n"
                                 "    if (p.r > $THRESHOLD)\n"
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(lowCounter));\n"
                                 "         if (i < $NB_KEYPOINTS_LOW)\n"
                                 "         {\n"
                                 "             imageStore(lowImage, ivec2(i, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (p.g > $THRESHOLD)\n"
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(mediumCounter));\n"
                                 "         if (i < $NB_KEYPOINTS_MEDIUM)\n"
                                 "         {\n"
                                 "             imageStore(mediumImage, ivec2(i, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (p.b > $THRESHOLD)\n"
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(highCounter));\n"
                                 "         if (i < $NB_KEYPOINTS_HIGH)\n"
                                 "         {\n"
                                 "             imageStore(highImage, ivec2(i, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "}\n"
                                 ;

/*
static std::string descriptors = ""
                                 "layout (rgba8i) uniform readonly iimage2D pattern;\n"
                                 "\n"
                                 "ivec4 brief()\n"
                                 "{\n"
                                 "    ivec4 desc = 0;\n"
                                 "    for (int i = 0; i < 32; i++)\n"
                                 "    {\n"
                                 "        v = imageLoad(pattern, ivec2(i, 0);\n"
                                 "        c1 = texture(tex, vec2(float(v.x)/w, float(v.y)/h));\n"
                                 "        c2 = texture(tex, vec2(float(v.z)/w, float(v.t)/h));\n"
                                 "        desc.r = desc.r | ((c1 > c2) << i);\n"
                                 "    }\n"
                                 "    for (int i = 0; i < 32; i++)\n"
                                 "    {\n"
                                 "        v = imageLoad(pattern, ivec2(i+32, 0);\n"
                                 "        c1 = texture(tex, vec2(float(v.x)/w, float(v.y)/h));\n"
                                 "        c2 = texture(tex, vec2(float(v.z)/w, float(v.t)/h));\n"
                                 "        desc.g = desc.g | ((c1 > c2) << i);\n"
                                 "    }\n"
                                 "    for (int i = 0; i < 32; i++)\n"
                                 "    {\n"
                                 "        v = imageLoad(pattern, ivec2(i+64, 0);\n"
                                 "        c1 = texture(tex, vec2(float(v.x)/w, float(v.y)/h));\n"
                                 "        c2 = texture(tex, vec2(float(v.z)/w, float(v.t)/h));\n"
                                 "        desc.b = desc.b | ((c1 > c2) << i);\n"
                                 "    }\n"
                                 "    for (int i = 0; i < 32; i++)\n"
                                 "    {\n"
                                 "        v = imageLoad(pattern, ivec2(i+96, 0);\n"
                                 "        c1 = texture(tex, vec2(float(v.x)/w, float(v.y)/h));\n"
                                 "        c2 = texture(tex, vec2(float(v.z)/w, float(v.t)/h));\n"
                                 "        desc.a = desc.a | ((c1 > c2) << i);\n"
                                 "    }\n"
                                 "    return desc;\n"
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
*/

GLuint GLSLHessian::buildShaderFromSource(string source, GLenum shaderType)
{
    // Compile Shader code
    GLuint shaderHandle = glCreateShader(shaderType);
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
    Utils::replaceString(completeSrc, "#include gaussianKernel", gaussianKernelStr);
    Utils::replaceString(completeSrc, "#include gaussianD1Kernel", gaussianD1KernelStr);
    Utils::replaceString(completeSrc, "#include gaussianD2Kernel", gaussianD2KernelStr);
    Utils::replaceString(completeSrc, "#include kernelSize", kernelSizeStr);
    Utils::replaceString(completeSrc, "#include fast", fast);
    Utils::replaceString(completeSrc, "$NB_KEYPOINTS_LOW", nbKeypointsLowStr);
    Utils::replaceString(completeSrc, "$NB_KEYPOINTS_MEDIUM", nbKeypointsMediumStr);
    Utils::replaceString(completeSrc, "$NB_KEYPOINTS_HIGH", nbKeypointsHighStr);
    Utils::replaceString(completeSrc, "$THRESHOLD", thresholdStr);

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
        Utils::log("AAAA %s\n", src);
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

    extractorTexLoc            = glGetUniformLocation(extractor, "tex");
    extractorPatternLoc        = glGetUniformLocation(extractor, "pattern");
    extractorOffsetLoc         = glGetUniformLocation(extractor, "ofst");
    extractorSizeLoc           = glGetUniformLocation(extractor, "s");
    extractorIdxLoc            = glGetUniformLocation(extractor, "idx");
    extractorWLoc              = glGetUniformLocation(extractor, "w");
    extractorHLoc              = glGetUniformLocation(extractor, "h");
    extractorLowCountersLoc    = glGetUniformLocation(extractor, "lowCounters");
    extractorMediumCountersLoc = glGetUniformLocation(extractor, "mediumCounters");
    extractorHighCountersLoc   = glGetUniformLocation(extractor, "highCounters");
    extractorLowImageLoc       = glGetUniformLocation(extractor, "lowImage");
    extractorMediumImageLoc    = glGetUniformLocation(extractor, "mediumImage");
    extractorHighImageLoc      = glGetUniformLocation(extractor, "highImage");
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void GLSLHessian::textureRGBA(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void GLSLHessian::textureRGBF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_HALF_FLOAT, nullptr);
}

void GLSLHessian::textureRF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_HALF_FLOAT, nullptr);
}

void GLSLHessian::textureR(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
}


void GLSLHessian::initTextureBuffers(int width, int height)
{
    glGenTextures(12, renderTextures);

    glBindTexture(GL_TEXTURE_2D, renderTextures[0]);
    setTextureParameters();
    textureR(width, height);

    int i = 1;
    for (; i < 12; i++)
    {
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
        setTextureParameters();
        textureRGBF(width, height);
    }
}

void GLSLHessian::clearCounterBuffer()
{
    int i[3] = {0};
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 12, i, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

void GLSLHessian::initKeypointBuffers()
{
    /* Buffers to store keypoints */
    glGenTextures(1, &highImages);
    glGenTextures(1, &mediumImages);
    glGenTextures(1, &lowImages);

    glGenFramebuffers(1, &highImagesFB);
    glGenFramebuffers(1, &mediumImagesFB);
    glGenFramebuffers(1, &lowImagesFB);

    glGenBuffers(2, highImagePBOs);
    glGenBuffers(2, mediumImagePBOs);
    glGenBuffers(2, lowImagePBOs);

    int i[3] = {0};
    glGenBuffers(1, &atomicCounter);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 12, i, GL_DYNAMIC_DRAW);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glClearColor(0, 0, 0, 0);

    glBindTexture(GL_TEXTURE_2D, highImages);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypointsHigh, 64);
    glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, highImages, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, mediumImages);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypointsMedium, 64);
    glBindFramebuffer(GL_FRAMEBUFFER, mediumImagesFB);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mediumImages, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, lowImages);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypointsLow, 64);
    glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lowImages, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypointsHigh * 64 * 4 * 4, 0, GL_DYNAMIC_READ);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, mediumImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypointsMedium * 64 * 4 * 4, 0, GL_DYNAMIC_READ);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypointsLow * 64 * 4 * 4, 0, GL_DYNAMIC_READ);
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

/*
void GLSLHessian::initPattern()
{
    glGenTextures(1, &patternTexture);
    glBindTexture(GL_TEXTURE_2D, patternTexture);
    setTextureParameters();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, bit_pattern_31);
}
*/

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
    glUniform1i(extractorLowImageLoc, 0);
    glUniform1i(extractorMediumImageLoc, 1);
    glUniform1i(extractorHighImageLoc, 2);

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    float offsetx = 15.0 / (float)m_w;
    float offsety = 15.0 / (float)m_h;
    float sizex = ((float)m_w - 30.0) / (8.0 * float(m_w));
    float sizey = ((float)m_h - 30.0) / (8.0 * float(m_h));

    glUniform2f(extractorSizeLoc, sizex, sizey);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            clearCounterBuffer();
            glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, atomicCounter);
            glUniform1i(extractorIdxLoc, i * 8 + j);
            glUniform2f(extractorOffsetLoc, offsetx + j * sizex, offsety + i * sizey);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        }
    }
}

/* These function are scaled such that the value after all filters are about between 0-1 */
string GLSLHessian::gaussian(int size, int halfSize, float sigma)
{
    float v = 2.0 * (1.0 / sigma) * exp(-(halfSize*halfSize) / (2.0 * sigma*sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - halfSize);
        float v = 2.0 * (1.0 / sigma) * exp(-(x*x) / (2.0 * sigma*sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

string GLSLHessian::gaussianD1(int size, int half_size, float sigma)
{
    float v = -2.0 * (half_size / (sigma*sigma*sigma)) * exp(-(half_size*half_size) / (2.0 * sigma*sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - half_size);
        float v = -2.0 * (x / (sigma*sigma*sigma)) * exp(-(x*x) / (2.0 * sigma*sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

string GLSLHessian::gaussianD2(int size, int half_size, float sigma)
{
    float v = 2.0 * (half_size*half_size - sigma*sigma) / (sigma*sigma*sigma*sigma*sigma) * exp(-(half_size*half_size) / (2.0 * sigma*sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - half_size);
        float v = 2.0 * (x*x - sigma*sigma) / (sigma*sigma*sigma*sigma*sigma) * exp(-(x*x) / (2.0 * sigma*sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

void GLSLHessian::init(int w, int h, int nbKeypointsLow, int nbKeypointsMedium, int nbKeypointsHigh, float thrs, float lowSigma, float mediumSigma, float highSigma)
{
    m_w = w;
    m_h = h;
    curr = 1;
    ready = 0;
    mNbKeypointsLow = nbKeypointsLow;
    mNbKeypointsMedium = nbKeypointsMedium;
    mNbKeypointsHigh = nbKeypointsHigh;
    nbKeypointsLowStr = std::to_string(nbKeypointsLow);
    nbKeypointsMediumStr = std::to_string(nbKeypointsMedium);
    nbKeypointsHighStr = std::to_string(nbKeypointsHigh);
    thresholdStr = std::to_string(thrs);

    //At a radius of 3 sigma from the center, we keep ~97% of the gaussian fct.
    // | 0x1 to ensure this is a odd number (not divisible per 2)
    int size = ((int)floor(lowSigma * 6.0)) | 0x1;
    int halfSize = size >> 1;
    string sz_s = to_string(size);

    std::string gaussianLowKernelStr   = "const float lowKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussian(size, halfSize, lowSigma) + ");\n";
    std::string gaussianD1LowKernelStr = "const float lowKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD1(size, halfSize, lowSigma) + ");\n";
    std::string gaussianD2LowKernelStr = "const float lowKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD2(size, halfSize, lowSigma) + ");\n";

    std::string gaussianMediumKernelStr   = "const float mediumKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussian(size, halfSize, mediumSigma) + ");\n";
    std::string gaussianD1MediumKernelStr = "const float mediumKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD1(size, halfSize, mediumSigma) + ");\n";
    std::string gaussianD2MediumKernelStr = "const float mediumKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD2(size, halfSize, mediumSigma) + ");\n";

    std::string gaussianHighKernelStr   = "const float highKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussian(size, halfSize, highSigma) + ");\n";
    std::string gaussianD1HighKernelStr = "const float highKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD1(size, halfSize, highSigma) + ");\n";
    std::string gaussianD2HighKernelStr = "const float highKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD2(size, halfSize, highSigma) + ");\n";

    gaussianKernelStr = gaussianLowKernelStr + gaussianMediumKernelStr + gaussianHighKernelStr;
    gaussianD1KernelStr = gaussianD1LowKernelStr + gaussianD1MediumKernelStr + gaussianD1HighKernelStr;
    gaussianD2KernelStr = gaussianD2LowKernelStr  + gaussianD2MediumKernelStr + gaussianD2HighKernelStr;

    kernelSizeStr       = "const float kHalfSize = " + to_string((float)halfSize) + ";\nconst int kSize = " + sz_s + ";\n";

    initShaders();
    initVBO();
    initTextureBuffers(w, h);
    initKeypointBuffers();
    initFBO();

}

GLSLHessian::GLSLHessian() { }

GLSLHessian::GLSLHessian(int w, int h, int nbKeypointsLow, int nbKeypointsMedium, int nbKeypointsHigh, float thrs, float lowSigma, float mediumSigma, float highSigma)
{
    init(w, h, nbKeypointsLow, nbKeypointsMedium, nbKeypointsHigh, thrs, lowSigma, mediumSigma, highSigma);
}

GLSLHessian::~GLSLHessian()
{
    if (externalTexture)
    {
        glDeleteTextures(11, renderTextures+1);
    }
    else
    {
        glDeleteTextures(12, renderTextures);
    }

    glDeleteFramebuffers(12, renderFBO);

    glDeleteBuffers(1, &atomicCounter);
    glDeleteFramebuffers(1, &lowImagesFB);
    glDeleteFramebuffers(1, &mediumImagesFB);
    glDeleteFramebuffers(1, &highImagesFB);

    glDeleteTextures(1, &lowImages);
    glDeleteTextures(1, &mediumImages);
    glDeleteTextures(1, &highImages);

    glDeleteBuffers(2, lowImagePBOs);
    glDeleteBuffers(2, mediumImagePBOs);
    glDeleteBuffers(2, highImagePBOs);

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

void GLSLHessian::setInputTexture(SLGLTexture &tex)
{
    if (!externalTexture)
    {
        Utils::log("Error", "externalTexture is not set\n");
        return;
    }
    renderTextures[0] = (GLuint)tex.texID();
}

void GLSLHessian::setInputTexture(cv::Mat &image)
{
    glBindTexture(GL_TEXTURE_2D, renderTextures[0]);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_R8,
                 image.cols,
                 image.rows,
                 0,
                 GL_RED,
                 GL_UNSIGNED_BYTE,
                 image.data);
    glBindTexture(GL_TEXTURE_2D, 0);
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

    glActiveTexture(GL_TEXTURE12);
    glBindTexture(GL_TEXTURE_2D, patternTexture);

    glClearColor(0, 0, 0, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, mediumImagesFB);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindImageTexture(0, lowImages, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(1, mediumImages, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(2, highImages, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);

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
    glBindFramebuffer(GL_FRAMEBUFFER, lowImagesFB);
    glFlush();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypointsLow, 64, GL_RGBA_INTEGER, GL_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, mediumImagesFB);
    glFlush();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mediumImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypointsMedium, 64, GL_RGBA_INTEGER, GL_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, highImagesFB);
    glFlush();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypointsHigh, 64, GL_RGBA_INTEGER, GL_INT, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[ready]);
    unsigned int * lData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypointsLow * 64 * 4 * 4, GL_MAP_READ_BIT);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, mediumImagePBOs[ready]);
    unsigned int * mData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypointsMedium * 64 * 4 * 4, GL_MAP_READ_BIT);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, highImagePBOs[ready]);
    unsigned int * hData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypointsHigh * 64 * 4 * 4, GL_MAP_READ_BIT);

    if (hData && mData && lData)
    {
        for (int i = 0; i < 64; i++)
        {
            int n = 0;
            int j = 0;
            for (j = 0; j < mNbKeypointsLow; j++)
            {
                int idx = (i * mNbKeypointsLow + j) * 4;
                int x   = lData[idx];
                int y   = lData[idx + 1];
                if (x == 0)
                {
                    break;
                }
                if (x < 15 || x > m_w - 15)
                {
                    Utils::log("AAAA Error reading the low thres texture\n");
                    break;
                }

                kps.push_back(cv::KeyPoint(cv::Point2f(x, y), 1));
            }
            n = j;
            if (n < 8)
            {
                for (int j = 0; j < mNbKeypointsMedium; j++)
                {
                    int idx = (i * mNbKeypointsMedium + j) * 4;
                    int x = mData[idx];
                    int y = mData[idx+1];
                    if (x == 0)
                        break;

                    if (x < 15 || x > m_w - 15)
                    {
                        Utils::log("AAAA Error reading low thres texture\n");
                        break;
                    }
                    kps.push_back(cv::KeyPoint(cv::Point2f(x, y), 1));
                }
            }
            n += j;
            if (n < 8)
            {
                for (int j = 0; j < mNbKeypointsHigh; j++)
                {
                    int idx = (i * mNbKeypointsHigh + j) * 4;
                    int x = hData[idx];
                    int y = hData[idx+1];
                    if (x == 0)
                        break;

                    if (x < 15 || x > m_w - 15)
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

    glBindBuffer(GL_PIXEL_PACK_BUFFER, mediumImagePBOs[ready]);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, lowImagePBOs[ready]);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

