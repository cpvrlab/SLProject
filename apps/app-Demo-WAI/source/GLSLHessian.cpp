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
                                    "vec2 Ix(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(ofst, 0.0)).rg;\n"
                                    "}\n"
                                    "\n"
                                    "vec2 Iy(float ofst)\n"
                                    "{\n"
                                    "    return texture(tex, texcoords + vec2(0.0, ofst)).rg;\n"
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
                                 "out vec2 pixel;\n"
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
                                 "    vec2 response = vec2(0.0);\n"
                                 "    for (int i = 0; i < kSize; i++)\n"
                                 "    {\n"
                                 "        vec2 v = Ix((float(i) - kHalfSize) / w);\n"
                                 "        response.r += bigSigmaKernel[i] * v.r;\n"
                                 "        response.g += smallSigmaKernel[i] * v.g;\n"
                                 "    }\n"
                                 "    pixel = response;\n"
                                 "}\n";

static std::string vGaussianFs = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "#endif\n"
                                 "out vec2 pixel;\n"
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
                                 "    vec2 response = vec2(0.0);\n"
                                 "    for (int i = 0; i < kSize; i++)\n"
                                 "    {\n"
                                 "        vec2 v = Iy((float(i) - kHalfSize) / w);\n"
                                 "        response.r += bigSigmaKernel[i] * v.r;\n"
                                 "        response.g += smallSigmaKernel[i] * v.g;\n"
                                 "    }\n"
                                 "    pixel = response;\n"
                                 "}\n";

static std::string hGaussianDxFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out vec2 pixel;\n"
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
                                   "    vec2 response = vec2(0.0);\n"
                                   "    for (int i = 0; i < kSize; i++)\n"
                                   "    {\n"
                                   "        vec2 v = Ix((float(i) - kHalfSize) / w);\n"
                                   "        response.r += bigSigmaKernel[i] * v.r;\n"
                                   "        response.g += smallSigmaKernel[i] * v.r;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";

static std::string vGaussianDyFs = "#ifdef GL_ES\n"
                                   "precision highp float;\n"
                                   "#endif\n"
                                   "out vec2 pixel;\n"
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
                                   "    vec2 response = vec2(0.0);\n"
                                   "    for (int i = 0; i < kSize; i++)\n"
                                   "    {\n"
                                   "        vec2 v = Iy((float(i) - kHalfSize) / w);\n"
                                   "        response.r += bigSigmaKernel[i] * v.r;\n"
                                   "        response.g += smallSigmaKernel[i] * v.g;\n"
                                   "    }\n"
                                   "    pixel = response;\n"
                                   "}\n";

static std::string hGaussianDx2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out vec2 pixel;\n"
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
                                    "    vec2 response = vec2(0.0);\n"
                                    "    for (int i = 0; i < kSize; i++)\n"
                                    "    {\n"
                                    "        vec2 v = Ix((float(i) - kHalfSize) / w);\n"
                                    "        response.r += bigSigmaKernel[i] * v.r;\n"
                                    "        response.g += smallSigmaKernel[i] * v.r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string vGaussianDy2Fs = "#ifdef GL_ES\n"
                                    "precision highp float;\n"
                                    "#endif\n"
                                    "out vec2 pixel;\n"
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
                                    "    vec2 response = vec2(0.0);\n"
                                    "    for (int i = 0; i < kSize; i++)\n"
                                    "    {\n"
                                    "        vec2 v = Iy((float(i) - kHalfSize) / w);\n"
                                    "        response.r += bigSigmaKernel[i] * v.r;\n"
                                    "        response.g += smallSigmaKernel[i] * v.r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string detHFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec2 pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tgxx;\n"
                            "uniform sampler2D tgyy;\n"
                            "uniform sampler2D tgxy;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    \n"
                            "    vec2 gxx = texture(tgxx, texcoords).rg;\n"
                            "    vec2 gyy = texture(tgyy, texcoords).rg;\n"
                            "    vec2 gxy = texture(tgxy, texcoords).rg;\n"
                            "    pixel = gxx*gyy - gxy*gxy;\n"
                            "}\n";

static std::string nmsxFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec2 pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tex;\n"
                            "uniform float w;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    vec2 o = texture(tex, texcoords).rg;\n"
                            "    vec2 px = texture(tex, texcoords + vec2(1.0/w, 0.0f)).rg;\n"
                            "    vec2 nx = texture(tex, texcoords - vec2(1.0/w, 0.0f)).rg;\n"
                            "    pixel = o;\n"
                            "    if (o.r <= nx.r || o.r <= px.r)\n"
                            "    {\n"
                            "        pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= nx.g || o.g <= px.g)\n"
                            "    {\n"
                            "        pixel.g = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string nmsyFs = "#ifdef GL_ES\n"
                            "precision highp float;\n"
                            "#endif\n"
                            "out vec2 pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tex;\n"
                            "uniform float w;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    \n"
                            "    vec2 o = texture(tex, texcoords).rg;\n"
                            "    vec2 py = texture(tex, texcoords + vec2(0.0f, 1.0/w)).rg;\n"
                            "    vec2 ny = texture(tex, texcoords - vec2(0.0f, 1.0/w)).rg;\n"
                            "    pixel = o;"
                            "    if (o.r <= ny.r || o.r <= py.r)\n"
                            "    {\n"
                            "        pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= ny.g || o.g <= py.g)\n"
                            "    {\n"
                            "        pixel.g = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string fast = "bool fast(float t)\n"
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
                                "out vec2 pixel;\n"
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
                                "    vec2 nms_det = texture(det, texcoords).rg;\n"
                                "    vec2 gxx = texture(tgxx, texcoords).rg;\n"
                                "    vec2 gyy = texture(tgyy, texcoords).rg;\n"
                                "    vec2 tr = gxx + gyy;\n"
                                "    vec2 r = tr*tr / nms_det;\n"
                                "    pixel = vec2(0.0);\n"
                                "    if (r.r < 5.0)\n"
                                "    {\n"
                                "        pixel.r = nms_det.r;\n"
                                "    }\n"
                                "    if (r.g < 5.0)\n"
                                "    {\n"
                                "        pixel.g = nms_det.g;\n"
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
                                        "    texcoords = ofst + coords * s;\n"                // offset + [0, s]
                                        "    gl_Position = vec4((2.0 * texcoords) - vec2(1.0), 0.0, 1.0);\n"
                                        "}\n";

static std::string extractorFS = "#ifdef GL_ES\n"
                                 "precision highp float;\n"
                                 "precision highp iimage2D;\n"
                                 "#endif\n"
                                 "layout (binding = 0, offset = 0) uniform atomic_uint bigSigmaCounterLowThrs;\n"
                                 "layout (binding = 0, offset = 4) uniform atomic_uint bigSigmaCounterHighThrs;\n"
                                 "layout (binding = 0, offset = 8) uniform atomic_uint smallSigmaCounterLowThrs;\n"
                                 "layout (binding = 0, offset = 12) uniform atomic_uint smallSigmaCounterHighThrs;\n"
                                 "layout (rgba32i) readonly uniform iimage2D bigSigmaImageR;\n"
                                 "layout (rgba32i) writeonly uniform iimage2D bigSigmaImageW;\n"
                                 "layout (rgba32i) readonly uniform iimage2D smallSigmaImageR;\n"
                                 "layout (rgba32i) writeonly uniform iimage2D smallSigmaImageW;\n"
                                 "uniform sampler2D tex;\n" //r big sigma, b big sigma
                                 "uniform float w;\n"
                                 "uniform float h;\n"
                                 "uniform int idx;\n"
                                 "in vec2 texcoords;\n"
                                 "\n"
                                 "void main()\n"
                                 "{\n"
                                 "    ivec2 pos = ivec2(int(w * texcoords.x), int(h * texcoords.y));\n"
                                 "    ivec4 pos_low = ivec4(0, 0, pos);\n"
                                 "\n"
                                 "    vec2 p = texture(tex, texcoords).rg;\n"
                                 "    if (p.r > $THRESHOLD_HIGH)\n" //big sigma, high thrs
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(bigSigmaCounterHighThrs));\n"
                                 "         int j = int(atomicCounterIncrement(bigSigmaCounterLowThrs));\n"
                                 "         if (j < $NB_MAX_KEYPOINTS)\n"
                                 "             imageStore(bigSigmaImageW, ivec2(j, idx), pos_low);\n"
                                 "         \n"
                                 "         if (i < $NB_MAX_KEYPOINTS)\n"
                                 "         {\n"
                                 "             ivec4 lastSave = imageLoad(bigSigmaImageR, ivec2(i, idx));\n" //A low thrs kp may be already saved at this position
                                 "             lastSave.rg = pos;\n"                                         //{b,a} should be already set as low thrs point
                                 "             imageStore(bigSigmaImageW, ivec2(i, idx), lastSave);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (p.r > $THRESHOLD_LOW)\n"
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(bigSigmaCounterLowThrs));\n"
                                 "         if (i < $NB_MAX_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(bigSigmaImageW, ivec2(i, idx), pos_low);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    \n"
                                 "    if (p.g > $THRESHOLD_HIGH)\n" //big sigma, high thrs
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(smallSigmaCounterHighThrs));\n"
                                 "         int j = int(atomicCounterIncrement(smallSigmaCounterLowThrs));\n"
                                 "         if (j < $NB_MAX_KEYPOINTS)\n"
                                 "             imageStore(smallSigmaImageW, ivec2(j, idx), pos_low);\n"
                                 "         \n"
                                 "         if (i < $NB_MAX_KEYPOINTS)\n"
                                 "         {\n"
                                 "             ivec4 lastSave = imageLoad(smallSigmaImageR, ivec2(i, idx));\n" //A low thrs kp may be already saved at this position
                                 "             lastSave.rg = pos;\n"                                           //{b,a} should be already set as low thrs point
                                 "             imageStore(smallSigmaImageW, ivec2(i, idx), lastSave);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (p.g > $THRESHOLD_LOW)\n"
                                 "    {\n"
                                 "         int i = int(atomicCounterIncrement(smallSigmaCounterLowThrs));\n"
                                 "         if (i < $NB_MAX_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(smallSigmaImageW, ivec2(i, idx), pos_low);\n"
                                 "         }\n"
                                 "    }\n"
                                 "}\n";

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
                                 "layout (binding = 0, offset = 0) uniform atomic_uint smallSigmaCounter;\n"
                                 "layout (binding = 0, offset = 4) uniform atomic_uint bigSigmaCounter;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D bigSigmaImage;\n"
                                 "layout (rgba32i) uniform writeonly iimage2D smallSigmaImage;\n"
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
                                 "         int ih = int(atomicCounterIncrement(smallSigmaCounter));\n"
                                 "         int il = int(atomicCounterIncrement(bigSigmaCounter));\n"
                                 "         if (il < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(smallSigmaImage, ivec2(ih, idx), pos);\n"
                                 "             imageStore(bigSigmaImage, ivec2(il, idx), pos);\n"
                                 "         }\n"
                                 "         if (ih < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(smallSigmaImage, ivec2(ih, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "    else if (r > $LOW_THRESHOLD)\n"
                                 "    {\n"
                                 "         int il = int(atomicCounterIncrement(bigSigmaCounter));\n"
                                 "         if (il < $NB_KEYPOINTS)\n"
                                 "         {\n"
                                 "             imageStore(bigSigmaImage, ivec2(il, idx), pos);\n"
                                 "         }\n"
                                 "    }\n"
                                 "}\n"
                                 ;
*/

GLuint GLSLHessian::buildShaderFromSource(string source, GLenum shaderType)
{
    // Compile Shader code
    GLuint     shaderHandle = glCreateShader(shaderType);
    string     version;
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
    Utils::replaceString(completeSrc, "$NB_MAX_KEYPOINTS", nbKeypointsBigSigmaStr);
    Utils::replaceString(completeSrc, "$THRESHOLD_HIGH", highThresholdStr);
    Utils::replaceString(completeSrc, "$THRESHOLD_LOW", lowThresholdStr);

    std::cout << "nb keypoints " << nbKeypointsBigSigmaStr << "  high thrs " << highThresholdStr << "  low thrs " << lowThresholdStr << std::endl;

    const char* src = completeSrc.c_str();

    glShaderSource(shaderHandle, 1, &src, nullptr);
    glCompileShader(shaderHandle);

    // Check compile success
    GLint compileSuccess;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compileSuccess);

    GLint logSize = 0;
    glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &logSize);

    GLchar* log = new GLchar[logSize];

    glGetShaderInfoLog(shaderHandle, logSize, nullptr, log);

    if (!compileSuccess)
    {
        Utils::log("AAAA Cannot compile shader %s", log);
        Utils::log("AAAA %s", src);
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

    extractorTexLoc     = glGetUniformLocation(extractor, "tex");
    extractorPatternLoc = glGetUniformLocation(extractor, "pattern");
    extractorOffsetLoc  = glGetUniformLocation(extractor, "ofst");
    extractorSizeLoc    = glGetUniformLocation(extractor, "s");
    extractorIdxLoc     = glGetUniformLocation(extractor, "idx");
    extractorWLoc       = glGetUniformLocation(extractor, "w");
    extractorHLoc       = glGetUniformLocation(extractor, "h");

    extractorBigSigmaCountersLowThrsLoc    = glGetUniformLocation(extractor, "bigSigmaCounterLowThrs");
    extractorBigSigmaCountersHighThrsLoc   = glGetUniformLocation(extractor, "bigSigmaCounterHighThrs");
    extractorSmallSigmaCountersLowThrsLoc  = glGetUniformLocation(extractor, "smallSigmaCountersLowThrs");
    extractorSmallSigmaCountersHighThrsLoc = glGetUniformLocation(extractor, "smallSigmaCountersHighThrs");
    extractorBigSigmaImageRLoc             = glGetUniformLocation(extractor, "bigSigmaImageR");
    extractorBigSigmaImageWLoc             = glGetUniformLocation(extractor, "bigSigmaImageW");
    extractorSmallSigmaImageRLoc           = glGetUniformLocation(extractor, "smallSigmaImageR");
    extractorSmallSigmaImageWLoc           = glGetUniformLocation(extractor, "smallSigmaImageW");
}

void GLSLHessian::initVBO()
{
    float vertices[12] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};

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
    int i[4] = {0};
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 16, i, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

void GLSLHessian::initKeypointBuffers()
{
    /* Buffers to store keypoints */
    glGenTextures(1, &smallSigmaImages);
    glGenTextures(1, &bigSigmaImages);

    glGenFramebuffers(1, &smallSigmaImagesFB);
    glGenFramebuffers(1, &bigSigmaImagesFB);

    glGenBuffers(2, smallSigmaImagePBOs);
    glGenBuffers(2, bigSigmaImagePBOs);

    int i[4] = {0};
    glGenBuffers(1, &atomicCounter);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicCounter);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 16, i, GL_DYNAMIC_DRAW);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);

    glClearColor(0, 0, 0, 0);

    glBindTexture(GL_TEXTURE_2D, smallSigmaImages);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypointsSmallSigma, 64);
    glBindFramebuffer(GL_FRAMEBUFFER, smallSigmaImagesFB);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, smallSigmaImages, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, bigSigmaImages);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32I, mNbKeypointsBigSigma, 64);
    glBindFramebuffer(GL_FRAMEBUFFER, bigSigmaImagesFB);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bigSigmaImages, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, smallSigmaImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypointsSmallSigma * 64 * 4 * 4, 0, GL_DYNAMIC_READ);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, bigSigmaImagePBOs[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, mNbKeypointsBigSigma * 64 * 4 * 4, 0, GL_DYNAMIC_READ);
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
    glUniform1i(extractorBigSigmaImageRLoc, 0);
    glUniform1i(extractorBigSigmaImageWLoc, 0);
    glUniform1i(extractorSmallSigmaImageRLoc, 1);
    glUniform1i(extractorSmallSigmaImageWLoc, 1);

    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    float offsetx = 15.0 / (float)m_w;
    float offsety = 15.0 / (float)m_h;
    float sizex   = ((float)m_w - 30.0) / (8.0 * float(m_w));
    float sizey   = ((float)m_h - 30.0) / (8.0 * float(m_h));

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
    float v = 2.0 * (1.0 / sigma) * exp(-(halfSize * halfSize) / (2.0 * sigma * sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - halfSize);
        float v = 2.0 * (1.0 / sigma) * exp(-(x * x) / (2.0 * sigma * sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

string GLSLHessian::gaussianD1(int size, int half_size, float sigma)
{
    float v = -2.0 * (half_size / (sigma * sigma * sigma)) * exp(-(half_size * half_size) / (2.0 * sigma * sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - half_size);
        float v = -2.0 * (x / (sigma * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

string GLSLHessian::gaussianD2(int size, int half_size, float sigma)
{
    float v = 2.0 * (half_size * half_size - sigma * sigma) / (sigma * sigma * sigma * sigma * sigma) * exp(-(half_size * half_size) / (2.0 * sigma * sigma));

    string fctStr = std::to_string(v);

    for (int i = 1; i < size; i++)
    {
        float x = (float)(i - half_size);
        float v = 2.0 * (x * x - sigma * sigma) / (sigma * sigma * sigma * sigma * sigma) * exp(-(x * x) / (2.0 * sigma * sigma));

        fctStr = fctStr + ", ";
        fctStr = fctStr + std::to_string(v);
    }
    return fctStr;
}

void GLSLHessian::init(int w, int h, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma)
{
    m_w                      = w;
    m_h                      = h;
    curr                     = 1;
    ready                    = 0;
    mNbKeypointsBigSigma     = nbKeypointsBigSigma;
    mNbKeypointsSmallSigma   = nbKeypointsSmallSigma;
    nbKeypointsBigSigmaStr   = std::to_string(nbKeypointsBigSigma);
    nbKeypointsSmallSigmaStr = std::to_string(nbKeypointsSmallSigma);
    highThresholdStr         = std::to_string(highThrs);
    lowThresholdStr          = std::to_string(lowThrs);

    //At a radius of 3 sigma from the center, we keep ~97% of the gaussian fct.
    // | 0x1 to ensure this is a odd number (not divisible per 2)
    int    size     = ((int)floor(bigSigma * 6.0)) | 0x1;
    int    halfSize = size >> 1;
    string sz_s     = to_string(size);

    std::string gaussianBigSigmaKernelStr   = "const float bigSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussian(size, halfSize, bigSigma) + ");\n";
    std::string gaussianD1BigSigmaKernelStr = "const float bigSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD1(size, halfSize, bigSigma) + ");\n";
    std::string gaussianD2BigSigmaKernelStr = "const float bigSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD2(size, halfSize, bigSigma) + ");\n";

    std::string gaussianSmallSigmaKernelStr   = "const float smallSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussian(size, halfSize, smallSigma) + ");\n";
    std::string gaussianD1SmallSigmaKernelStr = "const float smallSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD1(size, halfSize, smallSigma) + ");\n";
    std::string gaussianD2SmallSigmaKernelStr = "const float smallSigmaKernel[" + sz_s + "] = float[" + sz_s + "](" + gaussianD2(size, halfSize, smallSigma) + ");\n";

    gaussianKernelStr   = gaussianBigSigmaKernelStr + gaussianSmallSigmaKernelStr;
    gaussianD1KernelStr = gaussianD1BigSigmaKernelStr + gaussianD1SmallSigmaKernelStr;
    gaussianD2KernelStr = gaussianD2BigSigmaKernelStr + gaussianD2SmallSigmaKernelStr;

    kernelSizeStr = "const float kHalfSize = " + to_string((float)halfSize) + ";\nconst int kSize = " + sz_s + ";\n";

    initShaders();
    initVBO();
    initTextureBuffers(w, h);
    initKeypointBuffers();
    initFBO();
}

GLSLHessian::GLSLHessian() {}

GLSLHessian::GLSLHessian(int w, int h, int nbKeypointsBigSigma, int nbKeypointsSmallSigma, float highThrs, float lowThrs, float bigSigma, float smallSigma)
{
    init(w, h, nbKeypointsBigSigma, nbKeypointsSmallSigma, highThrs, lowThrs, bigSigma, smallSigma);
}

GLSLHessian::~GLSLHessian()
{
    if (externalTexture)
    {
        glDeleteTextures(11, renderTextures + 1);
    }
    else
    {
        glDeleteTextures(12, renderTextures);
    }

    glDeleteFramebuffers(12, renderFBO);

    glDeleteBuffers(1, &atomicCounter);
    glDeleteFramebuffers(1, &bigSigmaImagesFB);
    glDeleteFramebuffers(1, &smallSigmaImagesFB);

    glDeleteTextures(1, &bigSigmaImages);
    glDeleteTextures(1, &smallSigmaImages);

    glDeleteBuffers(2, bigSigmaImagePBOs);
    glDeleteBuffers(2, smallSigmaImagePBOs);

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

void GLSLHessian::setInputTexture(SLGLTexture& tex)
{
    if (!externalTexture)
    {
        Utils::log("Error", "externalTexture is not set");
        return;
    }
    renderTextures[0] = (GLuint)tex.texID();
}

void GLSLHessian::setInputTexture(cv::Mat& image)
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
    curr  = (curr + 1) % 2; //Set rendering buffers

    SLVec4i wp = SLGLState::instance()->getViewport();
    SLGLState::instance()->viewport(0, 0, m_w, m_h);

    for (int i = 0; i < 12; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
    }

    glClearColor(0, 0, 0, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, bigSigmaImagesFB);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, smallSigmaImagesFB);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindImageTexture(0, bigSigmaImages, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glBindImageTexture(1, smallSigmaImages, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);

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

void GLSLHessian::readResult(std::vector<cv::KeyPoint>& kps)
{
    glBindFramebuffer(GL_FRAMEBUFFER, bigSigmaImagesFB);
    glFlush();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, bigSigmaImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypointsBigSigma, 64, GL_RGBA_INTEGER, GL_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, smallSigmaImagesFB);
    glFlush();
    glBindBuffer(GL_PIXEL_PACK_BUFFER, smallSigmaImagePBOs[curr]);
    glReadPixels(0, 0, mNbKeypointsSmallSigma, 64, GL_RGBA_INTEGER, GL_INT, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, bigSigmaImagePBOs[ready]);
    unsigned int* bigSigmaData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypointsBigSigma * 64 * 4 * 4, GL_MAP_READ_BIT);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, smallSigmaImagePBOs[ready]);
    unsigned int* smallSigmaData = (unsigned int*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, mNbKeypointsSmallSigma * 64 * 4 * 4, GL_MAP_READ_BIT);

    if (bigSigmaData && smallSigmaData)
    {
        for (int i = 0; i < 64; i++)
        {
            int n = 0;
            int j = 0;

            if (bigSigmaData[8 * 4] > 0) //If there are more than 8 keypoints with high threshold, take high threshold points
            {
                for (j = 0; j < mNbKeypointsBigSigma; j++)
                {
                    int idx       = (i * mNbKeypointsBigSigma + j) * 4; //4 channels
                    int highThrsX = bigSigmaData[idx + 0];
                    int highThrsY = bigSigmaData[idx + 1];

                    if (highThrsX == 0)
                    {
                        break;
                    }

                    if (highThrsX > 15 && highThrsX < m_w - 15 && highThrsY > 15 && highThrsY < m_h - 15)
                        kps.push_back(cv::KeyPoint(cv::Point2f(highThrsX, highThrsY), 1));
                }
            }
            else
            {
                for (j = 0; j < mNbKeypointsBigSigma; j++)
                {
                    int idx      = (i * mNbKeypointsBigSigma + j) * 4;
                    int lowThrsX = bigSigmaData[idx + 2];
                    int lowThrsY = bigSigmaData[idx + 3];
                    if (lowThrsX == 0)
                    {
                        break;
                    }
                    if (lowThrsX > 15 && lowThrsX < m_w - 15 && lowThrsY > 15 && lowThrsY < m_h - 15)
                        kps.push_back(cv::KeyPoint(cv::Point2f(lowThrsX, lowThrsY), 1));
                }
            }

            n = j;
            if (n < 8) // if there are less than 8 keypoints with THICK corners, take the one with higher granularity
            {
                if (smallSigmaData[8 * 4] > 0) //If there are more than 8 keypoints with high threshold, take high threshold points
                {
                    for (j = 0; j < mNbKeypointsSmallSigma; j++)
                    {
                        int idx       = (i * mNbKeypointsSmallSigma + j) * 4;
                        int highThrsX = smallSigmaData[idx + 0];
                        int highThrsY = smallSigmaData[idx + 1];
                        if (highThrsX == 0)
                        {
                            break;
                        }
                        if (highThrsX > 15 && highThrsX < m_w - 15 && highThrsY > 15 && highThrsY < m_h - 15)
                            kps.push_back(cv::KeyPoint(cv::Point2f(highThrsX, highThrsY), 1));
                    }
                }
                else
                {
                    for (j = 0; j < mNbKeypointsSmallSigma; j++)
                    {
                        int idx      = (i * mNbKeypointsSmallSigma + j) * 4;
                        int lowThrsX = smallSigmaData[idx + 2];
                        int lowThrsY = smallSigmaData[idx + 3];
                        if (lowThrsX == 0)
                        {
                            break;
                        }
                        if (lowThrsX > 15 && lowThrsX < m_w - 15 && lowThrsY > 15 && lowThrsY < m_h - 15)
                            kps.push_back(cv::KeyPoint(cv::Point2f(lowThrsX, lowThrsY), 1));
                    }
                }
            }
        }
    }

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, bigSigmaImagePBOs[ready]);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}
