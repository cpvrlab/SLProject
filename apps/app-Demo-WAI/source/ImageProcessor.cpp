#include <AverageTiming.h>
#include <ImageProcessor.h>
#include <SLSceneView.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <CVCalibration.h>

#define INPUT 0
#define BLURREDX 1
#define BLURRED 2
#define D2GDX2 3
#define D2GDY2 4
#define DGDX 5
#define GXX 6
#define GYY 7
#define GXY 8
#define DETH 9
#define NMSX 10
#define NMSY 11

//#version 320 es
static std::string screeQuadVs = "#version 330\n"
                                 "layout (location = 0) in vec3 vcoords;\n"
                                 "out vec2 texcoords;\n"
                                 "\n"
                                 "void main()\n"
                                 "{\n"
                                 "    texcoords = 0.5 * (vcoords.xy + vec2(1.0));\n"
                                 "    gl_Position = vec4(vcoords, 1.0);\n"
                                 "}\n";

static std::string hGaussian1chFs = "#version 330\n"
                                    "out float pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "\n"
                                    "const float kernel05[7] = float[7]("
                                    "0.012560200468474598, 0.07882796468172999, 0.23729607711717057, 0.3426315154652495, 0.23729607711717066, 0.07882796468173005, 0.012560200468474616);"
                                    "\n"
                                    "    \n"
                                    "    float response = 0.0;\n"
                                    "    for (int i = 0; i < 7; i++)\n"
                                    "    {\n"
                                    "        float ofst = (float(i) - 3.0) / w;\n"
                                    "        response += kernel05[i] * texture(tex, texcoords + vec2(ofst, 0.0)).r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string vGaussian1chFs = "#version 330\n"
                                    "out float pixel;\n"
                                    "in vec2 texcoords;\n"
                                    "uniform float w;\n"
                                    "uniform sampler2D tex;\n"
                                    "\n"
                                    "void main()\n"
                                    "{\n"
                                    "\n"
                                    "const float kernel05[7] = float[7]("
                                    "0.012560200468474598, 0.07882796468172999, 0.23729607711717057, 0.3426315154652495, 0.23729607711717066, 0.07882796468173005, 0.012560200468474616);"
                                    "\n"
                                    "    \n"
                                    "    float response = 0.0;\n"
                                    "    for (int i = 0; i < 7; i++)\n"
                                    "    {\n"
                                    "        float ofst = (float(i) - 3.0) / w;\n"
                                    "        response += kernel05[i] * texture(tex, texcoords + vec2(0.0, ofst)).r;\n"
                                    "    }\n"
                                    "    pixel = response;\n"
                                    "}\n";

static std::string hGaussianFs = "#version 330\n"
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

static std::string vGaussianFs = "#version 330\n"
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

static std::string hGaussianDxFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.01853595173921492, 0.06586358220027054, 0.1333839310872644, 0.12989821155417797, 4.5025992541296685e-17, -0.12989821155417794, -0.13338393108726446, -0.06586358220027058, -0.018535951739214945);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.006284601927830796, 0.015240430887981849, 0.030619291961256016, 0.05032423223328462, 0.06607593710199453, 0.06571580992569773, 0.04177058376536309, 1.2559268403669998e-17, -0.04177058376536307, -0.06571580992569771, -0.06607593710199453, -0.05032423223328462, -0.030619291961256016, -0.015240430887981849, -0.006284601927830796);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.003293818707797897, 0.006437867048236782, 0.011453459188775942, 0.018485844966375946, 0.026936085804141863, 0.03516758938635828, 0.04062275143556737, 0.040542893121886005, 0.0331478224816762, 0.018732852987741393, -0.0, -0.018732852987741393, -0.0331478224816762, -0.040542893121886005, -0.04062275143556737, -0.03516758938635828, -0.026936085804141863, -0.018485844966375946, -0.011453459188775942, -0.006437867048236782, -0.003293818707797897);\n"
                               "\n"
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


static std::string vGaussianDyFs = "#version 330\n"
                               "out vec3 pixel;\n"
                               "in vec2 texcoords;\n"
                               "uniform float w;\n"
                               "uniform sampler2D tex;\n"
                               "\n"
                               "void main()\n"
                               "{\n"
                               "\n"
                               "const float kernel12[9] = float[9]("
                               "0.01853595173921492, 0.06586358220027054, 0.1333839310872644, 0.12989821155417797, 4.5025992541296685e-17, -0.12989821155417794, -0.13338393108726446, -0.06586358220027058, -0.018535951739214945);\n"
                               "\n"
                               "const float kernel20[15] = float[15]("
                               "0.006284601927830796, 0.015240430887981849, 0.030619291961256016, 0.05032423223328462, 0.06607593710199453, 0.06571580992569773, 0.04177058376536309, 1.2559268403669998e-17, -0.04177058376536307, -0.06571580992569771, -0.06607593710199453, -0.05032423223328462, -0.030619291961256016, -0.015240430887981849, -0.006284601927830796);\n"
                               "\n"
                               "const float kernel28[21] = float[21]("
                               "0.003293818707797897, 0.006437867048236782, 0.011453459188775942, 0.018485844966375946, 0.026936085804141863, 0.03516758938635828, 0.04062275143556737, 0.040542893121886005, 0.0331478224816762, 0.018732852987741393, -0.0, -0.018732852987741393, -0.0331478224816762, -0.040542893121886005, -0.04062275143556737, -0.03516758938635828, -0.026936085804141863, -0.018485844966375946, -0.011453459188775942, -0.006437867048236782, -0.003293818707797897);\n"
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


static std::string hGaussianDx2Fs  = "#version 330\n"
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

static std::string vGaussianDy2Fs  = "#version 330\n"
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

static std::string detHFs = "#version 330\n"
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
                            "    vec3 det = gxx * gyy - gxy * gxy;\n"
                            "    pixel = det;\n"
                            "}\n";

static std::string nmsxFs = "#version 330\n"
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
                            "       pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= nx.g || o.g <= px.g)\n"
                            "    {\n"
                            "       pixel.g = 0.0;\n"
                            "    }\n"
                            "    if (o.b <= nx.b || o.b <= px.b)\n"
                            "    {\n"
                            "       pixel.b = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string nmsyFs = "#version 330\n"
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
                            "       pixel.r = 0.0;\n"
                            "    }\n"
                            "    if (o.g <= ny.g || o.g <= py.g)\n"
                            "    {\n"
                            "       pixel.g = 0.0;\n"
                            "    }\n"
                            "    if (o.b <= ny.b || o.b <= py.b)\n"
                            "    {\n"
                            "       pixel.b = 0.0;\n"
                            "    }\n"
                            "}\n";

static std::string nmszFs = "#version 330\n"
                            "out float pixel;\n"
                            "in vec2 texcoords;\n"
                            "uniform sampler2D tex;\n"
                            "\n"
                            "void main()\n"
                            "{\n"
                            "    \n"
                            "    float l0 = texture(tex, texcoords).r;\n"
                            "    float l1 = texture(tex, texcoords).g;\n"
                            "    float l2 = texture(tex, texcoords).b;\n"
                            "    pixel = l1;\n"
                            "}\n";

GLuint ImageProcessor::buildShaderFromSource(string source, GLenum shaderType)
{
    // Compile Shader code
    GLuint      shaderHandle = glCreateShader(shaderType);
    const char* src          = source.c_str();
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
        Utils::log("Cannot compile shader %s\n", log);
        std::cout << source << std::endl;
        exit(1);
    }
    return shaderHandle;
}

void ImageProcessor::initShaders()
{
    GLuint vscreenQuad = buildShaderFromSource(screeQuadVs, GL_VERTEX_SHADER);
    GLuint fd2Gdx2 = buildShaderFromSource(hGaussianDx2Fs, GL_FRAGMENT_SHADER);
    GLuint fd2Gdy2 = buildShaderFromSource(vGaussianDy2Fs, GL_FRAGMENT_SHADER);
    GLuint fdGdx = buildShaderFromSource(hGaussianDxFs, GL_FRAGMENT_SHADER);
    GLuint fdGdy = buildShaderFromSource(vGaussianDyFs, GL_FRAGMENT_SHADER);
    GLuint fGx = buildShaderFromSource(hGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fGy = buildShaderFromSource(vGaussianFs, GL_FRAGMENT_SHADER);
    GLuint fGx1ch = buildShaderFromSource(hGaussian1chFs, GL_FRAGMENT_SHADER);
    GLuint fGy1ch = buildShaderFromSource(vGaussian1chFs, GL_FRAGMENT_SHADER);
    GLuint fdetH = buildShaderFromSource(detHFs, GL_FRAGMENT_SHADER);
    GLuint fnmsx = buildShaderFromSource(nmsxFs, GL_FRAGMENT_SHADER);
    GLuint fnmsy = buildShaderFromSource(nmsyFs, GL_FRAGMENT_SHADER);
    GLuint fnmsz = buildShaderFromSource(nmszFs, GL_FRAGMENT_SHADER);

    d2Gdx2      = glCreateProgram();
    d2Gdy2      = glCreateProgram();
    dGdx        = glCreateProgram();
    dGdy        = glCreateProgram();
    Gx          = glCreateProgram();
    Gy          = glCreateProgram();
    Gx1ch       = glCreateProgram();
    Gy1ch       = glCreateProgram();
    detH        = glCreateProgram();
    nmsx        = glCreateProgram();
    nmsy        = glCreateProgram();
    nmsz        = glCreateProgram();

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

    glAttachShader(Gx1ch, vscreenQuad);
    glAttachShader(Gx1ch, fGx1ch);
    glLinkProgram(Gx1ch);

    glAttachShader(Gy1ch, vscreenQuad);
    glAttachShader(Gy1ch, fGy1ch);
    glLinkProgram(Gy1ch);

    glAttachShader(detH, vscreenQuad);
    glAttachShader(detH, fdetH);
    glLinkProgram(detH);

    glAttachShader(nmsx, vscreenQuad);
    glAttachShader(nmsx, fnmsx);
    glLinkProgram(nmsx);

    glAttachShader(nmsy, vscreenQuad);
    glAttachShader(nmsy, fnmsy);
    glLinkProgram(nmsy);

    glAttachShader(nmsz, vscreenQuad);
    glAttachShader(nmsz, fnmsz);
    glLinkProgram(nmsz);

    d2Gdx2TexLoc = glGetUniformLocation(d2Gdx2, "tex");
    d2Gdx2WLoc = glGetUniformLocation(d2Gdx2, "w");
    d2Gdy2TexLoc = glGetUniformLocation(d2Gdy2, "tex");
    d2Gdy2WLoc = glGetUniformLocation(d2Gdy2, "w");
    dGdxTexLoc = glGetUniformLocation(dGdx, "tex");
    dGdxWLoc = glGetUniformLocation(dGdx, "w");
    dGdyTexLoc = glGetUniformLocation(dGdy, "tex");
    dGdyWLoc = glGetUniformLocation(dGdy, "w");
    GxTexLoc = glGetUniformLocation(Gx, "tex");
    GxWLoc = glGetUniformLocation(Gx, "w");
    GyTexLoc = glGetUniformLocation(Gy, "tex");
    GyWLoc = glGetUniformLocation(Gy, "w");
    Gx1chTexLoc = glGetUniformLocation(Gx1ch, "tex");
    Gx1chWLoc = glGetUniformLocation(Gx1ch, "w");
    Gy1chTexLoc = glGetUniformLocation(Gy1ch, "tex");
    Gy1chWLoc = glGetUniformLocation(Gy1ch, "w");
    detHGxxLoc = glGetUniformLocation(detH, "tgxx");
    detHGyyLoc = glGetUniformLocation(detH, "tgyy");
    detHGxyLoc = glGetUniformLocation(detH, "tgxy");
    nmsxTexLoc = glGetUniformLocation(nmsx, "tex");
    nmsxWLoc = glGetUniformLocation(nmsx, "w");
    nmsyTexLoc = glGetUniformLocation(nmsy, "tex");
    nmsyWLoc = glGetUniformLocation(nmsy, "w");
    nmszTexLoc = glGetUniformLocation(nmsz, "tex");
}

void ImageProcessor::initVBO()
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

void ImageProcessor::setTextureParameters()
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
}

void ImageProcessor::textureRGBF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
}

void ImageProcessor::textureRF(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_FLOAT, nullptr);
}

void ImageProcessor::textureRB(int w, int h)
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
}

void ImageProcessor::initTextureBuffers(int width, int height)
{
    glGenTextures(12, renderTextures);
    glGenTextures(2, outTextures);

    glBindTexture(GL_TEXTURE_2D, renderTextures[0]);
    setTextureParameters();
    textureRB(width, height);

    int i = 1;
    for (; i < 3; i++)
    {
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
        setTextureParameters();
        textureRF(width, height);
    }

    for (; i < 12; i++)
    {
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
        setTextureParameters();
        textureRGBF(width, height);
    }

    for (i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, outTextures[i]);
        setTextureParameters();
        textureRB(width, height);
    }

    glGenBuffers(2, pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, width * height, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void ImageProcessor::initFBO()
{
    glGenFramebuffers(12, renderFBO);
    glGenFramebuffers(2, outFBO);

    for (int i = 0; i < 12; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTextures[i], 0);
    }

    for (int i = 0; i < 2; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, outFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTextures[i], 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ImageProcessor::blur(int w, int h)
{
    glUseProgram(Gx1ch);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[BLURREDX]);

    glUniform1i(Gx1chTexLoc, INPUT);
    glUniform1f(Gx1chWLoc, (float)w);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glUseProgram(Gy1ch);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[BLURRED]);

    glUniform1i(Gy1chTexLoc, BLURREDX);
    glUniform1f(GyWLoc, (float)h);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void ImageProcessor::gxx(int w, int h)
{
    glUseProgram(d2Gdx2);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[D2GDX2]);

    glUniform1i(d2Gdx2TexLoc, BLURRED);
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

void ImageProcessor::gyy(int w, int h)
{
    glUseProgram(d2Gdy2);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[D2GDY2]);

    glUniform1i(d2Gdy2TexLoc, BLURRED);
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

void ImageProcessor::gxy(int w, int h)
{
    glUseProgram(dGdx);
    glBindFramebuffer(GL_FRAMEBUFFER, renderFBO[DGDX]);

    glUniform1i(dGdxTexLoc, BLURRED);
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

void ImageProcessor::det(int w, int h)
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

void ImageProcessor::nms(int w, int h, int id)
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

    glUseProgram(nmsz);
    glBindFramebuffer(GL_FRAMEBUFFER, outFBO[id]);
    glUniform1i(nmszTexLoc, NMSY);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboi);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void ImageProcessor::init(int w, int h)
{
    m_w = w;
    m_h = h;
    curr = 1;
    ready = 0;
    initShaders();
    initVBO();
    initTextureBuffers(w, h);
    initFBO();
}

ImageProcessor::ImageProcessor(){}

ImageProcessor::ImageProcessor(int w, int h)
{
    init(w, h);
}

void ImageProcessor::gpu_kp()
{
    glDisable(GL_DEPTH_TEST);

    ready = curr;
    curr = (curr+1) % 2; //Set rendering F

    for (int i = 0; i < 12; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, renderTextures[i]);
    }

    blur(m_w, m_h);
    gxx(m_w, m_h);
    gyy(m_w, m_h);
    gxy(m_w, m_h);
    det(m_w, m_h);
    nms(m_w, m_h, curr);

    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
}


void ImageProcessor::readResult(SLGLTexture * tex)
{
    HighResTimer t;
    t.start();
    AVERAGE_TIMING_START("PBO");

    /* Copy pixel to curr pbo */
    glBindFramebuffer(GL_FRAMEBUFFER, outFBO[ready]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[curr]);
    glReadPixels(0, 0, m_w, m_h, GL_RED, GL_UNSIGNED_BYTE, 0);
    /* Continue processing without stall */

    /* Read pixels from ready pbo */
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[ready]); //Read pixel from ready pbo
    //unsigned char * data = (unsigned char*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, m_w * m_h, GL_MAP_READ_BIT);
    unsigned char * data = (unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    Utils::log("timing %f\n", (float)t.elapsedTimeInMicroSec());

    AVERAGE_TIMING_STOP("PBO");

    if (data)
        tex->copyVideoImage(m_w, m_h, PF_red, PF_red, data, true, true);
    else
        std::cout << "null data" << std::endl;

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ImageProcessor::readResult(cv::Mat * tex)
{
    //HighResTimer t;
    //t.start();
    //AVERAGE_TIMING_START("PBO");

    /* Copy pixel to curr pbo */
    glBindFramebuffer(GL_FRAMEBUFFER, outFBO[ready]);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[curr]);
    glReadPixels(0, 0, m_w, m_h, GL_RED, GL_UNSIGNED_BYTE, 0);
    /* Continue processing without stall */

    /* Read pixels from ready pbo */
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[ready]); //Read pixel from ready pbo
    //unsigned char * data = (unsigned char*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, m_w * m_h, GL_MAP_READ_BIT);
    unsigned char * data = (unsigned char*)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    //Utils::log("timing %f\n", (float)t.elapsedTimeInMicroSec());

    //AVERAGE_TIMING_STOP("PBO");

    if (data)
        *tex = cv::Mat(m_h, m_w, CV_8UC1, data);
    else
        std::cout << "null data" << std::endl;

    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
