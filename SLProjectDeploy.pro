##############################################################################
#  File:      SLProjectDeploy.pro
#  Purpose:   QMake project definition for SLProject deployment
#  Author:    Marcus Hudritsch
#  Date:      February 2014
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

textures.files = \
  ../_data/images/textures/brick0512_C.png \
  ../_data/images/textures/bricks1_0256_C.jpg \
  ../_data/images/textures/bricks1_0512_C.jpg \
  ../_data/images/textures/brickwall0512_C.jpg \
  ../_data/images/textures/brickwall0512_G.jpg \
  ../_data/images/textures/brickwall0512_H.jpg \
  ../_data/images/textures/brickwall0512_N.jpg \
  ../_data/images/textures/Checkerboard0512_C.png \
  ../_data/images/textures/Chess0256_C.bmp \
  ../_data/images/textures/CompileError.png \
  ../_data/images/textures/cursor.png \
  ../_data/images/textures/earth1024_C.jpg \
  ../_data/images/textures/earth1024_G.jpg \
  ../_data/images/textures/earth1024_H.jpg \
  ../_data/images/textures/earth1024_N.jpg \
  ../_data/images/textures/earthNight1024_C.jpg \
  ../_data/images/textures/earth2048_C.jpg \
  ../_data/images/textures/earth2048_N.jpg \
  ../_data/images/textures/earth2048_H.jpg \
  ../_data/images/textures/earth2048_G.jpg \
  ../_data/images/textures/earthNight2048_C.jpg \
  ../_data/images/textures/earthCloud1024_A.jpg \
  ../_data/images/textures/earthCloud1024_C.jpg \
  ../_data/images/textures/features_abstract.png \
  ../_data/images/textures/features_road.png \
  ../_data/images/textures/features_stones.png \
  ../_data/images/textures/grass0512_C.jpg \
  ../_data/images/textures/gray_0256_C.jpg \
  ../_data/images/textures/MuttenzerBox+X0512_C.png \
  ../_data/images/textures/MuttenzerBox+Y0512_C.png \
  ../_data/images/textures/MuttenzerBox+Z0512_C.png \
  ../_data/images/textures/MuttenzerBox-X0512_C.png \
  ../_data/images/textures/MuttenzerBox-Y0512_C.png \
  ../_data/images/textures/MuttenzerBox-Z0512_C.png \
  ../_data/images/textures/Pool+X0512_C.png \
  ../_data/images/textures/Pool+Y0512_C.png \
  ../_data/images/textures/Pool+Z0512_C.png \
  ../_data/images/textures/Pool-X0512_C.png \
  ../_data/images/textures/Pool-Y0512_C.png \
  ../_data/images/textures/Pool-Z0512_C.png \
  ../_data/images/textures/rusty-metal_2048C.png \
  ../_data/images/textures/rusty-metal_2048N.png \
  ../_data/images/textures/rusty-metal_2048M.png \
  ../_data/images/textures/rusty-metal_2048R.png \
  ../_data/images/textures/mountain_lake+X1024_C.jpg \
  ../_data/images/textures/mountain_lake-X1024_C.jpg \
  ../_data/images/textures/mountain_lake+Y1024_C.jpg \
  ../_data/images/textures/mountain_lake-Y1024_C.jpg \
  ../_data/images/textures/mountain_lake+Z1024_C.jpg \
  ../_data/images/textures/mountain_lake-Z1024_C.jpg \
  ../_data/images/textures/Testmap_0512_C.png \
  ../_data/images/textures/tile1_0256_C.jpg \
  ../_data/images/textures/tree1_1024_C.png \
  ../_data/images/textures/tron_floor2.png \
  ../_data/images/textures/VisionExample.png \
  ../_data/images/textures/VisionTest.png \
  ../_data/images/textures/Wave_radial10_256C.jpg \
  ../_data/images/textures/wood0_0256_C.jpg \
  ../_data/images/textures/wood0_0512_C.jpg \
  ../_data/images/textures/wood2_0256_C.jpg \
  ../_data/images/textures/wood2_0512_C.jpg \
  ../_data/images/textures/LiveVideoError.png \
  ../_data/images/textures/LogoCPVR_256L.png \
  ../_data/images/textures/i0000_0000b.png \
  ../_data/images/textures/i0001_0000b.png \
  ../_data/images/textures/i0002_0000b.png \
  ../_data/images/textures/i0003_0000b.png \
  ../_data/images/textures/i0004_0000b.png \
  ../_data/images/textures/i0005_0000b.png \
  ../_data/images/textures/i0006_0000b.png \
  ../_data/images/textures/i0007_0000b.png \
  ../_data/images/textures/i0008_0000b.png \
  ../_data/images/textures/i0009_0000b.png \
  ../_data/images/textures/i0010_0000b.png \
  ../_data/images/textures/i0011_0000b.png \
  ../_data/images/textures/i0012_0000b.png \
  ../_data/images/textures/i0013_0000b.png \
  ../_data/images/textures/i0014_0000b.png \
  ../_data/images/textures/i0015_0000b.png \
  ../_data/images/textures/i0016_0000b.png \
  ../_data/images/textures/i0017_0000b.png \
  ../_data/images/textures/i0018_0000b.png \
  ../_data/images/textures/i0019_0000b.png \
  ../_data/images/textures/i0020_0000b.png \
  ../_data/images/textures/i0021_0000b.png \
  ../_data/images/textures/i0022_0000b.png \
  ../_data/images/textures/i0023_0000b.png \
  ../_data/images/textures/i0024_0000b.png \
  ../_data/images/textures/i0025_0000b.png \
  ../_data/images/textures/i0026_0000b.png \
  ../_data/images/textures/i0027_0000b.png \
  ../_data/images/textures/i0028_0000b.png \
  ../_data/images/textures/i0029_0000b.png \
  ../_data/images/textures/i0030_0000b.png \
  ../_data/images/textures/i0031_0000b.png \
  ../_data/images/textures/i0032_0000b.png \
  ../_data/images/textures/i0033_0000b.png \
  ../_data/images/textures/i0034_0000b.png \
  ../_data/images/textures/i0035_0000b.png \
  ../_data/images/textures/i0036_0000b.png \
  ../_data/images/textures/i0037_0000b.png \
  ../_data/images/textures/i0038_0000b.png \
  ../_data/images/textures/i0039_0000b.png \
  ../_data/images/textures/i0040_0000b.png \
  ../_data/images/textures/i0041_0000b.png \
  ../_data/images/textures/i0042_0000b.png \
  ../_data/images/textures/i0043_0000b.png \
  ../_data/images/textures/i0044_0000b.png \
  ../_data/images/textures/i0045_0000b.png \
  ../_data/images/textures/i0046_0000b.png \
  ../_data/images/textures/i0047_0000b.png \
  ../_data/images/textures/i0048_0000b.png \
  ../_data/images/textures/i0049_0000b.png \
  ../_data/images/textures/i0050_0000b.png \
  ../_data/images/textures/i0051_0000b.png \
  ../_data/images/textures/i0052_0000b.png \
  ../_data/images/textures/i0053_0000b.png \
  ../_data/images/textures/i0054_0000b.png \
  ../_data/images/textures/i0055_0000b.png \
  ../_data/images/textures/i0056_0000b.png \
  ../_data/images/textures/i0057_0000b.png \
  ../_data/images/textures/i0058_0000b.png \
  ../_data/images/textures/i0059_0000b.png \
  ../_data/images/textures/i0060_0000b.png \
  ../_data/images/textures/i0061_0000b.png \
  ../_data/images/textures/i0062_0000b.png \
  ../_data/images/textures/i0063_0000b.png \
  ../_data/images/textures/i0064_0000b.png \
  ../_data/images/textures/i0065_0000b.png \
  ../_data/images/textures/i0066_0000b.png \
  ../_data/images/textures/i0067_0000b.png \
  ../_data/images/textures/i0068_0000b.png \
  ../_data/images/textures/i0069_0000b.png \
  ../_data/images/textures/i0070_0000b.png \
  ../_data/images/textures/i0071_0000b.png \
  ../_data/images/textures/i0072_0000b.png \
  ../_data/images/textures/i0073_0000b.png \
  ../_data/images/textures/i0074_0000b.png \
  ../_data/images/textures/i0075_0000b.png \
  ../_data/images/textures/i0076_0000b.png \
  ../_data/images/textures/i0077_0000b.png \
  ../_data/images/textures/i0078_0000b.png \
  ../_data/images/textures/i0079_0000b.png \
  ../_data/images/textures/i0080_0000b.png \
  ../_data/images/textures/i0081_0000b.png \
  ../_data/images/textures/i0082_0000b.png \
  ../_data/images/textures/i0083_0000b.png \
  ../_data/images/textures/i0084_0000b.png \
  ../_data/images/textures/i0085_0000b.png \
  ../_data/images/textures/i0086_0000b.png \
  ../_data/images/textures/i0087_0000b.png \
  ../_data/images/textures/i0088_0000b.png \
  ../_data/images/textures/i0089_0000b.png \
  ../_data/images/textures/i0090_0000b.png \
  ../_data/images/textures/i0091_0000b.png \
  ../_data/images/textures/i0092_0000b.png \
  ../_data/images/textures/i0093_0000b.png \
  ../_data/images/textures/i0094_0000b.png \
  ../_data/images/textures/i0095_0000b.png \
  ../_data/images/textures/i0096_0000b.png \
  ../_data/images/textures/i0097_0000b.png \
  ../_data/images/textures/i0098_0000b.png \
  ../_data/images/textures/i0099_0000b.png \
  ../_data/images/textures/i0100_0000b.png \
  ../_data/images/textures/i0101_0000b.png \
  ../_data/images/textures/i0102_0000b.png \
  ../_data/images/textures/i0103_0000b.png \
  ../_data/images/textures/i0104_0000b.png \
  ../_data/images/textures/i0105_0000b.png \
  ../_data/images/textures/i0106_0000b.png \
  ../_data/images/textures/i0107_0000b.png \
  ../_data/images/textures/i0108_0000b.png \
  ../_data/images/textures/i0109_0000b.png \
  ../_data/images/textures/i0110_0000b.png \
  ../_data/images/textures/i0111_0000b.png \
  ../_data/images/textures/i0112_0000b.png \
  ../_data/images/textures/i0113_0000b.png \
  ../_data/images/textures/i0114_0000b.png \
  ../_data/images/textures/i0115_0000b.png \
  ../_data/images/textures/i0116_0000b.png \
  ../_data/images/textures/i0117_0000b.png \
  ../_data/images/textures/i0118_0000b.png \
  ../_data/images/textures/i0119_0000b.png \
  ../_data/images/textures/i0120_0000b.png \
  ../_data/images/textures/i0121_0000b.png \
  ../_data/images/textures/i0122_0000b.png \
  ../_data/images/textures/i0123_0000b.png \
  ../_data/images/textures/i0124_0000b.png \
  ../_data/images/textures/i0125_0000b.png \
  ../_data/images/textures/i0126_0000b.png \
  ../_data/images/textures/i0127_0000b.png \
  ../_data/images/textures/i0128_0000b.png \
  ../_data/images/textures/i0129_0000b.png \
  ../_data/images/textures/i0130_0000b.png \
  ../_data/images/textures/i0131_0000b.png \
  ../_data/images/textures/i0132_0000b.png \
  ../_data/images/textures/i0133_0000b.png \
  ../_data/images/textures/i0134_0000b.png \
  ../_data/images/textures/i0135_0000b.png \
  ../_data/images/textures/i0136_0000b.png \
  ../_data/images/textures/i0137_0000b.png \
  ../_data/images/textures/i0138_0000b.png \
  ../_data/images/textures/i0139_0000b.png \
  ../_data/images/textures/i0140_0000b.png \
  ../_data/images/textures/i0141_0000b.png \
  ../_data/images/textures/i0142_0000b.png \
  ../_data/images/textures/i0143_0000b.png \
  ../_data/images/textures/i0144_0000b.png \
  ../_data/images/textures/i0145_0000b.png \
  ../_data/images/textures/i0146_0000b.png \
  ../_data/images/textures/i0147_0000b.png \
  ../_data/images/textures/i0148_0000b.png \
  ../_data/images/textures/i0149_0000b.png \
  ../_data/images/textures/i0150_0000b.png \
  ../_data/images/textures/i0151_0000b.png \
  ../_data/images/textures/i0152_0000b.png \
  ../_data/images/textures/i0153_0000b.png \
  ../_data/images/textures/i0154_0000b.png \
  ../_data/images/textures/i0155_0000b.png \
  ../_data/images/textures/i0156_0000b.png \
  ../_data/images/textures/i0157_0000b.png \
  ../_data/images/textures/i0158_0000b.png \
  ../_data/images/textures/i0159_0000b.png \
  ../_data/images/textures/i0160_0000b.png \
  ../_data/images/textures/i0161_0000b.png \
  ../_data/images/textures/i0162_0000b.png \
  ../_data/images/textures/i0163_0000b.png \
  ../_data/images/textures/i0164_0000b.png \
  ../_data/images/textures/i0165_0000b.png \
  ../_data/images/textures/i0166_0000b.png \
  ../_data/images/textures/i0167_0000b.png \
  ../_data/images/textures/i0168_0000b.png \
  ../_data/images/textures/i0169_0000b.png \
  ../_data/images/textures/i0170_0000b.png \
  ../_data/images/textures/i0171_0000b.png \
  ../_data/images/textures/i0172_0000b.png \
  ../_data/images/textures/i0173_0000b.png \
  ../_data/images/textures/i0174_0000b.png \
  ../_data/images/textures/i0175_0000b.png \
  ../_data/images/textures/i0176_0000b.png \
  ../_data/images/textures/i0177_0000b.png \
  ../_data/images/textures/i0178_0000b.png \
  ../_data/images/textures/i0179_0000b.png \
  ../_data/images/textures/i0180_0000b.png \
  ../_data/images/textures/i0181_0000b.png \
  ../_data/images/textures/i0182_0000b.png \
  ../_data/images/textures/i0183_0000b.png \
  ../_data/images/textures/i0184_0000b.png \
  ../_data/images/textures/i0185_0000b.png \
  ../_data/images/textures/i0186_0000b.png \
  ../_data/images/textures/i0187_0000b.png \
  ../_data/images/textures/i0188_0000b.png \
  ../_data/images/textures/i0189_0000b.png \
  ../_data/images/textures/i0190_0000b.png \
  ../_data/images/textures/i0191_0000b.png \
  ../_data/images/textures/i0192_0000b.png \
  ../_data/images/textures/i0193_0000b.png \
  ../_data/images/textures/i0194_0000b.png \
  ../_data/images/textures/i0195_0000b.png \
  ../_data/images/textures/i0196_0000b.png \
  ../_data/images/textures/i0197_0000b.png \
  ../_data/images/textures/i0198_0000b.png \
  ../_data/images/textures/i0199_0000b.png \
  ../_data/images/textures/i0200_0000b.png \
  ../_data/images/textures/i0201_0000b.png \
  ../_data/images/textures/i0202_0000b.png \
  ../_data/images/textures/i0203_0000b.png \
  ../_data/images/textures/i0204_0000b.png \
  ../_data/images/textures/i0205_0000b.png \
  ../_data/images/textures/i0206_0000b.png \

fonts.files =  \
  ../_data/images/fonts/Font07.png \
  ../_data/images/fonts/Font08.png \
  ../_data/images/fonts/Font09.png \
  ../_data/images/fonts/Font10.png \
  ../_data/images/fonts/Font12.png \
  ../_data/images/fonts/Font14.png \
  ../_data/images/fonts/Font16.png \
  ../_data/images/fonts/Font18.png \
  ../_data/images/fonts/Font20.png \
  ../_data/images/fonts/Font22.png \
  ../_data/images/fonts/Font24.png \
  ../_data/images/fonts/DroidSans.ttf \
  ../_data/images/fonts/ProggyClean.ttf \

shaders.files = \
  ../_data/shaders/BumpNormal.frag \
  ../_data/shaders/BumpNormal.vert \
  ../_data/shaders/BumpNormalEarth.frag \
  ../_data/shaders/BumpNormalParallax.frag \
  ../_data/shaders/Color.frag \
  ../_data/shaders/ColorAttribute.vert \
  ../_data/shaders/ColorUniform.vert \
  ../_data/shaders/ColorUniformPoint.vert \
  ../_data/shaders/Diffuse.frag \
  ../_data/shaders/Diffuse.vert \
  ../_data/shaders/Earth.frag \
  ../_data/shaders/ErrorTex.frag \
  ../_data/shaders/ErrorTex.vert \
  ../_data/shaders/FontTex.frag \
  ../_data/shaders/FontTex.vert \
  ../_data/shaders/PerPixBlinn.frag \
  ../_data/shaders/PerPixBlinn.vert \
  ../_data/shaders/PerPixBlinnSkinned.vert \
  ../_data/shaders/PerVrtBlinnColorAttrib.vert \
  ../_data/shaders/PerPixBlinnTex.frag \
  ../_data/shaders/PerPixBlinnTex.vert \
  ../_data/shaders/PerPixBlinnTexSkinned.vert \
  ../_data/shaders/PerPixCookTorrance.frag \
  ../_data/shaders/PerPixCookTorrance.vert \
  ../_data/shaders/PerPixCookTorranceTex.frag \
  ../_data/shaders/PerPixCookTorranceTex.vert \
  ../_data/shaders/PerVrtBlinn.frag \
  ../_data/shaders/PerVrtBlinn.vert \
  ../_data/shaders/PerVrtBlinnSkinned.vert \
  ../_data/shaders/PerVrtBlinnTex.frag \
  ../_data/shaders/PerVrtBlinnTex.vert \
  ../_data/shaders/PerVrtBlinnTexSkinned.vert \
  ../_data/shaders/Reflect.frag \
  ../_data/shaders/Reflect.vert \
  ../_data/shaders/RefractReflect.frag \
  ../_data/shaders/RefractReflect.vert \
  ../_data/shaders/RefractReflectDisp.frag \
  ../_data/shaders/RefractReflectDisp.vert \
  ../_data/shaders/SceneOculus.frag \
  ../_data/shaders/SceneOculus.vert \
  ../_data/shaders/ShadowMapping.frag \
  ../_data/shaders/ShadowMapping.vert \
  ../_data/shaders/SkyBox.frag \
  ../_data/shaders/SkyBox.vert \
  ../_data/shaders/StereoOculus.frag \
  ../_data/shaders/StereoOculus.vert \
  ../_data/shaders/StereoOculusDistortionMesh.frag \
  ../_data/shaders/StereoOculusDistortionMesh.vert \
  ../_data/shaders/TextureOnly.frag \
  ../_data/shaders/TextureOnly.vert \
  ../_data/shaders/TextureOnly3D.frag \
  ../_data/shaders/TextureOnly3D.vert \
  ../_data/shaders/VolumeRenderingRayCast.frag \
  ../_data/shaders/VolumeRenderingRayCastLighted.frag \
  ../_data/shaders/VolumeRenderingRayCast.vert \
  ../_data/shaders/Wave.frag \
  ../_data/shaders/Wave.vert \
  ../_data/shaders/WaveRefractReflect.vert \

models_3DS_Halloween.files = \
  ../_data/models/3DS/Halloween/GUTS.BMP \
  ../_data/models/3DS/Halloween/JACKTOP.BMP \
  ../_data/models/3DS/Halloween/JACKTO_B.BMP \
  ../_data/models/3DS/Halloween/JACK_B.BMP \
  ../_data/models/3DS/Halloween/JACK_C.BMP \
  ../_data/models/3DS/Halloween/STEM.BMP \
  ../_data/models/3DS/Halloween/STEM_B.BMP \
  ../_data/models/3DS/Halloween/jackolan.3ds \
  ../_data/models/3DS/Halloween/kerze.3DS \
  ../_data/models/3DS/Halloween/stem.3DS \

models_DAE_AstroBoy.files = \
  ../_data/models/DAE/AstroBoy/AstroBoy.dae \
  ../_data/models/DAE/AstroBoy/boy_10.jpg \

models_DAE_Sintel.files = \
  ../_data/models/DAE/Sintel/SintelLowResOwnRig.dae \
  ../_data/models/DAE/Sintel/sintel_diff.png \
  ../_data/models/DAE/Sintel/eyelash.png \
  ../_data/models/DAE/Sintel/sintel_eyeball_diff.png \
  ../_data/models/DAE/Sintel/sintel_hair_solid.jpg \

models_DAE_SkinnedCube.files = \
  ../_data/models/DAE/SkinnedCube/skinnedcube.dae \
  ../_data/models/DAE/SkinnedCube/skinnedcube2.dae \
  ../_data/models/DAE/SkinnedCube/skinnedcube4.dae \
  ../_data/models/DAE/SkinnedCube/skinnedcube5.dae \

models_DAE_Table.files = \
  ../_data/models/DAE/Table/table.dae \
  ../_data/models/DAE/Table/Noyer_france.jpg \

models_DAE_Crate.files = \
  ../_data/models/DAE/Crate/crate.dae \
  ../_data/models/DAE/Crate/Crate1.png \
  ../_data/models/DAE/Crate/Crate2.png \

models_FBX_Duck.files = \
  ../_data/models/FBX/Duck/duck.fbx \
  ../_data/models/FBX/Duck/duckCM.png \

models_FBX_Axes.files = \
  ../_data/models/FBX/Axes/axes_blender.fbx \

models_OBJ_Christoffelturm.files = \
  ../_data/models/Wavefront-OBJ/Christoffelturm/christoffelturm.obj \
  ../_data/models/Wavefront-OBJ/Christoffelturm/christoffelturm.mtl \
  ../_data/models/Wavefront-OBJ/Christoffelturm/texture1.jpg \
  ../_data/models/Wavefront-OBJ/Christoffelturm/texture2.jpg \
  ../_data/models/Wavefront-OBJ/Christoffelturm/shadow.png \

calibrations.files = \
  ../_data/calibrations/calib_in_params.yml \
  ../_data/calibrations/aruco_detector_params.yml \


# Copies the given files to the destination directory
defineTest(copyToDestdir) {
    files = $$1
    for(FILE, files) {
        DDIR = $$DESTDIR
        win32:FILE ~= s,/,\\,g # Replace slashes in paths with backslashes for Windows
        win32:DDIR ~= s,/,\\,g
        QMAKE_POST_LINK += $$QMAKE_COPY $$quote($$FILE) $$quote($$DDIR) $$escape_expand(\\n\\t)
    }
    export(QMAKE_POST_LINK)
}

# OpenCV
win32 {
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_aruco320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_calib3d320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_core320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_features2d320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_highgui320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_flann320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_imgproc320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_imgcodecs320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_objdetect320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_video320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_videoio320.lib)
    copyToDestdir($$PWD\_lib\prebuilt\OpenCV\x64\vc12\bin\opencv_xfeatures2d320.lib)
}
macx {
    cvlibs.files += \
        ../_lib/prebuilt/OpenCV/macx/libopencv_core.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_core.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_core.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgproc.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgproc.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgproc.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_video.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_video.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_video.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_videoio.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_videoio.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_videoio.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgcodecs.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgcodecs.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_imgcodecs.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_calib3d.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_calib3d.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_calib3d.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_aruco.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_aruco.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_aruco.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_features2d.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_features2d.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_features2d.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_xfeatures2d.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_xfeatures2d.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_xfeatures2d.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_objdetect.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_objdetect.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_objdetect.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_flann.3.2.0.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_flann.3.2.dylib \
        ../_lib/prebuilt/OpenCV/macx/libopencv_flann.dylib
}
unix:!macx:!android {
    # linux only
}

# Deployment
macx: {
    textures.path = Contents/_data/images/textures
    fonts.path = Contents/_data/images/fonts
    shaders.path = Contents/_data/shaders
    models_3DS_Halloween.path = Contents/_data/models/3DS/Halloween
    models_DAE_AstroBoy.path = Contents/_data/models/DAE/AstroBoy
    models_DAE_Sintel.path = Contents/_data/models/DAE/Sintel
    models_DAE_SkinnedCube.path = Contents/_data/models/DAE/SkinnedCube
    models_DAE_Table.path = Contents/_data/models/DAE/Table
    models_DAE_Crate.path = Contents/_data/models/DAE/Crate
    models_FBX_Duck.path = Contents/_data/models/FBX/Duck
    models_FBX_Axes.path = Contents/_data/models/FBX/Axes
    models_OBJ_Christoffelturm.path = Contents/_data/models/Wavefront-OBJ/Christoffelturm
    calibrations.path = Contents/_data/calibrations
    cvlibs.path = Contents/Frameworks

    QMAKE_BUNDLE_DATA += textures
    QMAKE_BUNDLE_DATA += fonts
    QMAKE_BUNDLE_DATA += shaders
    QMAKE_BUNDLE_DATA += models_3DS_Halloween
    QMAKE_BUNDLE_DATA += models_DAE_AstroBoy
    QMAKE_BUNDLE_DATA += models_DAE_Sintel
    QMAKE_BUNDLE_DATA += models_DAE_SkinnedCube
    QMAKE_BUNDLE_DATA += models_DAE_Table
    QMAKE_BUNDLE_DATA += models_DAE_Crate
    QMAKE_BUNDLE_DATA += models_FBX_Duck
    QMAKE_BUNDLE_DATA += models_FBX_Axes
    QMAKE_BUNDLE_DATA += models_OBJ_Christoffelturm
    QMAKE_BUNDLE_DATA += calibrations
    QMAKE_BUNDLE_DATA += cvlibs

    macx {ICON = ../lib-SLProject/SLProject-Icon.icns}

    #run macdeployqt
    QMAKE_POST_LINK += macdeployqt ../_bin-$$CONFIGURATION-$$PLATFORM/$${TARGET}.app/
}

message(-----------------------------------------)
message(Target: $$TARGET)
message(Config: $$CONFIGURATION)
message(Platform: $$PLATFORM)
