##############################################################################
#  File:      assimp.pro
#  Purpose:   QMake project definition file for the asset import library
#  Author:    Marcus Hudritsch
#  Date:      September 2012 (HS12)
#  Copyright: Marcus Hudritsch, Switzerland
#             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
#             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
##############################################################################

TEMPLATE = lib
TARGET = lib-assimp

CONFIG += staticlib
CONFIG -= qt
CONFIG += warn_off

DEFINES += ASSIMP_BUILD_BOOST_WORKAROUND
DEFINES += ASSIMP_BUILD_NO_OWN_ZLIB

DEFINES += ASSIMP_BUILD_NO_MD3_IMPORTER
DEFINES += ASSIMP_BUILD_NO_MDL_IMPORTER
DEFINES += ASSIMP_BUILD_NO_MD2_IMPORTER
DEFINES += ASSIMP_BUILD_NO_ASE_IMPORTER
DEFINES += ASSIMP_BUILD_NO_HMP_IMPORTER
DEFINES += ASSIMP_BUILD_NO_SMD_IMPORTER
DEFINES += ASSIMP_BUILD_NO_MDC_IMPORTER
DEFINES += ASSIMP_BUILD_NO_MD5_IMPORTER
DEFINES += ASSIMP_BUILD_NO_LWO_IMPORTER
DEFINES += ASSIMP_BUILD_NO_NFF_IMPORTER
DEFINES += ASSIMP_BUILD_NO_RAW_IMPORTER
DEFINES += ASSIMP_BUILD_NO_OFF_IMPORTER
DEFINES += ASSIMP_BUILD_NO_AC_IMPORTER
DEFINES += ASSIMP_BUILD_NO_BVH_IMPORTER
DEFINES += ASSIMP_BUILD_NO_IRRMESH_IMPORTER
DEFINES += ASSIMP_BUILD_NO_IRR_IMPORTER
DEFINES += ASSIMP_BUILD_NO_Q3D_IMPORTER
DEFINES += ASSIMP_BUILD_NO_B3D_IMPORTER
DEFINES += ASSIMP_BUILD_NO_TERRAGEN_IMPORTER
DEFINES += ASSIMP_BUILD_NO_CSM_IMPORTER
DEFINES += ASSIMP_BUILD_NO_3D_IMPORTER
DEFINES += ASSIMP_BUILD_NO_LWS_IMPORTER
DEFINES += ASSIMP_BUILD_NO_OGRE_IMPORTER
DEFINES += ASSIMP_BUILD_NO_MS3D_IMPORTER
DEFINES += ASSIMP_BUILD_NO_COB_IMPORTER
DEFINES += ASSIMP_BUILD_NO_Q3BSP_IMPORTER
DEFINES += ASSIMP_BUILD_NO_NDO_IMPORTER
DEFINES += ASSIMP_BUILD_NO_IFC_IMPORTER
DEFINES += ASSIMP_BUILD_NO_XGL_IMPORTER

#define platform variable for folder name
win32 {contains(QMAKE_TARGET.arch, x86_64) {PLATFORM = x64} else {PLATFORM = Win32}}
macx {PLATFORM = macx}
unix:!macx:!android {PLATFORM = linux}
android {PLATFORM = android}

#define configuration variable for folder name
CONFIG(debug, debug|release) {CONFIGURATION = Debug} else {CONFIGURATION = Release}

DESTDIR = ../../_lib/$$CONFIGURATION/$$PLATFORM
OBJECTS_DIR = ../../intermediate/$$TARGET/$$CONFIGURATION/$$PLATFORM

win32 {
    # windows only
    LIBS += -lOpenGL32
    LIBS += -lwinmm
    LIBS += -lgdi32
    LIBS += -luser32
    LIBS += -lkernel32
}
macx {
    CONFIG -= app_bundle
    CONFIG += c++11
    QMAKE_CXXFLAGS += -mmacosx-version-min=10.7
    QMAKE_LFLAGS   += -mmacosx-version-min=10.7
    QMAKE_CXXFLAGS += -stdlib=libc++
    QMAKE_CXXFLAGS += -std=c++11
    QMAKE_CXXFLAGS += -Wno-unused-parameter
    LIBS += -framework Cocoa
    LIBS += -framework IOKit
    LIBS += -framework OpenGL
    LIBS += -framework QuartzCore
    LIBS += -stdlib=libc++
}
unix:!macx:!android {
    # linux only
    LIBS += -lGL
    LIBS += -lX11
    LIBS += -lXrandr    #livrandr-dev
    LIBS += -lXi        #libxi-dev
    LIBS += -lXinerama  #libxinerama-dev
    LIBS += -lXxf86vm   #libxf86vm
    LIBS += -ludev      #libudev-dev
    QMAKE_CXXFLAGS += -Wunused-parameter
    QMAKE_CXXFLAGS += -Wno-unused-parameter

}
android {
    # android only
    QMAKE_CXXFLAGS += -Wunused-parameter
}


INCLUDEPATH += \
   include \
   code \
   code/BoostWorkaround \
   ../zlib

HEADERS += \
   code/3DSHelper.h \
   code/3DSLoader.h \
   code/ACLoader.h \
   code/ASELoader.h \
   code/ASEParser.h \
   code/AssimpPCH.h \
   code/B3DImporter.h \
   code/BaseImporter.h \
   code/BaseProcess.h \
   code/BlenderBMesh.h \
   code/BlenderDNA.h \
   code/BlenderIntermediate.h \
   code/BlenderLoader.h \
   code/BlenderModifier.h \
   code/BlenderScene.h \
   code/BlenderSceneGen.h \
   code/BlenderTessellator.h \
   code/BlobIOSystem.h \
   code/BoostWorkaround/boost/foreach.hpp \
   code/BoostWorkaround/boost/format.hpp \
   code/BoostWorkaround/boost/lexical_cast.hpp \
   code/BoostWorkaround/boost/make_shared.hpp \
   code/BoostWorkaround/boost/math/common_factor_rt.hpp \
   code/BoostWorkaround/boost/pointer_cast.hpp \
   code/BoostWorkaround/boost/scoped_array.hpp \
   code/BoostWorkaround/boost/scoped_ptr.hpp \
   code/BoostWorkaround/boost/shared_array.hpp \
   code/BoostWorkaround/boost/shared_ptr.hpp \
   code/BoostWorkaround/boost/static_assert.hpp \
   code/BoostWorkaround/boost/tuple/tuple.hpp \
   code/BVHLoader.h \
   code/ByteSwap.h \
   code/CalcTangentsProcess.h \
   code/CInterfaceIOWrapper.h \
   code/COBLoader.h \
   code/COBScene.h \
   code/ColladaExporter.h \
   code/ColladaHelper.h \
   code/ColladaLoader.h \
   code/ColladaParser.h \
   code/ComputeUVMappingProcess.h \
   code/ConvertToLHProcess.h \
   code/CSMLoader.h \
   code/DeboneProcess.h \
   code/DefaultIOStream.h \
   code/DefaultIOSystem.h \
   code/DefaultProgressHandler.h \
   code/DXFHelper.h \
   code/DXFLoader.h \
   code/Exceptional.h \
   code/fast_atof.h \
   code/FBXCompileConfig.h \
   code/FBXConverter.h \
   code/FBXDocument.h \
   code/FBXDocumentUtil.h \
   code/FBXImporter.h \
   code/FBXImportSettings.h \
   code/FBXParser.h \
   code/FBXProperties.h \
   code/FBXTokenizer.h \
   code/FBXUtil.h \
   code/FileLogStream.h \
   code/FileSystemFilter.h \
   code/FindDegenerates.h \
   code/FindInstancesProcess.h \
   code/FindInvalidDataProcess.h \
   code/FixNormalsStep.h \
   code/GenericProperty.h \
   code/GenFaceNormalsProcess.h \
   code/GenVertexNormalsProcess.h \
   code/HalfLifeFileData.h \
   code/Hash.h \
   code/HMPFileData.h \
   code/HMPLoader.h \
   code/IFCLoader.h \
   code/IFCReaderGen.h \
   code/IFCUtil.h \
   code/IFF.h \
   code/Importer.h \
   code/ImproveCacheLocality.h \
   code/IRRLoader.h \
   code/IRRMeshLoader.h \
   code/IRRShared.h \
   code/irrXMLWrapper.h \
   code/JoinVerticesProcess.h \
   code/LimitBoneWeightsProcess.h \
   code/LineSplitter.h \
   code/LogAux.h \
   code/LWOAnimation.h \
   code/LWOFileData.h \
   code/LWOLoader.h \
   code/LWSLoader.h \
   code/MakeVerboseFormat.h \
   code/MaterialSystem.h \
   code/MD2FileData.h \
   code/MD2Loader.h \
   code/MD2NormalTable.h \
   code/MD3FileData.h \
   code/MD3Loader.h \
   code/MD5Loader.h \
   code/MD5Parser.h \
   code/MDCFileData.h \
   code/MDCLoader.h \
   code/MDCNormalTable.h \
   code/MDLDefaultColorMap.h \
   code/MDLFileData.h \
   code/MDLLoader.h \
   code/MemoryIOWrapper.h \
   code/MS3DLoader.h \
   code/NDOLoader.h \
   code/NFFLoader.h \
   code/ObjExporter.h \
   code/ObjFileData.h \
   code/ObjFileImporter.h \
   code/ObjFileMtlImporter.h \
   code/ObjFileParser.h \
   code/ObjTools.h \
   code/OFFLoader.h \
   code/OptimizeGraph.h \
   code/OptimizeMeshes.h \
   code/ParsingUtils.h \
   code/PlyExporter.h \
   code/PlyLoader.h \
   code/PlyParser.h \
   code/PolyTools.h \
   code/PretransformVertices.h \
   code/ProcessHelper.h \
   code/Profiler.h \
   code/Q3BSPFileData.h \
   code/Q3BSPFileImporter.h \
   code/Q3BSPFileParser.h \
   code/Q3BSPZipArchive.h \
   code/Q3DLoader.h \
   code/qnan.h \
   code/RawLoader.h \
   code/RemoveComments.h \
   code/RemoveRedundantMaterials.h \
   code/RemoveVCProcess.h \
   code/SceneCombiner.h \
   code/ScenePreprocessor.h \
   code/ScenePrivate.h \
   code/SGSpatialSort.h \
   code/SkeletonMeshBuilder.h \
   code/SMDLoader.h \
   code/SmoothingGroups.h \
   code/SortByPTypeProcess.h \
   code/SpatialSort.h \
   code/SplitByBoneCountProcess.h \
   code/SplitLargeMeshes.h \
   code/StandardShapes.h \
   code/StdOStreamLogStream.h \
   code/STEPFile.h \
   code/STEPFileEncoding.h \
   code/STEPFileReader.h \
   code/STLExporter.h \
   code/STLLoader.h \
   code/StreamReader.h \
   code/StringComparison.h \
   code/Subdivision.h \
   code/TargetAnimation.h \
   code/TerragenLoader.h \
   code/TextureTransform.h \
   code/TinyFormatter.h \
   code/TriangulateProcess.h \
   code/UnrealLoader.h \
   code/ValidateDataStructure.h \
   code/Vertex.h \
   code/VertexTriangleAdjacency.h \
   code/Win32DebugLogStream.h \
   code/XFileHelper.h \
   code/XFileImporter.h \
   code/XFileParser.h \
   code/XGLLoader.h \
   contrib/clipper/clipper.hpp \
   contrib/ConvertUTF/ConvertUTF.h \
   contrib/irrXML/CXMLReaderImpl.h \
   contrib/irrXML/fast_atof.h \
   contrib/irrXML/heapsort.h \
   contrib/irrXML/irrArray.h \
   contrib/irrXML/irrString.h \
   contrib/irrXML/irrTypes.h \
   contrib/irrXML/irrXML.h \
   contrib/poly2tri/poly2tri/common/shapes.h \
   contrib/poly2tri/poly2tri/common/utils.h \
   contrib/poly2tri/poly2tri/poly2tri.h \
   contrib/poly2tri/poly2tri/sweep/advancing_front.h \
   contrib/poly2tri/poly2tri/sweep/cdt.h \
   contrib/poly2tri/poly2tri/sweep/sweep.h \
   contrib/poly2tri/poly2tri/sweep/sweep_context.h \
   include/assimp/ai_assert.h \
   include/assimp/anim.h \
   include/assimp/camera.h \
   include/assimp/cexport.h \
   include/assimp/cfileio.h \
   include/assimp/cimport.h \
   include/assimp/color4.h \
   include/assimp/Compiler/poppack1.h \
   include/assimp/Compiler/pushpack1.h \
   include/assimp/config.h \
   include/assimp/DefaultLogger.hpp \
   include/assimp/defs.h \
   include/assimp/Exporter.hpp \
   include/assimp/Importer.hpp \
   include/assimp/importerdesc.h \
   include/assimp/IOStream.hpp \
   include/assimp/IOSystem.hpp \
   include/assimp/light.h \
   include/assimp/Logger.hpp \
   include/assimp/LogStream.hpp \
   include/assimp/material.h \
   include/assimp/matrix3x3.h \
   include/assimp/matrix4x4.h \
   include/assimp/mesh.h \
   include/assimp/NullLogger.hpp \
   include/assimp/postprocess.h \
   include/assimp/ProgressHandler.hpp \
   include/assimp/quaternion.h \
   include/assimp/scene.h \
   include/assimp/texture.h \
   include/assimp/types.h \
   include/assimp/vector2.h \
   include/assimp/vector3.h \
   include/assimp/version.h

SOURCES += \
   code/3DSConverter.cpp \
   code/3DSLoader.cpp \
   code/ACLoader.cpp \
   code/ASELoader.cpp \
   code/ASEParser.cpp \
   code/Assimp.cpp \
   code/AssimpCExport.cpp \
   code/AssimpPCH.cpp \
   code/B3DImporter.cpp \
   code/BaseImporter.cpp \
   code/BaseProcess.cpp \
   code/BlenderBMesh.cpp \
   code/BlenderDNA.cpp \
   code/BlenderLoader.cpp \
   code/BlenderModifier.cpp \
   code/BlenderScene.cpp \
   code/BlenderTessellator.cpp \
   code/BVHLoader.cpp \
   code/CalcTangentsProcess.cpp \
   code/COBLoader.cpp \
   code/ColladaExporter.cpp \
   code/ColladaLoader.cpp \
   code/ColladaParser.cpp \
   code/ComputeUVMappingProcess.cpp \
   code/ConvertToLHProcess.cpp \
   code/CSMLoader.cpp \
   code/DeboneProcess.cpp \
   code/DefaultIOStream.cpp \
   code/DefaultIOSystem.cpp \
   code/DefaultLogger.cpp \
   code/DXFLoader.cpp \
   code/Exporter.cpp \
   code/FBXAnimation.cpp \
   code/FBXBinaryTokenizer.cpp \
   code/FBXConverter.cpp \
   code/FBXDeformer.cpp \
   code/FBXDocument.cpp \
   code/FBXDocumentUtil.cpp \
   code/FBXImporter.cpp \
   code/FBXMaterial.cpp \
   code/FBXMeshGeometry.cpp \
   code/FBXModel.cpp \
   code/FBXNodeAttribute.cpp \
   code/FBXParser.cpp \
   code/FBXProperties.cpp \
   code/FBXTokenizer.cpp \
   code/FBXUtil.cpp \
   code/FindDegenerates.cpp \
   code/FindInstancesProcess.cpp \
   code/FindInvalidDataProcess.cpp \
   code/FixNormalsStep.cpp \
   code/GenFaceNormalsProcess.cpp \
   code/GenVertexNormalsProcess.cpp \
   code/HMPLoader.cpp \
   code/IFCBoolean.cpp \
   code/IFCCurve.cpp \
   code/IFCGeometry.cpp \
   code/IFCLoader.cpp \
   code/IFCMaterial.cpp \
   code/IFCOpenings.cpp \
   code/IFCProfile.cpp \
   code/IFCReaderGen.cpp \
   code/IFCUtil.cpp \
   code/Importer.cpp \
   code/ImporterRegistry.cpp \
   code/ImproveCacheLocality.cpp \
   code/IRRLoader.cpp \
   code/IRRMeshLoader.cpp \
   code/IRRShared.cpp \
   code/JoinVerticesProcess.cpp \
   code/LimitBoneWeightsProcess.cpp \
   code/LWOAnimation.cpp \
   code/LWOBLoader.cpp \
   code/LWOLoader.cpp \
   code/LWOMaterial.cpp \
   code/LWSLoader.cpp \
   code/MakeVerboseFormat.cpp \
   code/MaterialSystem.cpp \
   code/MD2Loader.cpp \
   code/MD3Loader.cpp \
   code/MD5Loader.cpp \
   code/MD5Parser.cpp \
   code/MDCLoader.cpp \
   code/MDLLoader.cpp \
   code/MDLMaterialLoader.cpp \
   code/MS3DLoader.cpp \
   code/NDOLoader.cpp \
   code/NFFLoader.cpp \
   code/ObjExporter.cpp \
   code/ObjFileImporter.cpp \
   code/ObjFileMtlImporter.cpp \
   code/ObjFileParser.cpp \
   code/OFFLoader.cpp \
   code/OptimizeGraph.cpp \
   code/OptimizeMeshes.cpp \
   code/PlyExporter.cpp \
   code/PlyLoader.cpp \
   code/PlyParser.cpp \
   code/PostStepRegistry.cpp \
   code/PretransformVertices.cpp \
   code/ProcessHelper.cpp \
   code/Q3BSPFileImporter.cpp \
   code/Q3BSPFileParser.cpp \
   code/Q3BSPZipArchive.cpp \
   code/Q3DLoader.cpp \
   code/RawLoader.cpp \
   code/RemoveComments.cpp \
   code/RemoveRedundantMaterials.cpp \
   code/RemoveVCProcess.cpp \
   code/SceneCombiner.cpp \
   code/ScenePreprocessor.cpp \
   code/SGSpatialSort.cpp \
   code/SkeletonMeshBuilder.cpp \
   code/SMDLoader.cpp \
   code/SortByPTypeProcess.cpp \
   code/SpatialSort.cpp \
   code/SplitByBoneCountProcess.cpp \
   code/SplitLargeMeshes.cpp \
   code/StandardShapes.cpp \
   code/STEPFileEncoding.cpp \
   code/STEPFileReader.cpp \
   code/STLExporter.cpp \
   code/STLLoader.cpp \
   code/Subdivision.cpp \
   code/TargetAnimation.cpp \
   code/TerragenLoader.cpp \
   code/TextureTransform.cpp \
   code/TriangulateProcess.cpp \
   code/UnrealLoader.cpp \
   code/ValidateDataStructure.cpp \
   code/VertexTriangleAdjacency.cpp \
   code/XFileImporter.cpp \
   code/XFileParser.cpp \
   code/XGLLoader.cpp \
   contrib/clipper/clipper.cpp \
   contrib/ConvertUTF/ConvertUTF.c \
   contrib/irrXML/irrXML.cpp \
   contrib/poly2tri/poly2tri/common/shapes.cc \
   contrib/poly2tri/poly2tri/sweep/advancing_front.cc \
   contrib/poly2tri/poly2tri/sweep/cdt.cc \
   contrib/poly2tri/poly2tri/sweep/sweep.cc \
   contrib/poly2tri/poly2tri/sweep/sweep_context.cc


