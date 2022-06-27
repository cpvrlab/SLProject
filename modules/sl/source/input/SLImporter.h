//#############################################################################
//  File:      sl/SLImporter.h
//  Authors:   Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLIMPORTER_H
#define SLIMPORTER_H

#include <fstream>
#include <map>

#include <SL.h>
#include <SLAnimation.h>
#include <SLEnums.h>
#include <SLMesh.h>

class SLNode;
class SLMaterial;
class SLAssetManager;
class SLAnimManager;
class SLProgressHandler;
class SLSkybox;

//-----------------------------------------------------------------------------
//! Copy of the aiPostProcessStep enum for usage in the wrapper load function
enum SLPostProcessSteps
{
    SLProcess_CalcTangentSpace         = 0x1,
    SLProcess_JoinIdenticalVertices    = 0x2,
    SLProcess_MakeLeftHanded           = 0x4,
    SLProcess_Triangulate              = 0x8,
    SLProcess_RemoveComponent          = 0x10,
    SLProcess_GenNormals               = 0x20,
    SLProcess_GenSmoothNormals         = 0x40,
    SLProcess_SplitLargeMeshes         = 0x80,
    SLProcess_PreTransformVertices     = 0x100,
    SLProcess_LimitJointWeights        = 0x200,
    SLProcess_ValidateDataStructure    = 0x400,
    SLProcess_ImproveCacheLocality     = 0x800,
    SLProcess_RemoveRedundantMaterials = 0x1000,
    SLProcess_FixInfacingNormals       = 0x2000,
    SLProcess_SortByPType              = 0x8000,
    SLProcess_FindDegenerates          = 0x10000,
    SLProcess_FindInvalidData          = 0x20000,
    SLProcess_GenUVCoords              = 0x40000,
    SLProcess_TransformUVCoords        = 0x80000,
    SLProcess_FindInstances            = 0x100000,
    SLProcess_OptimizeMeshes           = 0x200000,
    SLProcess_OptimizeGraph            = 0x400000,
    SLProcess_FlipUVs                  = 0x800000,
    SLProcess_FlipWindingOrder         = 0x1000000,
    SLProcess_SplitByJointCount        = 0x2000000,
    SLProcess_Dejoint                  = 0x4000000
};

//-----------------------------------------------------------------------------
typedef std::map<int, SLMesh*> SLMeshMap;
//-----------------------------------------------------------------------------
//! Interface for 3D file format importer implementations
class SLImporter
{
public:
    SLImporter();
    explicit SLImporter(SLLogVerbosity consoleVerb);
    explicit SLImporter(const SLstring& logFile,
                        SLLogVerbosity  logConsoleVerb = LV_normal,
                        SLLogVerbosity  logFileVerb    = LV_diagnostic);
    virtual ~SLImporter();

    void logConsoleVerbosity(SLLogVerbosity verb) { _logConsoleVerbosity = verb; }
    void logFileVerbosity(SLLogVerbosity verb) { _logFileVerbosity = verb; }

    virtual SLNode* load(SLAnimManager&     aniMan,
                         SLAssetManager*    assetMgr,
                         SLstring           pathFilename,
                         SLstring           texturePath,
                         SLSkybox*          skybox                 = nullptr,
                         SLbool             deleteTexImgAfterBuild = false,
                         SLbool             loadMeshesOnly         = true,
                         SLMaterial*        overrideMat            = nullptr,
                         float              ambientFactor          = 0.0f,
                         SLbool             forceCookTorranceRM    = false,
                         SLProgressHandler* progressHandler        = nullptr,
                         SLuint             flags =
                           SLProcess_Triangulate |
                           SLProcess_JoinIdenticalVertices |
                           SLProcess_SplitLargeMeshes |
                           SLProcess_RemoveRedundantMaterials |
                           SLProcess_SortByPType |
                           SLProcess_FindDegenerates |
                           SLProcess_FindInvalidData
                         //|SLProcess_OptimizeMeshes
                         //|SLProcess_OptimizeGraph
                         //|SLProcess_CalcTangentSpace
                         //|SLProcess_MakeLeftHanded
                         //|SLProcess_RemoveComponent
                         //|SLProcess_GenNormals
                         //|SLProcess_GenSmoothNormals
                         //|SLProcess_PreTransformVertices
                         //|SLProcess_LimitJointWeights
                         //|SLProcess_ValidateDataStructure
                         //|SLProcess_ImproveCacheLocality
                         //|SLProcess_FixInfacingNormals
                         //|SLProcess_GenUVCoords
                         //|SLProcess_TransformUVCoords
                         //|SLProcess_FindInstances
                         //|SLProcess_FlipUVs
                         //|SLProcess_FlipWindingOrder
                         //|SLProcess_SplitByJointCount
                         //|SLProcess_Dejoint
                         ) = 0;

    SLNode*         rootNode() { return _sceneRoot; }
    SLVMesh&        meshes() { return _meshes; }
    SLAnimSkeleton* skeleton() { return _skeleton; }
    SLVAnimation&   nodeAnimations() { return _nodeAnimations; }

protected:
    std::ofstream  _log;                 //!< log stream
    SLstring       _logFile;             //!< name of the log file
    SLLogVerbosity _logConsoleVerbosity; //!< verbosity level of log output to the console
    SLLogVerbosity _logFileVerbosity;    //!< verbosity level of log output to the file

    // the imported data for easy access after importing it
    SLNode*         _sceneRoot;      //!< the root node of the scene
    SLVMesh         _meshes;         //!< all imported meshes
    SLAnimSkeleton* _skeleton;       //!< the imported skeleton for this file
    SLVAnimation    _nodeAnimations; //!< all imported node animations

    // misc helper
    void logMessage(SLLogVerbosity verbosity, const char* msg, ...);
};
//-----------------------------------------------------------------------------
#endif // SLIMPORTER_H
