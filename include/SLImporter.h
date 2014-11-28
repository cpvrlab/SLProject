//#############################################################################
//  File:      SL/SLImporter.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLNode.h>
#include <SLGLTexture.h>
#include <SLSkeleton.h>


#ifndef SLIMPORTER_H
#define SLIMPORTER_H

// copy of the aiPostProcessStep enum for usage in the wrapper load function
enum SLPostProcessSteps
{
    SLProcess_CalcTangentSpace = 0x1,
    SLProcess_JoinIdenticalVertices = 0x2,
    SLProcess_MakeLeftHanded = 0x4,
    SLProcess_Triangulate = 0x8,
    SLProcess_RemoveComponent = 0x10,
    SLProcess_GenNormals = 0x20,
    SLProcess_GenSmoothNormals = 0x40,
    SLProcess_SplitLargeMeshes = 0x80,
    SLProcess_PreTransformVertices = 0x100,
    SLProcess_LimitBoneWeights = 0x200,
    SLProcess_ValidateDataStructure = 0x400,
    SLProcess_ImproveCacheLocality = 0x800,
    SLProcess_RemoveRedundantMaterials = 0x1000,
    SLProcess_FixInfacingNormals = 0x2000,
    SLProcess_SortByPType = 0x8000,
    SLProcess_FindDegenerates = 0x10000,
    SLProcess_FindInvalidData = 0x20000,
    SLProcess_GenUVCoords = 0x40000,
    SLProcess_TransformUVCoords = 0x80000,
    SLProcess_FindInstances = 0x100000,
    SLProcess_OptimizeMeshes = 0x200000,
    SLProcess_OptimizeGraph = 0x400000,
    SLProcess_FlipUVs = 0x800000,
    SLProcess_FlipWindingOrder = 0x1000000,
    SLProcess_SplitByBoneCount = 0x2000000,
    SLProcess_Debone = 0x4000000
};

// @todo build a dedicated log class that defines this verbosity levels
enum SLLogVerbosity {
    LV_Quiet = 0,
    LV_Minimal = 1,
    LV_Normal = 2,
    LV_Detailed = 3,
    LV_Diagnostic = 4
};

//-----------------------------------------------------------------------------
typedef std::map<int, SLMesh*> SLMeshMap;
//-----------------------------------------------------------------------------
//! Interface for importer implementations
/*
*/
class SLImporter
{  
public:

protected:
    ofstream        _log;                   //!< log stream
    SLstring        _logFile;               //!< name of the log file
    SLLogVerbosity  _logConsoleVerbosity;   //!< verbosity level of log output to the console
    SLLogVerbosity  _logFileVerbosity;      //!< verbosity level of log output to the file

    // misc helper
    void logMessage(SLLogVerbosity verbosity, const char* msg, ...);

public:
    SLImporter();
    SLImporter(SLLogVerbosity consoleVerb);
    SLImporter(const SLstring& logFile, SLLogVerbosity logConsoleVerb = LV_Normal, SLLogVerbosity logFileVerb = LV_Diagnostic);
    ~SLImporter();
    
    void logConsoleVerbosity(SLLogVerbosity verb) { _logConsoleVerbosity = verb; }
    void logFileVerbosity(SLLogVerbosity verb) { _logFileVerbosity = verb; }

    virtual SLNode* load    (SLstring pathFilename,
                            SLbool loadMeshesOnly = true,
                            SLuint flags = 
                                SLProcess_Triangulate
                            |SLProcess_JoinIdenticalVertices
                            |SLProcess_SplitLargeMeshes
                            |SLProcess_RemoveRedundantMaterials
                            |SLProcess_SortByPType
                            |SLProcess_FindDegenerates
                            |SLProcess_FindInvalidData
                            //|SLProcess_OptimizeMeshes
                            //|SLProcess_OptimizeGraph
                            //|SLProcess_CalcTangentSpace
                            //|SLProcess_MakeLeftHanded
                            //|SLProcess_RemoveComponent
                            //|SLProcess_GenNormals
                            //|SLProcess_GenSmoothNormals
                            //|SLProcess_PreTransformVertices
                            //|SLProcess_LimitBoneWeights
                            //|SLProcess_ValidateDataStructure
                            //|SLProcess_ImproveCacheLocality
                            //|SLProcess_FixInfacingNormals
                            //|SLProcess_GenUVCoords
                            //|SLProcess_TransformUVCoords
                            //|SLProcess_FindInstances
                            //|SLProcess_FlipUVs
                            //|SLProcess_FlipWindingOrder
                            //|SLProcess_SplitByBoneCount
                            //|SLProcess_Debone
                            ) = 0;

      static SLstring      defaultPath;
};
//-----------------------------------------------------------------------------
#endif // SLASSIMP_H
