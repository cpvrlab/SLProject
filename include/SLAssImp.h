//#############################################################################
//  File:      SL/SLAssImp.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://code.google.com/p/slproject/wiki/CodingStyleGuidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLNode.h>
#include <SLGLTexture.h>

#ifndef SLASSIMP_H
#define SLASSIMP_H

// copy of the aiPostProcessStep enum for usage in the wrapper load function
enum slPostProcessSteps
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

//-----------------------------------------------------------------------------
typedef std::map<int, SLMesh*> SLMeshMap;
//-----------------------------------------------------------------------------
//! Small class interface into the AssImp library for importing 3D assets. 
/*! See AssImp library (http://assimp.sourceforge.net/) documentation for 
supported file formats and the import processing options.
*/
class SLAssImp
{  
   public:
      static SLNode*       load           (SLstring pathFilename,
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
                                          );
      static SLstring      defaultPath;
};
//-----------------------------------------------------------------------------
#endif // SLASSIMP_H
