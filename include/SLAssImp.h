//#############################################################################
//  File:      SL/SLAssImp.h
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

// @todo    dont make this class static
// @todo    rename SLAssImp to SLImporter
//          make SLImporter a base class and put the implementation of the assimp importer in a derived class
 //         find the best way to make this nice and clean and to provide an interface for others to
//          write their own importers (for custom file formats for example)

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

// forward declarations of assimp types
// @todo    Is it good practice to not include the assimp headers here and just use forward declaration?
//          Do some research on best practices.
struct aiNode;
struct aiMaterial;
struct aiAnimation;
struct aiMesh;
struct aiVectorKey;
struct aiQuatKey;

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
    enum LogVerbosity {
        Quiet = 0,
        Minimal = 1,
        Normal = 2,
        Detailed = 3,
        Diagnostic = 4
    };

protected:

    LogVerbosity _logVerbosity;

    // @todo get the bone information struct out of here
    struct BoneInformation
    {
        SLstring name;
        SLuint id;
        SLMat4f offsetMat;
    };
    

    std::map<SLstring, BoneInformation>   bones; // bone node to bone id mapping

    // old global vars just put in the class for now
    // @todo clean up names, remove unneded, etc.
    
    SLSkeleton* skel = NULL;
    std::map<SLstring, SLNode*>  nameToNodeMapping;   // node name to SLNode instance mapping

    // list of meshes utilising a skeleton
    std::vector<SLMesh*> skinnedMeshes;

    // helper functions
    SLMaterial*   loadMaterial(SLint index, aiMaterial* material, SLstring modelPath);
    SLGLTexture*  loadTexture(SLstring &path, SLTexType texType);
    SLMesh*       loadMesh(aiMesh *mesh);
    SLNode*       loadNodesRec(SLNode *curNode, aiNode *aiNode, SLMeshMap& meshes, SLbool loadMeshesOnly = true);
    void          findAndLoadSkeleton(aiNode* node);
    void          loadSkeleton(aiNode* node);
    SLBone*       loadSkeletonRec(aiNode* node, SLBone* parent);
    SLAnimation*  loadAnimation(aiAnimation* anim);
    SLstring      checkFilePath(SLstring modelPath, SLstring texFile);
    SLbool        aiNodeHasMesh(aiNode* node);
    SLbool        isBone(const SLstring& name);
    BoneInformation* getBoneInformation(const SLstring& name);
    SLNode* findLoadedNodeByName(const SLstring& name);

public:
    SLAssImp(LogVerbosity logVerb = Diagnostic)
        : _logVerbosity(logVerb)
    { }

      SLNode*       load           (SLstring pathFilename,
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
