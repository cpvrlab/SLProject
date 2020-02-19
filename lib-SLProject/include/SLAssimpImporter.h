//#############################################################################
//  File:      SLAssimpImporter.h
//  Author:    Marcus Hudritsch, Marc Wacker
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLASSIMPIMPORTER_H
#define SLASSIMPIMPORTER_H

#include <SLGLTexture.h>
#include <SLImporter.h>

// forward declarations of assimp types
struct aiScene;
struct aiNode;
struct aiMaterial;
struct aiAnimation;
struct aiMesh;

//-----------------------------------------------------------------------------
typedef std::map<int, SLMesh*> SLMeshMap;
//-----------------------------------------------------------------------------
//! Small class interface into the AssImp library for importing 3D assets.
/*! See AssImp library (http://assimp.sourceforge.net/) documentation for 
supported file formats and the import processing options.
*/
class SLAssimpImporter : public SLImporter
{
public:
    SLAssimpImporter() = default;
    explicit SLAssimpImporter(SLLogVerbosity consoleVerb)
      : SLImporter(consoleVerb) {}
    explicit SLAssimpImporter(SLstring&      logFile,
                              SLLogVerbosity logConsoleVerb = LV_normal,
                              SLLogVerbosity logFileVerb    = LV_diagnostic)
      : SLImporter(logFile, logConsoleVerb, logFileVerb) {}

    SLNode* load(SLstring    pathFilename,
                 SLbool      loadMeshesOnly = true,
                 SLMaterial* overrideMat    = nullptr,
                 float       ambientFactor  = 0.0f,
                 SLuint      flags =
                   SLProcess_Triangulate |
                   SLProcess_JoinIdenticalVertices |
                   SLProcess_RemoveRedundantMaterials |
                   SLProcess_FindDegenerates |
                   SLProcess_FindInvalidData |
                   SLProcess_SplitLargeMeshes
                 //|SLProcess_SortByPType
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
    );

protected:
    // intermediate containers
    typedef std::map<SLstring, aiNode*> SLNodeMap;
    typedef std::map<SLstring, SLMat4f> SLJointOffsetMap;
    typedef std::vector<aiNode*>        SLVaiNode;

    SLNodeMap        _nodeMap;        //!< map containing name to aiNode releationships
    SLJointOffsetMap _jointOffsets;   //!< map containing name to joint offset matrices
    aiNode*          _skeletonRoot{}; //!< the common aiNode root for the skeleton of this file

    // SL type containers
    typedef std::vector<SLMesh*> MeshList;

    SLuint   _jointIndex{};  //!< index counter used when iterating over joints
    MeshList _skinnedMeshes; //!< list containing all of the skinned meshes, used to assign the skinned materials

    // loading helper
    aiNode* getNodeByName(const SLstring& name); // return an aiNode ptr if name exists, or null if it doesn't
    SLMat4f getOffsetMat(const SLstring& name);  // return an aiJoint ptr if name exists, or null if it doesn't

    void performInitialScan(const aiScene* scene); // populates nameToNode, nameToJoint, jointGroups, skinnedMeshes,
    void findNodes(aiNode*  node,
                   SLstring padding,
                   SLbool   lastChild);      // scans the assimp scene graph structure and populates nameToNode
    void findJoints(const aiScene* scene); // scans all meshes in the assimp scene and populates nameToJoint and jointGroups
    void findSkeletonRoot();               // finds the common ancestor for each remaining group in jointGroups, these are our final skeleton roots

    void                loadSkeleton(SLJoint* parent, aiNode* node);
    static SLMaterial*  loadMaterial(SLint           index,
                                     aiMaterial*     material,
                                     const SLstring& modelPath,
                                     float           ambientFactor = 0.0f);
    static SLGLTexture* loadTexture(SLstring&     path,
                                    SLTextureType texType);
    SLMesh*             loadMesh(aiMesh* mesh);
    SLNode*             loadNodesRec(SLNode*    curNode,
                                     aiNode*    aiNode,
                                     SLMeshMap& meshes,
                                     SLbool     loadMeshesOnly = true);
    SLAnimation*        loadAnimation(aiAnimation* anim);
    static SLstring     checkFilePath(const SLstring& modelPath, SLstring texFile);
    SLbool              aiNodeHasMesh(aiNode* node);

    // misc helper
    void clear();
};
//-----------------------------------------------------------------------------
#endif // SLASSIMP_H
