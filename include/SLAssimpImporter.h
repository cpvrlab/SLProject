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
#include <SLImporter.h>
#include <SLNode.h>
#include <SLGLTexture.h>
#include <SLSkeleton.h>

// @todo    dont make this class static
// @todo    rename SLAssImp to SLImporter
//          make SLImporter a base class and put the implementation of the assimp importer in a derived class
 //         find the best way to make this nice and clean and to provide an interface for others to
//          write their own importers (for custom file formats for example)

#ifndef SLASSIMPIMPORTER_H
#define SLASSIMPIMPORTER_H

// forward declarations of assimp types
// @todo    Is it good practice to not include the assimp headers here and just use forward declaration?
//          Do some research on best practices.
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
protected:
    // intermediate containers
    typedef std::map<SLstring, aiNode*> NodeMap;
    typedef std::map<SLstring, SLMat4f> JointOffsetMap;
    typedef std::vector<aiNode*>        NodeList;

    NodeMap		    _nodeMap;           //!< map containing name to aiNode releationships
    JointOffsetMap	_jointOffsets;    //!< map containing name to joint offset matrices
    aiNode*         _skeletonRoot;      //!< the common aiNode root for the skeleton of this file

    // SL type containers
    typedef std::vector<SLMesh*>        MeshList;

    SLuint      _jointIndex;         //!< index counter used when iterating over joints
    MeshList	_skinnedMeshes;     //!< list containing all of the skinned meshes, used to assign the skinned materials


    // loading helper
    aiNode*         getNodeByName(const SLstring& name);    // return an aiNode ptr if name exists, or null if it doesn't
	const SLMat4f   getOffsetMat(const SLstring& name);    // return an aiJoint ptr if name exists, or null if it doesn't

    void            performInitialScan(const aiScene* scene);     // populates nameToNode, nameToJoint, jointGroups, skinnedMeshes,
    void            findNodes(aiNode* node, SLstring padding, SLbool lastChild);           // scans the assimp scene graph structure and populates nameToNode
    void            findJoints(const aiScene* scene);           // scans all meshes in the assimp scene and populates nameToJoint and jointGroups
    void            findSkeletonRoot();                    // finds the common ancestor for each remaining group in jointGroups, these are our final skeleton roots
    
    void            loadSkeleton(SLJoint* parent, aiNode* node);

    SLMaterial*     loadMaterial(SLint index, aiMaterial* material, SLstring modelPath);
    SLGLTexture*    loadTexture(SLstring &path, SLTexType texType);
    SLMesh*         loadMesh(aiMesh *mesh);
    // @todo    go over the loadNodesRec again (rename to loadSceneNodes for clarity) and improve it
    //          add functionality to load lights etc, make it cleaner overall
    //          add log output
    SLNode*         loadNodesRec(SLNode *curNode, aiNode *aiNode, SLMeshMap& meshes, SLbool loadMeshesOnly = true);
    SLAnimation*    loadAnimation(aiAnimation* anim);
    SLstring        checkFilePath(SLstring modelPath, SLstring texFile);
    SLbool          aiNodeHasMesh(aiNode* node);



    // misc helper
    void clear();

public:
    SLAssimpImporter()
    { }
    SLAssimpImporter(SLLogVerbosity consoleVerb)
         : SLImporter(consoleVerb)
    { }
    SLAssimpImporter(const SLstring& logFile, SLLogVerbosity logConsoleVerb = LV_Normal, SLLogVerbosity logFileVerb = LV_Diagnostic)
        : SLImporter(logFile, logConsoleVerb, logFileVerb)
    { }

    SLNode* load            (SLstring pathFilename,
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
};
//-----------------------------------------------------------------------------
#endif // SLASSIMP_H
