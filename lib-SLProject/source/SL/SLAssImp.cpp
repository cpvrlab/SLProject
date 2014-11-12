//#############################################################################
//  File:      SL/SLAssImp.cpp
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: 2002-2014 Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

/*

    1. load materials
    2. load textures
    3. load meshes
        > Put meshes that have bones in a list
        > Put bones in a list (including vertex binding lists)
            > but dont bind to mesh yet
    4. load scene graph and skeletons
        > if a node is a bone then load it as a skeleton root
        > add the skeleton instance to the individual bone info
        > determine boneId's
    5. go over all meshes in the 'hasBone' list again
        > add the appropriate skeleton to the mesh (based on the first skeleton in one of its bones)
        > add bone weights, bone ids etc to the meshes as vertex data
    6. load animations
        > seperate node animations from skeletal animations

*/

// @todo findAndLoadSkeleton, loadSkeleton and loadSkeleton rec can easily be put into a single function, do that.
// @todo add aiMat to SLMat helper function

#include <stdafx.h>
//Don't use the memory leak detector in Assimp

#include <SLAssImp.h>
#include <SLScene.h>
#include <SLGLTexture.h>
#include <SLMaterial.h>
#include <SLSkeleton.h>
#include <SLAnimation.h>
#include <SLGLShaderProg.h>

// assimp is only included in the source file to not expose it to the rest of the framework
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

/*

loadAnimation       // loads animations and builds a map of bone nodes
loadNodesRec        // loads the node structure and loads marked bone nodes into skeleton containers

// we need to assign skeletons to meshes somewhere
// also, for now a skeleton holds the animation instances, later we need entities or something to hold them

*/


// helper functions to load an assimp scene
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

// static vars to keep track of loaded resources
SLSkeleton* skel = NULL;

std::map<SLstring, SLNode*>  nameToNodeMapping;   // node name to SLNode instance mapping

// list of meshes utilising a skeleton
std::vector<SLMesh*> skinnedMeshes; // todo change to std::set

/* List to keep track of bone nodes including the bones id. 
    Nodes in this list won't be included in the scenegraph
    but will be added to an SLSkeleton instance.
*/

struct BoneInformation
{
    SLstring name;
    SLuint id;
    SLMat4f offsetMat;
};

std::map<SLstring, BoneInformation>   bones; // bone node to bone id mapping

//-----------------------------------------------------------------------------
//! Default path for 3DS models used when only filename is passed in load.
SLstring SLAssImp::defaultPath = "../_data/models/";
//-----------------------------------------------------------------------------
/*! Loads the scene from a file and creates materials with textures, the 
meshes and the nodes for the scene graph. Materials, textures and meshes are
added to the according vectors of SLScene for later deallocation.
*/
SLNode* SLAssImp::load(SLstring file,        //!< File with path or on default path 
                       SLbool loadMeshesOnly,//!< Only load nodes with meshes
                       SLuint flags)         //!< Import flags (see assimp/postprocess.h)
{
    // Check existance
    if (!SLFileSystem::fileExists(file))
    {   file = defaultPath + file;
        if (!SLFileSystem::fileExists(file))
        {   SLstring msg = "SLAssImp: File not found: " + file + "\n";
            SL_WARN_MSG(msg.c_str());
            return NULL;
        }
    }

    // Import file with assimp importer
    Assimp::Importer ai;
    const aiScene* scene = ai.ReadFile(file.c_str(), (SLuint)flags);
    if (!scene)
    {   SLstring msg = "Failed to load file: " + 
                        file + "\n" + ai.GetErrorString();
        SL_WARN_MSG(msg.c_str());
        return NULL;
    }

    // load materials
    SLstring modelPath = SLUtils::getPath(file);
    SLVMaterial materials;
    for(SLint i = 0; i < (SLint)scene->mNumMaterials; i++)
        materials.push_back(loadMaterial(i, scene->mMaterials[i], modelPath));

    // load meshes & set their material
    SLVMesh meshes;                  // vector of all loaded meshes
    std::map<int, SLMesh*> meshMap;  // map from the ai index to our mesh
    for(SLint i = 0; i < (SLint)scene->mNumMeshes; i++)
    {   SLMesh* mesh = loadMesh(scene->mMeshes[i]);
        if (mesh != 0)
        {   mesh->mat = materials[scene->mMeshes[i]->mMaterialIndex];
            meshes.push_back(mesh);
            meshMap[i] = mesh;
        }
    } 
    
    // add skinned material to the meshes with bone animations
    // @todo: can we do this wihtout the need to put this in SLMaterial?
    if (skinnedMeshes.size() > 0) {

        SLGLShaderProgGeneric* skinningShader = new SLGLShaderProgGeneric("PerVrtBlinnSkinned.vert","PerVrtBlinn.frag");
        for (SLint i = 0; i < skinnedMeshes.size(); i++)
        {
            SLMesh* mesh = skinnedMeshes[i];
            mesh->mat->shaderProg(skinningShader);
        }
    }


    // load the scene nodes recursively
    SLNode* root = loadNodesRec(NULL, scene->mRootNode, meshMap, loadMeshesOnly);

    // load animations
    vector<SLAnimation*> animations;
    for (SLint i = 0; i < (SLint)scene->mNumAnimations; i++)
        animations.push_back(loadAnimation(scene->mAnimations[i]));

    // test set the animation on the skeleton
    animations[0]->apply(skel, 3.0f);

    return root;
}

//-----------------------------------------------------------------------------
/*!
SLAssImp::loadMaterial loads the AssImp material an returns the SLMaterial.
The materials and textures are added to the SLScene material and texture 
vectors.
*/
SLMaterial* loadMaterial(SLint index, 
                         aiMaterial *material, 
                         SLstring modelPath)
{
    // Get the materials name
    aiString matName;
    material->Get(AI_MATKEY_NAME, matName);
    SLstring name = matName.data;
    if (name.empty()) name = "Import Material";
   
    // Create SLMaterial instance. It is also added to the SLScene::_materials vector
    SLMaterial* mat = new SLMaterial(name.c_str());

    // set the texture types to import into our material
    const SLint		textureCount = 4;
    aiTextureType	textureTypes[textureCount];
    textureTypes[0] = aiTextureType_DIFFUSE;
    textureTypes[1] = aiTextureType_NORMALS;
    textureTypes[2] = aiTextureType_SPECULAR;
    textureTypes[3] = aiTextureType_HEIGHT;
   
    // load all the textures for this material and add it to the material vector
    for(SLint i = 0; i < textureCount; ++i) 
    {   if(material->GetTextureCount(textureTypes[i]) > 0) 
        {   aiString aipath;
            material->GetTexture(textureTypes[i], 0, &aipath, NULL, NULL, NULL, NULL, NULL);
            SLTexType texType = textureTypes[i]==aiTextureType_DIFFUSE  ? ColorMap :
                                textureTypes[i]==aiTextureType_NORMALS  ? NormalMap :
                                textureTypes[i]==aiTextureType_SPECULAR ? GlossMap :
                                textureTypes[i]==aiTextureType_HEIGHT   ? HeightMap : 
                                UnknownMap;
            SLstring texFile = checkFilePath(modelPath, aipath.data);
            SLGLTexture* tex = loadTexture(texFile, texType);
            mat->textures().push_back(tex);
        }
    }
   
    // get color data
    aiColor3D ambient, diffuse, specular, emissive;
    SLfloat shininess, refracti, reflectivity, opacity;
    material->Get(AI_MATKEY_COLOR_AMBIENT, ambient);
    material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    material->Get(AI_MATKEY_COLOR_SPECULAR, specular);
    material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
    material->Get(AI_MATKEY_SHININESS, shininess);
    material->Get(AI_MATKEY_REFRACTI, refracti);
    material->Get(AI_MATKEY_REFLECTIVITY, reflectivity);
    material->Get(AI_MATKEY_OPACITY, opacity);

    // increase shininess if specular color is not low.
    // The material will otherwise be to bright
    if (specular.r > 0.5f &&
        specular.g > 0.5f &&
        specular.b > 0.5f &&
        shininess < 0.01f)
        shininess = 10.0f;

    // set color data
    mat->ambient(SLCol4f(ambient.r, ambient.g, ambient.b));
    mat->diffuse(SLCol4f(diffuse.r, diffuse.g, diffuse.b));
    mat->specular(SLCol4f(specular.r, specular.g, specular.b));
    mat->emission(SLCol4f(emissive.r, emissive.g, emissive.b));
    mat->shininess(shininess);
    //mat->kr(reflectivity);
    //mat->kt(1.0f-opacity);
    //mat->kn(refracti);

    return mat;
}
//-----------------------------------------------------------------------------
/*!
SLAssImp::loadTexture loads the AssImp texture an returns the SLGLTexture
*/
SLGLTexture* loadTexture(SLstring& textureFile, SLTexType texType)
{
    SLVGLTexture& sceneTex = SLScene::current->textures();

    // return if a texture with the same file allready exists
    SLbool exists = false;
    for (SLint i=0; i<sceneTex.size(); ++i)
        if (sceneTex[i]->name()==textureFile) 
            return sceneTex[i];

    // Create the new texture. It is also push back to SLScene::_textures
    SLGLTexture* texture = new SLGLTexture(textureFile,
                                           GL_LINEAR_MIPMAP_LINEAR,
                                           GL_LINEAR,
                                           texType);
    return texture;
}
SLbool isBone(const SLstring& name)
{
    if (bones.find(name) == bones.end())
        return false;

    return true;
}

BoneInformation* getBoneInformation(const SLstring& name)
{
    // create bone if its not already in the map
    if (bones.find(name) == bones.end())
    {
        BoneInformation newBone;
        newBone.name = name;
        newBone.id = bones.size();

        bones[name] = newBone;
    }

    return &bones[name];
}

//-----------------------------------------------------------------------------
/*!
SLAssImp::loadMesh creates a new SLMesh an copies the meshs vertex data and
triangle face indexes. Normals & tangents are not loaded. They are calculated
in SLMesh.
*/
SLMesh* loadMesh(aiMesh *mesh)
{
    // Count first the NO. of triangles in the mesh
    SLuint numTriangles = 0;
    for(unsigned int i = 0; i <  mesh->mNumFaces; ++i)
        if(mesh->mFaces[i].mNumIndices == 3)
            numTriangles++;

    // We only load meshes that contain triangles
    if (numTriangles==0 || mesh->mNumVertices==0)
        return NULL; 

    // create a new mesh. 
    // The mesh pointer is added automatically to the SLScene::meshes vector.
    SLstring name = mesh->mName.data;
    SLMesh *m = new SLMesh(name.empty() ? "Imported Mesh" : name);

    // create position & normal array
    m->numV = mesh->mNumVertices;
    m->P = new SLVec3f[m->numV];
    m->N = new SLVec3f[m->numV];

    // create texCoord array if needed
    if (mesh->HasTextureCoords(0))
        m->Tc = new SLVec2f[m->numV];

    // copy vertex positions & texCoord
    for(SLuint i = 0; i < m->numV; ++i)
    {   m->P[i].set(mesh->mVertices[i].x, 
        mesh->mVertices[i].y, 
        mesh->mVertices[i].z);
        if (m->Tc)
        m->Tc[i].set(mesh->mTextureCoords[0][i].x,
        mesh->mTextureCoords[0][i].y);
    }

    // create face index array
    m->numI = mesh->mNumFaces * 3;
    if (m->numV < 65536)
    {   m->I16 = new SLushort[m->numI];

        // load face triangle indexes only
        SLuint j = 0;
        for(SLuint i = 0; i <  mesh->mNumFaces; ++i)
        {   if(mesh->mFaces[i].mNumIndices == 3)
            {   m->I16[j++] = mesh->mFaces[i].mIndices[0];
                m->I16[j++] = mesh->mFaces[i].mIndices[1];
                m->I16[j++] = mesh->mFaces[i].mIndices[2];
            }
        }
    } else 
    {   m->I32 = new SLuint[m->numI];

        // load face triangle indexes only
        SLuint j = 0;
        for(SLuint i = 0; i <  mesh->mNumFaces; ++i)
        {  if(mesh->mFaces[i].mNumIndices == 3)
            {   m->I32[j++] = mesh->mFaces[i].mIndices[0];
                m->I32[j++] = mesh->mFaces[i].mIndices[1];
                m->I32[j++] = mesh->mFaces[i].mIndices[2];
            }
        }
    }

    m->calcNormals();

    // load bones
    if (mesh->HasBones())
    {
        skinnedMeshes.push_back(m);

        m->Bi = new SLVec4f[m->numV];
        m->Bw = new SLVec4f[m->numV];
        
        // make sure to initialize the weights with 0 vectors
        std::fill_n(m->Bi, m->numV, SLVec4f(0, 0, 0, 0));
        std::fill_n(m->Bw, m->numV, SLVec4f(0, 0, 0, 0));

        for (SLint i = 0; i < mesh->mNumBones; i++)
        {
            aiBone* bone = mesh->mBones[i];
            // get the bone information for this bone to add the vertex weights to its list
            BoneInformation* boneInfo = getBoneInformation(bone->mName.C_Str());
            
            // update the bones offset matrix
            memcpy(&boneInfo->offsetMat, &bone->mOffsetMatrix, sizeof(bone->mOffsetMatrix));
            boneInfo->offsetMat.transpose();


            for (SLint j = 0; j < bone->mNumWeights; j++)
            {
                // add the weight
                SLuint vertId = bone->mWeights[j].mVertexId;
                SLfloat weight = bone->mWeights[j].mWeight;

                m->addWeight(vertId, boneInfo->id, weight);
            }

        }

    }
    cout << "imported" << m->name() << "\n";
    cout << "weights: \n";

    for (SLint i = 0; i < m->numV; i++) {
        cout << "   "<< i << "-> (";
            
        if(m->Bw[i].x > 0.0f) cout << m->Bi[i].x << ": " << m->Bw[i].x << ";";
        if(m->Bw[i].y > 0.0f) cout << m->Bi[i].y << ": " << m->Bw[i].y << ";";
        if(m->Bw[i].z > 0.0f) cout << m->Bi[i].z << ": " << m->Bw[i].z << ";";
        if(m->Bw[i].w > 0.0f) cout << m->Bi[i].w << ": " << m->Bw[i].w << ";";
        
        cout << ")\n";
    }

    return m;
}
//-----------------------------------------------------------------------------
/*!
SLAssImp::loadNodesRec loads the scene graph node tree recursively.
*/
SLNode* loadNodesRec(
   SLNode *curNode,     //!< Pointer to the current node. Pass NULL for root node
   aiNode *node,        //!< The according assimp node. Pass NULL for root node
   SLMeshMap& meshes,   //!< Reference to the meshes vector
   SLbool loadMeshesOnly) //!< Only load nodes with meshes
{
    // we're at the root
    if(!curNode) 
        curNode = new SLNode(node->mName.data);
    
    // load local transform
   aiMatrix4x4* M = &node->mTransformation;
   SLMat4f SLM(M->a1, M->a2, M->a3, M->a4,
                      M->b1, M->b2, M->b3, M->b4,
                      M->c1, M->c2, M->c3, M->c4,
                      M->d1, M->d2, M->d3, M->d4);

   curNode->om(SLM);

    // add the meshes
    for (SLuint i=0; i < node->mNumMeshes; ++i)
    {
        // Only add meshes that were added to the meshMap (triangle meshes)
        if (meshes.count(node->mMeshes[i]))
            curNode->addMesh(meshes[node->mMeshes[i]]);
    }

    // load children recursively
    for(SLuint i = 0; i < node->mNumChildren; i++) 
    {  
        // only load children nodes with meshes or children
        if (!loadMeshesOnly || aiNodeHasMesh(node->mChildren[i]))
        {   SLNode *child = new SLNode(node->mChildren[i]->mName.data);
            curNode->addChild(child);
            loadNodesRec(child, node->mChildren[i], meshes);
        }
        // we found a branch without meshes attached to it, try and find the skeleton root
        else if (skinnedMeshes.size() > 0)
        {
            findAndLoadSkeleton(node->mChildren[i]);
        }
    }

    return curNode;
}
void findAndLoadSkeleton(aiNode* node)
{
    // load skeleton if child is a bone
    if (isBone(node->mName.C_Str()))
    {
        loadSkeleton(node);
    }
    else
    {
        for(SLuint i = 0; i < node->mNumChildren; i++) 
            findAndLoadSkeleton(node->mChildren[i]);
    }
}
SLNode* findLoadedNodeByName(const SLstring& name)
{
    if (nameToNodeMapping.find(name) != nameToNodeMapping.end())
        return nameToNodeMapping[name];

    return NULL;
}

void loadSkeleton(aiNode* node)
{
    // Don't allow files that contain more than one skeleton 
    // @todo an assert here isn't right. The app shouldn't crash just because the user is trying to load an illegal file. But we lack an exception structure atm.
    static SLint count = 0;
    assert(count == 0 && "The specified file contains more than one skeleton, the importer currently only supports one skeleton per file");
    count++;

    // new root skeleton node found, create a skeleton
    SLSkeleton* skeleton = new SLSkeleton;
    skel = skeleton;
    
    // @todo add the skeleton to the scene

    // set the skeleton reference for the skinned meshes
    for (SLint i = 0; i < skinnedMeshes.size(); i++)
    {
        skinnedMeshes.at(i)->skeleton(skeleton);
    }

    // load the bones
    SLBone* rootBone = loadSkeletonRec(node, NULL);
    skeleton->root(rootBone);
}
SLBone* loadSkeletonRec(aiNode* node, SLBone* parent)
{
    // find the bone information for this bone
    BoneInformation* boneInfo = getBoneInformation(node->mName.C_Str());

    assert(boneInfo && "Importer Error: a node previously not marked as a bone was loaded as a bone.");

    SLBone* bone;

    if (parent == NULL)
        bone = skel->createBone(boneInfo->id);
    else
        bone = parent->createChild(boneInfo->id);

    bone->name(boneInfo->name);
    bone->offsetMat(boneInfo->offsetMat);

    // set binding pose for the bone
    SLMat4f om;
    memcpy(&om, &node->mTransformation, sizeof(SLMat4f));
    om.transpose();
    bone->om(om);
    bone->setInitialState();
    // start building the node tree for the skeleton
    for (SLint i = 0; i < node->mNumChildren; i++)
    {
        SLBone* child = loadSkeletonRec(node->mChildren[i], bone);
    }
        
    return bone;
}
//-----------------------------------------------------------------------------
/*!
SLAssImp::loadAnimation loads the scene graph node tree recursively.
*/
// @todo how do we handle multiple skeletons in one file?
SLAnimation* loadAnimation(aiAnimation* anim)
{
    SLAnimation* result = new SLAnimation;
    result->length(anim->mDuration);
    if (anim->mName.length > 0)
        result->name(anim->mName.C_Str());
    else
        result->name("Unnamed Animation");

    
    // exit if we didn't load a skeleton but have animations for one
    if (skinnedMeshes.size() > 0)
        assert(skel != NULL && "The skeleton wasn't impoted correctly.");

    SLbool isSkeletonAnim = false;
    for (SLint i = 0; i < anim->mNumChannels; i++)
    {
        aiNodeAnim* channel = anim->mChannels[i];

        // find the node that is animated by this channel
        SLNode* animatedNode = findLoadedNodeByName(channel->mNodeName.C_Str());

        // is the affected node part of a skeleton?
        BoneInformation* boneInfo = getBoneInformation(channel->mNodeName.C_Str());
        SLbool isBoneNode = boneInfo != NULL;
        SLuint nodeId = 0;
        if (isBoneNode) nodeId = boneInfo->id;
        if (isBoneNode) isSkeletonAnim = true;
        
        // bone animation channels should receive the correct node id, normal node animations just get 0
        SLNodeAnimationTrack* track = result->createNodeAnimationTrack(nodeId);

        // @todo Assimp provides keyframes seperately for position, rotation and scale, we combine them into one
        //          So we have to add in the other two components for the assimp keyframes. The drawback here is
        //          that we have to interpolate the missing values and that might lead to differet results.
        //          One solution would be to just give a policy of providing the animation with keyframe values
        //          for all three components.
        SLint numKeyframes = max(max(channel->mNumPositionKeys, channel->mNumRotationKeys), channel->mNumScalingKeys);

        for (SLint i = 0; i < numKeyframes; i++)
        {
            aiQuatKey rotKey = channel->mRotationKeys[i];

            SLTransformKeyframe* kf = track->createNodeKeyframe(rotKey.mTime);
            kf->rotation(SLQuat4f(rotKey.mValue.x, rotKey.mValue.y, rotKey.mValue.z, rotKey.mValue.w));

            if (i < channel->mNumPositionKeys) {

            }
        }

    }

    if (isSkeletonAnim)
        skel->addAnimation(result);

    return result;
}
//-----------------------------------------------------------------------------
/*!
SLAssimp::aiNodeHasMesh returns true if the passed node or one of its children 
has a mesh. aiNode can contain only transform or bone nodes without any visuals. 
*/
SLbool aiNodeHasMesh(aiNode* node)
{
    if (node->mNumMeshes > 0) return true;
    for(SLuint i = 0; i < node->mNumChildren; i++) 
        return aiNodeHasMesh(node->mChildren[i]);
    return false;
}
//-----------------------------------------------------------------------------
/*! 
SLAssImp::checkFilePath tries to build the full absolut texture file path. 
Some file formats have absolut path stored, some have relative paths.
1st attempt: modelPath + aiTexFile
2nd attempt: aiTexFile
3rd attempt: modelPath + getFileName(aiTexFile)
If a model contains absolut path it is best to put all texture files beside the
model file in the same folder.
*/
SLstring checkFilePath(SLstring modelPath, SLstring aiTexFile)
{
    // Check path & file combination
    SLstring pathFile = modelPath + aiTexFile;
    if (SLFileSystem::fileExists(pathFile))
        return pathFile;

    // Check file alone
    if (SLFileSystem::fileExists(aiTexFile))
        return aiTexFile;

    // Check path & file combination
    pathFile = modelPath + SLUtils::getFileName(aiTexFile);
    if (SLFileSystem::fileExists(pathFile))
        return pathFile;

    SLstring msg = "SLAssImp: Texture file not found: \n" + aiTexFile + 
                    "\non model path: " + modelPath + "\n";
    SL_WARN_MSG(msg.c_str());

    // Return path for texture not found image;
    return SLGLTexture::defaultPath + "TexNotFound.png";
}
//-----------------------------------------------------------------------------

