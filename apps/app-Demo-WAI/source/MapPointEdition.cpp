#include <MapPointEdition.h>
#include <SLAssetManager.h>
#include <SLMaterial.h>
#include <SLPoints.h>
#include <WAISlam.h>

MapEdition::MapEdition(SLSceneView* sv, SLNode* mappointNode, vector<WAIMapPoint*> mp, SLstring shaderDir)
  : SLNode("Map Points Edit")
{ 
    _sv = sv;
    _mapNode = mappointNode;
    _mappoints = mp;

    _prog = new SLGLGenericProgram(nullptr, shaderDir + "ColorUniformPoint.vert", shaderDir + "Color.frag");
    _prog->addUniform1f(new SLGLUniform1f(UT_const, "u_pointSize", 3.0f));

    _yellow  = new SLMaterial(nullptr, "Yellow Opaque", SLCol4f::YELLOW, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);
    _green  = new SLMaterial(nullptr, "Green Opaque", SLCol4f::GREEN, SLVec4f::WHITE, 100.0f, 0.0f, 0.0f, 0.0f, _prog);

    _sv->s().eventHandlers().push_back(this);

    updateMapPointsMeshes("current map points", _mappoints, _mesh, _green);
}

MapEdition::~MapEdition()
{
    deleteMesh(_mesh);
    delete _prog;
    delete _yellow;
    delete _green;
    deleteChildren();
}

void MapEdition::selectByVid(std::vector<int>* kFVidMatching, int id)
{


}

void MapEdition::updateMapPointsMeshes(std::string                      name,
                                       const std::vector<WAIMapPoint*>& pts,
                                       SLPoints*&                       mesh,
                                       SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    deleteMesh(mesh);
    unsigned int i = 0;
    _meshToMP.clear();

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            i++;
            if (mapPt->isBad())
                continue;

            _meshToMP.push_back(i-1);

            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(_mapNode->om() * SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(_mapNode->om() * SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(nullptr, points, normals, name, material);
        addMesh(mesh);
        updateAABBRec();
        _sv->s().selectNodeMesh(this, mesh);
    }
}

void MapEdition::deleteMesh(SLPoints*& mesh)
{
    if (mesh)
    {
        if (removeMesh(mesh))
        {
            delete mesh;
            mesh = nullptr;
        }
    }
}
   
SLbool MapEdition::onKeyPress(const SLKey key, const SLKey mod)
{
    (void)key;
    (void)mod;

    return false;
}

SLbool MapEdition::onKeyRelease(const SLKey key, const SLKey mod)
{
    (void)mod;

    if (key == K_delete)
    {
        for (unsigned int i = 0; i < _mesh->IS32.size(); i++)
        {
            _mappoints[_meshToMP[_mesh->IS32[i]]]->SetBadFlag();
        }
        updateMapPointsMeshes("current map points", _mappoints, _mesh, _green);
    }
    return false;
}


SLbool MapEdition::onMouseDown(SLMouseButton button,
                               SLint         x,
                               SLint         y,
                               SLKey         mod)
{
    return false;
}

SLbool MapEdition::onMouseMove(const SLMouseButton button,
                               SLint               x,
                               SLint               y,
                               const SLKey         mod)
{
    return false;
}

SLbool MapEdition::onMouseUp(SLMouseButton button,
                             SLint         x,
                             SLint         y,
                             SLKey         mod)
{
    return false;
}
