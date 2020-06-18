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
    updateMapPointsMeshes("current map points", _mappoints, _mesh1, _yellow);
}

MapEdition::~MapEdition()
{
    deleteMesh(_mesh1);
    deleteMesh(_mesh2);
    delete _prog;
    delete _yellow;
    delete _green;
    deleteChildren();
    std::cout << "dajdjaosd  3 " << std::endl;
}

void MapEdition::updateMapPointsMeshes(std::string                      name,
                                       const std::vector<WAIMapPoint*>& pts,
                                       SLPoints*&                       mesh,
                                       SLMaterial*&                     material)
{
    //remove old mesh, if it exists
    deleteMesh(mesh);

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            if (mapPt->isBad())
                continue;
            WAI::V3 wP = mapPt->worldPosVec();
            WAI::V3 wN = mapPt->normalVec();
            points.push_back(_mapNode->om() * SLVec3f(wP.x, wP.y, wP.z));
            normals.push_back(_mapNode->om() * SLVec3f(wN.x, wN.y, wN.z));
        }

        mesh = new SLPoints(nullptr, points, normals, name, material);
        addMesh(mesh);
        updateAABBRec();
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
    (void)key;
    (void)mod;


    if (key == K_delete)
    {
        for (auto mapPt : _selected)
        {
            mapPt->SetBadFlag();
        }
    }
    _selected.clear();
    deleteMesh(_mesh2);
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

    if (!(mod & K_ctrl))
    {
        std::cout << "ctrl modifier not there" << std::endl;
        return false;
    }

    vector<WAIMapPoint*> mp;

    if (!(mod & K_shift))
    {
        std::cout << "shift modifier not there, clear selected" << std::endl;
        _selected.clear();
        mp = _mappoints;
    }
    else
    {
        mp = _unselected;
    }
    _unselected.clear(); 

    for (unsigned int i = 0; i < mp.size(); i++)
    {
        WAI::V3 vec = mp[i]->worldPosVec();
        SLVec2f p = _sv->camera()->projectWorldToNDC(_mapNode->om() * SLVec4f(vec.x, vec.y, vec.z, 1.0f));
        p.x = (p.x + 1.0) * 0.5 * _sv->viewportW();
        p.y = (-p.y + 1.0) * 0.5 * _sv->viewportH();

        
        if (_sv->camera()->selectedRect().contains(p))
        {
            _selected.push_back(mp[i]);
        }
        else
        {
            _unselected.push_back(mp[i]);
        }
    }
    updateMapPointsMeshes("current map points", _unselected, _mesh1, _yellow);
    updateMapPointsMeshes("selected mappoints", _selected, _mesh2, _green);

    return true;
}
