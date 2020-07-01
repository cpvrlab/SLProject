#include <MapPointEdition.h>
#include <SLAssetManager.h>
#include <SLMaterial.h>
#include <SLPoints.h>
#include <WAISlam.h>

MapEdition::MapEdition(SLSceneView* sv, SLNode* mappointNode, vector<WAIMapPoint*> mp, vector<WAIKeyFrame*> kf, SLstring shaderDir)
  : SLNode("Map Points Edit")
{ 
    _sv = sv;
    _mapNode = mappointNode;
    _mappoints = mp;
    _keyframes = kf;

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

void MapEdition::updateKFVidMatching(std::vector<int>* kFVidMatching)
{
    _kfSet.clear();
    for (int i = 0; i < kFVidMatching->size(); i++)
    {
        if (kFVidMatching->at(i) == -1)
            continue;
        if (kFVidMatching->at(i) >= _kfSet.size())
        {
            _kfSet.resize(kFVidMatching->at(i)+1);
        }
        _kfSet[kFVidMatching->at(i)].push_back(i); //add keyframe i to kfSet[videoId]
    }
}

void MapEdition::selectByVid(int id)
{
    std::set<WAIMapPoint*> mpset;
    int lastIdx = 0;
    for (int i = 0; i < _kfSet[id].size(); i++)
    {
        int kfmnid = _kfSet[id][i];
        int j;
        for (j = 0; j < _keyframes.size() && _keyframes[j]->mnId != kfmnid; j++);

        if (j == _keyframes.size())
        {
            std::cout << "kf with mnid " << kfmnid << "not found" << std::endl;
            continue;
        }

        WAIKeyFrame* kf = _keyframes[j];

        std::vector<WAIMapPoint*> mps = kf->GetMapPointMatches();
        for (WAIMapPoint* mp : mps)
        {
            if (mp == nullptr || mp->isBad())
                continue;
            mpset.insert(mp);
        }
    }

    std::vector<WAIMapPoint*> mpvec;
    mpvec.resize(mpset.size());
    std::copy(mpset.begin(), mpset.end(), mpvec.begin());
    updateMapPointsMeshes("current map points", mpvec, _mesh, _green);
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

    std::cout << "mpset size " << pts.size() << std::endl;

    //instantiate and add new mesh
    if (pts.size())
    {
        //get points as Vec3f
        std::vector<SLVec3f> points, normals;
        for (auto mapPt : pts)
        {
            i++;
            if (!mapPt || mapPt->isBad())
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
