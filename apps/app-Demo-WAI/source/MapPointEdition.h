#ifndef SL_MAP_EDITION_NODE_H
#define SL_MAP_EDITION_NODE_H

#include <SLSceneView.h>
#include <SLTransformNode.h>
#include <WAIMapPoint.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLMesh.h>

class MapEdition : public SLTransformNode
{
public:
    MapEdition(SLSceneView* sv, SLNode* mappointNode, vector<WAIMapPoint*> mp, vector<WAIKeyFrame*> kf, SLstring shaderDir);
    ~MapEdition() override;

    void updateKFVidMatching(std::vector<int>* kFVidMatching);
    void selectAllMap();
    void selectNMatched(std::vector<bool> nmatches);
    void selectByVid(std::vector<bool> vid);

    SLbool onKeyPress(const SLKey key, const SLKey mod) override;

    SLbool onKeyRelease(const SLKey key, const SLKey mod) override;

    SLbool onMouseDown(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;

    SLbool onMouseUp(SLMouseButton button,
                     SLint         x,
                     SLint         y,
                     SLKey         mod) override;

    SLbool onMouseMove(SLMouseButton button,
                       SLint         x,
                       SLint         y,
                       SLKey         mod) override;

    void editMode(SLNodeEditMode editMode) override;

    void deleteMesh(SLPoints*& mesh);

    void updateVisualization();

    void updateMapPointsMeshes(std::string                      name,
                               const std::vector<WAIMapPoint*>& pts,
                               SLPoints*&                       mesh,
                               SLMaterial*&                     material);

private:
    int                  xStart, yStart;
    vector<unsigned int> selected;
    SLSceneView*         _sv;
    SLCamera*            _camera;
    SLNode*              _mapNode;
    SLNode*              _workingNode;
    vector<WAIMapPoint*> _mappoints;
    vector<WAIKeyFrame*> _keyframes;
    SLGLProgram*         _prog = nullptr;
    SLMaterial*          _green;
    SLMaterial*          _yellow;
    std::vector<int>     _meshToMP;
    vector<WAIMapPoint*> _activeSet;
    vector<WAIMapPoint*> _temporarySet;
    SLPoints*            _mesh = nullptr;
    bool                 _transformMode;

    std::vector<std::vector<WAIKeyFrame*>> _vidToKeyframes;
    std::map<WAIKeyFrame*, int>            _keyFramesToVid;
};

#endif
