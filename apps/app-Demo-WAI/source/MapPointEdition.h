#ifndef SL_MAP_EDITION_NODE_H
#define SL_MAP_EDITION_NODE_H

#include <SLSceneView.h>
#include <WAIMapPoint.h>
#include <SLPoints.h>
#include <SLPolyline.h>
#include <SLMesh.h>

class MapEdition : public SLNode
{
public:
    MapEdition(SLSceneView* sv, SLNode* mappointNode, vector<WAIMapPoint*> mp, SLstring shaderDir);
    ~MapEdition() override;

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

    void deleteMesh(SLPoints*& mesh);
    void updateMapPointsMeshes(std::string                      name,
                         const std::vector<WAIMapPoint*>& pts,
                         SLPoints*&                       mesh,
                         SLMaterial*&                     material);

private:
    int xStart, yStart;
    vector<unsigned int> selected;
    SLSceneView * _sv;
    SLCamera * _camera;
    SLNode * _mapNode;
    vector<WAIMapPoint*> _selected;
    vector<WAIMapPoint*> _unselected;
    vector<WAIMapPoint*> _mappoints;
    SLPoints *_mesh1 = nullptr;
    SLPoints *_mesh2 = nullptr;
    SLGLProgram* _prog  = nullptr;
    SLMaterial *_green;
    SLMaterial *_yellow;
};

#endif
