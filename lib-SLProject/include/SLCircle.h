#ifndef SL_CIRCLE_H
#define SL_CIRCLE_H

#include <SLPolyline.h>
#include <SLNode.h>

class SLCircle : public SLPolyline
{
public:
    SLCircle(SLAssetManager* assetMgr, SLstring name = "Circle", SLMaterial* material = nullptr);
};

#endif