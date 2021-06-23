#include <SLSceneView.h>
#include <SLNodeLOD.h>

void SLNodeLOD::cullChildren3D(SLSceneView* sv)
{
    // TODO(dgj1): properly choose LOD to use
    if (!_children.empty())
    {
#if 0 // Distance based LOD selection
        SLint   numChildren  = _children.size();
        SLfloat maxDistance  = 100.0f;
        SLfloat lodThreshold = maxDistance / numChildren;

        SLVec3f cameraPos    = sv->camera()->translationWS();
        SLVec3f nodePos      = translationWS();
        SLfloat distToCamera = nodePos.distance(cameraPos);

        SLint childIndex = (SLint)distToCamera / lodThreshold;
        childIndex       = MAX(0, MIN(childIndex, numChildren - 1));
#else // Area based LOD selection \
      // TODO(dgj1): do we need to update aabb first? probably...
        updateAABBRec();
        SLAABBox* aabb = &_aabb;
        SLVec3f   min  = aabb->minWS();
        SLVec3f   max  = aabb->maxWS();

        SLVec3f points[8];
        points[0] = min;
        points[1] = SLVec3f(max.x, min.y, min.z);
        points[2] = SLVec3f(min.x, max.y, min.z);
        points[3] = SLVec3f(max.x, max.y, min.z);

        points[4] = SLVec3f(min.x, min.y, max.z);
        points[5] = SLVec3f(max.x, min.y, max.z);
        points[6] = SLVec3f(min.x, max.y, max.z);
        points[7] = max;

        SLGLState* stateGL              = SLGLState::instance();
        SLMat4f    viewProjectionMatrix = stateGL->projectionMatrix * stateGL->viewMatrix;

        for (SLint i = 0; i < 8; i++)
        {
            points[i] = viewProjectionMatrix.multVec(points[i]);
        }

        SLVec2f maxProjected = SLVec2f(points[0].x, points[0].y);
        SLVec2f minProjected = SLVec2f(points[0].x, points[0].y);
        for (SLint i = 0; i < 8; i++)
        {
            maxProjected.x = MAX(maxProjected.x, points[i].x);
            minProjected.x = MIN(minProjected.x, points[i].x);

            maxProjected.y = MAX(maxProjected.y, points[i].y);
            minProjected.y = MIN(minProjected.y, points[i].y);
        }

        SLVec2f areaVec        = 0.25f * ((maxProjected - minProjected) + SLVec2f(2.0f, 2.0f));
        SLfloat areaPercentage = areaVec.x * areaVec.y;

        // TODO(dgj1): make switching percentage parametrizable
        SLint childIndex;
        if (areaPercentage < 0.2f)
        {
            childIndex = 3;
        }
        else if (areaPercentage < 0.4f)
        {
            childIndex = 2;
        }
        else if (areaPercentage < 0.6f)
        {
            childIndex = 1;
        }
        else
        {
            childIndex = 0;
        }

        SLint numChildren = _children.size();
        childIndex        = MAX(0, MIN(childIndex, numChildren - 1));
#endif

        _children[childIndex]->cull3DRec(sv);
    }
}