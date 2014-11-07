
#include <stdafx.h>
#include <SLSkeleton.h>



SLBone* SLSkeleton::getBone(SLuint handle)
{
    if (_boneMap.find(handle) == _boneMap.end())
        return NULL;

    return _boneMap[handle];
}

void SLSkeleton::getBoneWorldMatrices(SLMat4f* boneWM)
{
    // @todo ...
}