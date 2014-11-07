
#include <stdafx.h>
#include <SLBone.h>

// set a new offset matrix
void SLBone::offsetMat(const SLMat4f& mat)
{
    _offsetMat = mat;
}

// 
SLMat4f SLBone::calculateFinalMat()
{
    return updateAndGetWM() * _offsetMat;
}