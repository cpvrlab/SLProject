//
// Created by nic on 26.12.19.
//

#ifndef SLPROJECT_SLLINE_H
#define SLPROJECT_SLLINE_H

#include <SLEnums.h>
#include <SLMesh.h>

class SLLine : public SLMesh
{
public:
    SLLine( SLVec3f     p1,
            SLVec3f     p2,
            SLMaterial* mat  = nullptr);

    void init(SLNode* node) override;

    void    createMeshAccelerationStructure() override;
    void    updateMeshAccelerationStructure() override;
    HitData createHitData() override;

private:
    SLVec3f _p1; //!< origin point
    SLVec3f _p2; //!< end point

    SLCudaBuffer<OptixAabb> _aabb = SLCudaBuffer<OptixAabb>();
};

#endif //SLPROJECT_SLLINE_H
