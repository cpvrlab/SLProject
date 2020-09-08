#ifndef GPSMAPPER2D_H
#define GPSMAPPER2D_H

#include <SLVec3.h>
#include <SLMat3.h>
#include <SLVec2.h>
#include <SLQuat4.h>

class GPSMapper2D
{
public:
    GPSMapper2D(const SLVec3d& tl, const SLVec3d& br, const int imgWPix, const int imgHPix);

    //transform given lla location to map coordinate system
    SLVec2i mapLLALocation(const SLVec3d& llaLoc);
    int toMapScale(const double valMeter);

private:
    //calcualte initial values
    void init();
    void calcMapOrigin();
    void calcScale();

    SLVec3d _tlLLA;
    SLVec3d _brLLA;
    int     _imgWPix = 0;
    int     _imgHPix = 0;

    //map origin in ecef frame
    SLVec3d _ecefO;
    //global map origin wrt. map frame (used to translate to local map coordinate frame)
    SLVec3d _mapO;
    //ecef to map rotation matrix
    SLMat3d _mapRecef;
    //scalefactor to transform meter to pixel in width direction
    double _mToPixW = 1.f;
    //scalefactor to transform meter to pixel in height direction
    double _mToPixH = 1.f;
};

#endif
