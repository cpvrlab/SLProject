#include "GPSMapper2D.h"
#include <Utils.h>

GPSMapper2D::GPSMapper2D(const SLVec3d& tl, const SLVec3d& br, const int imgWPix, const int imgHPix)
  : _tlLLA(tl),
    _brLLA(br),
    _imgWPix(imgWPix),
    _imgHPix(imgHPix)
{
    init();
}

SLVec2i GPSMapper2D::mapLLALocation(const SLVec3d& llaLoc)
{
    //transform to ecef frame
    SLVec3d ecefP;
    ecefP.latlonAlt2ecef(llaLoc);

    //rotate to map frame
    SLVec3d mapP = _mapRecef * ecefP;

    //translate to local map frame
    mapP -= _mapO;

    //scale to pixel
    SLVec2i pixPos;
    pixPos.x = (int)(mapP.x * _mToPixW);
    pixPos.y = (int)(mapP.y * _mToPixH);

    return pixPos;
}

int GPSMapper2D::toMapScale(const double valMeter)
{
    return (int)(valMeter * (_mToPixW + _mToPixH) * 0.5);
}

//calcualte initial values
void GPSMapper2D::init()
{
    //calculate map origin in ecef frame
    calcMapOrigin();

    //calculate scale from local to pixel coordinate system
    calcScale();
}

void GPSMapper2D::calcMapOrigin()
{
    //calculate map origin in ecef frame
    _ecefO.latlonAlt2ecef(_tlLLA);

    //find ecef to map rotation matrix:
    //1. calculate rotation of ecef frame wrt. enu frame
    double phiRad = _tlLLA.x * Utils::DEG2RAD; //   phi == latitude
    double lamRad = _tlLLA.y * Utils::DEG2RAD; //lambda == longitude
    double sinPhi = sin(phiRad);
    double cosPhi = cos(phiRad);
    double sinLam = sin(lamRad);
    double cosLam = cos(lamRad);

    SLMat3d enuRecef(-sinLam,
                     cosLam,
                     0,
                     -cosLam * sinPhi,
                     -sinLam * sinPhi,
                     cosPhi,
                     cosLam * cosPhi,
                     sinLam * cosPhi,
                     sinPhi);
    //2. ENU frame w.r.t. map origin
    SLMat3d mapRenu;
    mapRenu.rotation(180, 1, 0, 0);
    //todo: imgui origin? tl or bl?

    //3.ecef to map rotation matrix
    _mapRecef = mapRenu * enuRecef;

    //ECEF w.r.t. map frame
    _mapO = _mapRecef * _ecefO;
}

void GPSMapper2D::calcScale()
{
    //bottom right corner in ecef
    SLVec3d ecefBR;
    ecefBR.latlonAlt2ecef(_brLLA);
    //rotate br ecef to global map frame
    SLVec3d mapBR = _mapRecef * ecefBR;

    //image width and height in meter
    float imgWM = mapBR.x - _mapO.x;
    float imgHM = mapBR.y - _mapO.y;

    //calculate scale factors
    _mToPixW = _imgWPix / imgWM;
    _mToPixH = _imgHPix / imgHM;
}
