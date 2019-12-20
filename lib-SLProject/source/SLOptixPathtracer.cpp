//#############################################################################
//  File:      SLOptixPathtracer.cpp
//  Author:    Nic Dorner
//  Date:      Dezember 2019
//  Copyright: Nic Dorner
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################
#include <stdafx.h> // Must be the 1st include followed by  an empty line

#include <SLOptixPathtracer.h>

#ifdef SL_MEMLEAKDETECT    // set in SL.h for debug config only
#    include <debug_new.h> // memory leak detector
#endif

SLOptixPathtracer::SLOptixPathtracer(): SLOptixRaytracer() {
//    name("OptiX path tracer");
}

SLOptixPathtracer::~SLOptixPathtracer() {

}

void SLOptixPathtracer::setupOptix() {
    SLOptixRaytracer::setupOptix();

    _cameraModule   = _createModule("SLOptixPathtracerCamera.cu");
    _shadingModule  = _createModule("SLOptixPathtracerShading.cu");
}

void SLOptixPathtracer::setupScene(SLSceneView *sv) {
    SLOptixRaytracer::setupScene(sv);
}

void SLOptixPathtracer::updateScene(SLSceneView *sv) {
    SLOptixRaytracer::updateScene(sv);
}

SLbool SLOptixPathtracer::render() {
    return 0;
}

void SLOptixPathtracer::renderImage() {
    SLOptixRaytracer::renderImage();
}

