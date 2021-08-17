//
// Created by vwm1 on 17/08/2021.
//

#ifndef SLPROJECT_SLVRSYSTEM_H
#define SLPROJECT_SLVRSYSTEM_H

#include <openvr.h>

class SLVRSystem {

private:
    static vr::IVRSystem* _system;

public:
    static void startup();

private:
    static bool checkStartupConditions();

    static void shutdown();

};

#endif // SLPROJECT_SLVRSYSTEM_H
