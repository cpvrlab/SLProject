//
// Created by vwm1 on 17/08/2021.
//

#include <vr/SLVRSystem.h>
#include <iostream>

vr::IVRSystem* SLVRSystem::_system = nullptr;

void SLVRSystem::startup()
{
    if (!checkStartupConditions()) {
        return;
    }


}

bool SLVRSystem::checkStartupConditions()
{
    if (!vr::VR_IsRuntimeInstalled())
    {
        std::cout << "The SteamVR runtime is not installed.\n";
        std::cout << "Please download SteamVR from: https://store.steampowered.com/app/250820/SteamVR/" << std::endl;
        return false;
    }

    if (!vr::VR_IsHmdPresent())
    {
        std::cout << "No HMD was detected." << std::endl;
        return false;
    }

    return true;
}

void SLVRSystem::shutdown()
{
}