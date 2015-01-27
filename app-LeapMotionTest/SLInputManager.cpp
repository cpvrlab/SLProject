
#include <stdafx.h>
#include <SLInputManager.h>


SLInputManager SLInputManager::_instance;

SLInputManager& SLInputManager::instance()
{
    return _instance;
}

SLInputManager::~SLInputManager()
{
    for(SLint i = 0; i < _devices.size(); ++i)
        delete _devices[i];

    _devices.clear();
}

void SLInputManager::update()
{
    for(SLint i = 0; i < _devices.size(); ++i)
        _devices[i]->poll();
}