#include <SlamParams.h>

SlamParams* SlamParams::lastParams = nullptr;

SlamParams::SlamParams()
{
    lastParams = this;
}


SlamParams* SlamParams::lastInstance()
{
    return lastParams;
}

