#ifndef VKENUMS_H
#define VKENUMS_H

enum ShaderType
{
    NONE                    = 0x00000000,
    VERTEX                  = 0x00000001,
    TESSELLATION_CONTROL    = 0x00000002,
    TESSELLATION_EVALUATION = 0x00000004,
    GEOMETRY                = 0x00000008,
    FRAGMENT                = 0x00000010,
    COMPUTE                 = 0x00000020,
};

#endif
