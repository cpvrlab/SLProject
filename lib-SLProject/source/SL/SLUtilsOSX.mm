//#############################################################################
//  File:      SLUtilsOSX.mm
//  Author:    Marcus Hudritsch
//  Date:      August 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL PURPOSE ONLY AND
//             WITHOUT ANY WARRANTIES WHETHER EXPRESSED OR IMPLIED.
//#############################################################################

#include <stdafx.h>
#include <Foundation/Foundation.h>

//-----------------------------------------------------------------------------
//! Mac OSX NSLog
void SLNSLog(...)
{
    char* msg[200];
    sprintf(msg, __VA_ARGS__);
    NSLog(@"%s", msg);
}
//-----------------------------------------------------------------------------






