#pragma once

#include <string>

#include <Mesh.h>

using namespace std;
//-----------------------------------------------------------------------------
class Sphere : public Mesh
{

public:
    Sphere(string name) : Mesh(name) { ; }

protected:
    float _radius; //!< radius of sphere
    int   _stack;  //!< NO. of stacks
    int   _slices; //!< NO. of slices
};
//-----------------------------------------------------------------------------
