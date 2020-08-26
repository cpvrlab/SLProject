#pragma once

#include <string>
#include <Mesh.h>

using namespace std;
//-----------------------------------------------------------------------------
class Sphere : public Mesh
{
public:
    Sphere(string name) : Mesh(name) { build(); }
    Sphere(string name, float radius, int stacks, int slice) : Mesh(name), _radius(radius), _stacks(stacks), _slices(slice) { build(); }

    void build();

    // Setter
    // void setRadius(float radius) { _radius = radius; }
    // void setStacks(float stacks) { _stacks = stacks; }
    // void setSlices(float slices) { _slices = slices; }

    // Getters
    float radius() { return _radius; }
    int   stacks() { return _stacks; }
    int   slices() { return _slices; }

protected:
    float _radius; //!< radius of sphere
    int   _stacks; //!< NO. of stacks
    int   _slices; //!< NO. of slices
};
//-----------------------------------------------------------------------------
