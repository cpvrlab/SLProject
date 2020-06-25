#pragma once

#include <string>
#include <Mesh.h>

using namespace std;
//-----------------------------------------------------------------------------
class Rectangle : public Mesh
{
public:
    Rectangle(string name) : Mesh(name) { build(); }
    void build();
};
//-----------------------------------------------------------------------------
