#pragma once

#include <string>
#include <vector>

using namespace std;

//-----------------------------------------------------------------------------
class Object
{
public:
    Object(string name) : _name(name) { ; }

protected:
    string _name;
};
//-----------------------------------------------------------------------------
