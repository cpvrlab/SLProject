#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <vector>

#include <Object.h>
#include <CVImage.h>

using namespace std;

//-----------------------------------------------------------------------------
class Texture : public Object
{

public:
    Texture(string name, string filename);

    void load();

protected:
    string   _filename; //!< path and filename of the texture image file
    CVVImage _images;   //!< vector of opencv images
};
//-----------------------------------------------------------------------------
typedef vector<Texture> VTexture;
//-----------------------------------------------------------------------------
#endif
