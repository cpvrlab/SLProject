#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <vector>

#include <Object.h>
#include <cv/CVImage.h>

using namespace std;

//-----------------------------------------------------------------------------
class Texture : public Object
{

public:
    Texture(string name, const string filename);
    Texture(Texture&) = default;

    // Getter
    uint   imageHeight() { return _image.height(); };
    uint   imageWidth() { return _image.width(); };
    uchar* imageData() { return _image.data(); };

protected:
    string  _filename; //!< path and filename of the texture image file
    CVImage _image;    //!< opencv image
};
//-----------------------------------------------------------------------------
typedef vector<Texture*> VTexture;
//-----------------------------------------------------------------------------
#endif
