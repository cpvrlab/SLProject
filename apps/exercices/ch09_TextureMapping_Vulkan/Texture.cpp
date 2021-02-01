#include "Texture.h"

Texture::Texture(string name, const string filename) : Object(name), _filename(filename)
{
    _image.load(filename);
}

void Texture::load()
{
    // _image.load(_filename);
}
