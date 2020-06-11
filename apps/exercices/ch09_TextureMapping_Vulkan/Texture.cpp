#include "Texture.h"

Texture::Texture(string name, string filename) : Object(name), _filename(filename)
{
    load();
}

void Texture::load()
{
    CVImage image;
    image.load(_filename);
    _images.push_back(&image);
}
