#include <SLFileIO.h>

#ifdef SL_FS_WEB
SLReader::SLReader(unsigned char* data, int size) : data(data), size(size), offset(0)
{
}

void SLReader::read(unsigned char* dest, int size)
{
    if (offset + size > this->size)
    {
        SL_EXIT_MSG("Out of bounds read");
    }

    std::memcpy(dest, data + offset, size);
    offset += size;
}

// clang-format off
EM_ASYNC_JS(void, slFetch, (const char* urlPointer, int urlLength, int* outStatus, unsigned char** outData, int* outLength), {
    let url = UTF8ToString(urlPointer, urlLength);
    let response = await fetch(url);

    setValue(outStatus, response.status, "i32");

    if (response.status == 200) {
        let data = await response.arrayBuffer();

        let typedArray = new Uint8Array(data);
        let buffer = _malloc(data.byteLength);
        writeArrayToMemory(typedArray, buffer);

        setValue(outData, buffer, "i8*");
        setValue(outLength, data.byteLength, "i32");
    } else {
        setValue(outData, 0, "i8*");
        setValue(outLength, 0, "i32");
    }
})
// clang-format on
#endif

SLImageFile SLFileIO::loadImage(SLstring path)
{
#if defined(SL_FS_NATIVE)
    return cv::imread(path, -1);
#elif defined(SL_FS_WEB)
    SLFetchResult result = SLFileIO::fetch(path);
    if (result.status != 200)
    {
        std::cout << "ERROR: FAILED TO FETCH IMAGE" << std::endl;
        return SLImageFile();
    }

    if (Utils::endsWithString(path, ".png"))
    {
        std::cout << "DECODE PNG" << std::endl;
        SLImageFile image = SLFileIO::decodePNG(result);
        std::cout << "DONE" << std::endl;
        free(result.data);
        return image;
    }
    else if (Utils::endsWithString(path, ".jpg") || Utils::endsWithString(path, ".jpeg"))
    {
        std::cout << "DECODE JPEG" << std::endl;
        SLImageFile image = SLFileIO::decodeJPEG(result);
        std::cout << "DONE" << std::endl;
        free(result.data);
        return image;
    }
    else
    {
        std::cout << "ERROR: IMAGE FORMAT NOT SUPPORTED" << std::endl;
        free(result.data);
        return SLImageFile();
    }
#endif
}

#ifdef SL_FS_WEB
SLFetchResult SLFileIO::fetch(SLstring url)
{
    std::cout << "FETCH " << url << std::endl;

    const char*   urlPointer = url.c_str();
    int           urlLength  = (int)url.length();
    SLFetchResult result;
    slFetch(urlPointer, urlLength, &result.status, &result.data, &result.length);

    std::cout << "STATUS: " << result.status << ", SIZE: " << result.length << std::endl;
    return result;
}

SLImageFile SLFileIO::decodePNG(SLFetchResult& fetchResult)
{
    // Based on:
    //   https://github.com/opencv/opencv/blob/4.x/modules/imgcodecs/src/grfmt_png.cpp
    //   http://pulsarengine.com/2009/01/reading-png-images-from-memory/

    // Create PNG structs
    png_structp png  = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop   info = png_create_info_struct(png);

    // Read file into PNG structs
    SLReader reader(fetchResult.data, fetchResult.length);
    png_set_read_fn(png, &reader, SLFileIO::pngReadData);
    png_read_png(png, info, PNG_TRANSFORM_BGR, NULL);
    png_bytepp rows = png_get_rows(png, info);

    // Get width, height, color type and bit depth
    size_t   width     = (size_t)png_get_image_width(png, info);
    size_t   height    = (size_t)png_get_image_height(png, info);
    png_byte colorType = png_get_color_type(png, info);
    png_byte bitDepth  = png_get_bit_depth(png, info);

    // Get number of channels from color type
    int numChannels = 0;
    if (colorType == PNG_COLOR_TYPE_RGB)
        numChannels = 3;
    else if (colorType == PNG_COLOR_TYPE_RGBA)
        numChannels = 4;
    else if (colorType == PNG_COLOR_TYPE_GRAY)
        numChannels = 1;

    // Copy data to temporary buffer
    size_t         size      = numChannels * width * height;
    unsigned char* data      = new unsigned char[size];
    size_t         rowStride = numChannels * width;

    for (size_t row = 0; row < height; row++)
    {
        unsigned char* dest = &data[numChannels * width * row];
        std::memcpy(dest, rows[row], numChannels * width);
    }

    // Create OpenCV image
    int   cvBitDepth = bitDepth == 16 ? CV_16U : CV_8U;
    int   type       = CV_MAKE_TYPE(cvBitDepth, numChannels);
    CVMat image((int)width, (int)height, type, data);

    // Cleanup temporary objects
    png_destroy_read_struct(&png, &info, NULL);
    delete[] data;

    return image;
}

void SLFileIO::pngReadData(png_structp png, png_bytep data, png_size_t length)
{
    // Read bytes from the IO pointer and advance it
    SLReader* reader = (SLReader*)png_get_io_ptr(png);
    reader->read(data, (int)length);
}

SLImageFile SLFileIO::decodeJPEG(SLFetchResult& fetchResult)
{
    // Based on: https://gist.github.com/PhirePhly/3080633

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr         err;

    cinfo.err = jpeg_std_error(&err);
    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, fetchResult.data, fetchResult.length);
    if (!jpeg_read_header(&cinfo, TRUE)) return SLImageFile();

    jpeg_start_decompress(&cinfo);
    size_t width       = cinfo.output_width;
    size_t height      = cinfo.output_height;
    int    numChannels = cinfo.output_components;

    size_t         size      = numChannels * width * height;
    unsigned char* data      = new unsigned char[size];
    size_t         rowStride = numChannels * width;

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char* dest = &data[rowStride * cinfo.output_scanline];
        jpeg_read_scanlines(&cinfo, &dest, 1);
    }

    // Convert RGB to BGR
    for (int i = 0; i < numChannels * width * height; i += numChannels)
        std::swap(data[i + 0], data[i + 2]);

    // Create OpenCV image
    int   type = CV_MAKE_TYPE(CV_8U, numChannels);
    CVMat image((int)width, (int)height, type, data);

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    delete[] data;

    return image;
}
#endif
