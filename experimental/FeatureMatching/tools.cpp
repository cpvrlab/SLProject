#include <iostream>
#include <fstream>
#include "tools.h"

void init_patch(std::vector<int> &umax, int half_patch_size)
{
    // pre-compute the end of a row in a circular patch
    umax.resize(half_patch_size + 1);

    int          v, v0, vmax = cvFloor(half_patch_size * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(half_patch_size * sqrt(2.f) / 2);
    const double hp2  = half_patch_size * half_patch_size;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = half_patch_size, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

void init_pyramid_parameters(PyramidParameters &p, int nlevels, float scale_factor, int nfeatures)
{
    p.scale_factors.resize(nlevels);
    p.level_sigma2.resize(nlevels);
    p.inv_scale_factors.resize(nlevels);
    p.inv_level_sigma2.resize(nlevels);
    p.nb_feature_per_level.resize(nlevels);
    p.total_features = nfeatures;

    p.scale_factors[0] = 1.0f;
    p.level_sigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        p.scale_factors[i] = (float)(p.scale_factors[i - 1] * scale_factor);
        p.level_sigma2[i] = (float)(p.scale_factors[i] * p.scale_factors[i]);
    }

    for (int i = 0; i < nlevels; i++)
    {
        p.inv_scale_factors[i] = 1.0f / p.scale_factors[i];
        p.inv_level_sigma2[i] = 1.0f / p.level_sigma2[i];
    }

    float factor                   = 1.0f / scale_factor;
    float nb_features_per_scale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int total_features = 0;
    for (int level = 0; level < nlevels - 1; level++)
    {
        p.nb_feature_per_level[level] = cvRound(nb_features_per_scale);
        total_features += p.nb_feature_per_level[level];
        nb_features_per_scale *= factor;
    }
    p.nb_feature_per_level[nlevels - 1] = std::max(nfeatures - total_features, 0);

}

void build_pyramid(std::vector<cv::Mat> &image_pyramid, cv::Mat &image, PyramidParameters &p)
{
    image_pyramid.resize(p.scale_factors.size());
    for (int level = 0; level < p.scale_factors.size(); ++level)
    {
        float scale = p.inv_scale_factors[level];
        cv::Size  sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        cv::Size  wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        cv::Mat   temp(wholeSize, image.type());
        image_pyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if (level != 0)
        {
            cv::resize(image_pyramid[level - 1], image_pyramid[level], sz, 0, 0, cv::INTER_LINEAR);
            copyMakeBorder(image_pyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101 + cv::BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);
        }
    }
}

void flatten_keypoints(std::vector<cv::KeyPoint> &keypoints, std::vector<std::vector<cv::KeyPoint>>& all_keypoints, PyramidParameters &p)
{
    keypoints.insert(keypoints.begin(), all_keypoints[0].begin(), all_keypoints[0].end());

    for (int level = 1; level < p.scale_factors.size(); ++level)
    {
        std::vector<cv::KeyPoint>& kps = all_keypoints[level];
        keypoints.insert(keypoints.end(), kps.begin(), kps.end());
    }
}

void flatten_decriptors(std::vector<Descriptor> &desc, std::vector<std::vector<Descriptor>>& all_desc, PyramidParameters &p)
{
    desc.insert(desc.begin(), all_desc[0].begin(), all_desc[0].end());

    for (int level = 1; level < p.scale_factors.size(); ++level)
    {
        std::vector<Descriptor>& dsc = all_desc[level];
        desc.insert(desc.end(), dsc.begin(), dsc.end());
    }
}

unsigned int hamming_distance(unsigned int a, unsigned int b)
{
    unsigned int v = a ^ b;
    v              = v - ((v >> 1) & 0x55555555);
    v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
unsigned int hamming_distance(Descriptor &a, Descriptor &b)
{
    unsigned int dist = 0;
    uint32_t *pa = (uint32_t*)a.mem;
    uint32_t *pb = (uint32_t*)b.mem;

    for (int i = 0; i < 8; i++, pa++, pb++)
        dist += hamming_distance(*pa, *pb);

    return dist;
}

void print_uchar(uchar u)
{
    int i = (int)u;
    std::cout << (i & 0x1);
    std::cout << ((i & (0x1 << 1)) >> 1);
    std::cout << ((i & (0x1 << 2)) >> 2);
    std::cout << ((i & (0x1 << 3)) >> 3);
    std::cout << ((i & (0x1 << 4)) >> 4);
    std::cout << ((i & (0x1 << 5)) >> 5);
    std::cout << ((i & (0x1 << 6)) >> 6);
    std::cout << ((i & (0x1 << 7)) >> 7);
}

void print_desc(Descriptor &d)
{
    for (int i = 0; i < 32; i++)
    {
        print_uchar(d.mem[i]);
        std::cout << " ";
    }
    std::cout << std::endl << std::endl;
}

void compute_three_maxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

cv::Mat extract_patch(cv::Mat& image, cv::KeyPoint &kp)
{
    std::vector<int> umax;
    init_patch(umax, HALF_PATCH_SIZE);
    cv::Mat gray = rgb_to_grayscale (image);

    const uchar* center = &gray.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x));
    cv::Mat patch = cv::Mat::zeros(PATCH_SIZE, PATCH_SIZE, CV_8UC1);
    uchar * patch_center = &patch.at<uchar>(HALF_PATCH_SIZE, HALF_PATCH_SIZE);
 
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        patch_center[u] = center[u];

    // Go line by line in the circular patch
    int step = (int)gray.step1();

    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        int d = umax[v];
        for (int u = -d; u <= d; ++u)
        {
            int p = center[u + v * step];
            int psym = center[u - v * step];
            patch_center[u + v * patch.step] = p;
            patch_center[u - v * patch.step] = psym;
        }
    }
    return patch;
}

void get_inverted_matching(std::vector<int> &inverted_matching, std::vector<int> &matching)
{
    for (int i = 0; i < inverted_matching.size(); i++)
    {
        inverted_matching[i] = -1;
    }

    for (int i = 0; i < matching.size(); i++)
    {
        int idx1 = matching[i];

        if (idx1 < 0 || idx1 >= inverted_matching.size())
            continue;

        inverted_matching[idx1] = i;
    }
}

int select_closest_feature(std::vector<cv::KeyPoint> &keypoints, int x, int y)
{
    float min_dist = 10000000;
    int min_idx = -1;

    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f p1 = keypoints[i].pt;
        p1.x = p1.x - x;
        p1.y = p1.y - y;

        float dist = sqrt(p1.x * p1.x + p1.y * p1.y);

        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    return min_idx;
}

int select_closest_feature(std::vector<cv::KeyPoint> &keypoints, std::vector<int> matches, int x, int y)
{
    float min_dist = 10000000;
    int min_idx = -1;

    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f p1 = keypoints[i].pt;
        p1.x = p1.x - x;
        p1.y = p1.y - y;

        float dist = sqrt(p1.x * p1.x + p1.y * p1.y);

        if (dist < min_dist && matches[i] >= 0)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    return min_idx;
}

std::vector<int> select_closest_features(std::vector<cv::KeyPoint> &keypoints, float radius, int x, int y)
{
    std::vector<int> selection;
    int idx = select_closest_feature(keypoints, x, y);

    x = keypoints[idx].pt.x;
    y = keypoints[idx].pt.y;

    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::Point2f p1 = keypoints[i].pt;
        p1.x = p1.x - x;
        p1.y = p1.y - y;

        float dist = sqrt(p1.x * p1.x + p1.y * p1.y);

        if (dist < radius)
        {
            selection.push_back(i);
        }
    }
    return selection;
}

void compute_similarity(std::vector<cv::KeyPoint> &keypoints, std::vector<Descriptor> &descs, Descriptor &cur)
{
    float max_dist = 0;
    float min_dist = 0;

    for (int i = 0; i < keypoints.size(); i++)
    {
        float dist = hamming_distance(cur, descs[i]); 
        if (dist > max_dist)
            max_dist = dist;

        if (dist < min_dist)
            min_dist = dist;

        keypoints[i].response = dist;
    }

    for (int i = 0; i < keypoints.size(); i++)
    {
        keypoints[i].response = (keypoints[i].response - min_dist) / max_dist; //Set bet. [0 1]
        /*
        keypoints[i].response = 1.0 - keypoints[i].response;
        keypoints[i].response *= keypoints[i].response;
        keypoints[i].response = 1.0 - keypoints[i].response;
        */
    }
}

void reset_similarity_score(std::vector<cv::KeyPoint> &keypoints)
{
    for (int i = 0; i < keypoints.size(); i++)
    {
        keypoints[i].response = 1.0;
    }
}

std::vector<std::string> str_split(const std::string& str, char delim)
{
    std::vector<std::string> cont;
    std::size_t current, previous = 0;
    current = str.find(delim);
    while (current != std::string::npos) {
        cont.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delim, previous);
    }
    cont.push_back(str.substr(previous, current - previous));
    return cont;
}


void Tokenize(const std::string &mystring, std::vector<std::string> &tok,
              const std::string &sep = " ", int lp = 0, int p = 0)
{   
    lp = mystring.find_first_not_of(sep, p);
    p = mystring.find_first_of(sep, lp);
    if (std::string::npos != p || std::string::npos != lp) {
        tok.push_back(mystring.substr(lp, p - lp));
        Tokenize(mystring, tok, sep, lp, p);
    }
}

std::string delSpaces(std::string & str)
{
    std::stringstream trim;
    trim << str;
    trim >> str;
    return str;
}

void filters_open(std::string path, std::vector<float> &param, std::vector<float> &bias, std::vector<std::vector<float>> &coeffs, std::vector<cv::Mat> &filters, std::vector<std::string> &tokens)
{
    //std::ifstream fic(path, std::ios::in);
    std::ifstream fic(path);
    std::string lineread;

    if (!fic.is_open()) {
        std::cout << "Cannot open filter " << path << std::endl;
    }

    getline(fic, lineread);

    param.clear();
    tokens.clear();
    Tokenize(lineread, tokens, " ");

    for (int i = 0; i < tokens.size(); i++) {
        param.push_back(stof(delSpaces(tokens[i])));
    }

    getline(fic, lineread);

    tokens.clear();
    Tokenize(lineread, tokens);

    if (tokens.size() != 5) {
        std::cout << "Cannot open filter " << path << std::endl;
    }
    int nbMax = stoi(delSpaces(tokens[0]));
    int nbSum = stoi(delSpaces(tokens[1]));
    int nbOriginalFilters = nbMax * nbSum;
    int nbApproximatedFilters = stoi(delSpaces(tokens[2]));
    int nbChannels = stoi(delSpaces(tokens[3]));
    int sizeFilters = stoi(delSpaces(tokens[4]));

    param.push_back(nbMax);
    param.push_back(nbSum);
    param.push_back(nbApproximatedFilters);
    param.push_back(nbChannels);
    param.push_back(sizeFilters);

    //get bias
    getline(fic, lineread);
    tokens.clear();
    Tokenize(lineread, tokens);
    if (tokens.size() != nbOriginalFilters) {
        std::cout << "Wrong number of cascades" << std::endl;
    }
    //bias
    bias.resize(nbOriginalFilters);
    for (int i = 0; i < tokens.size(); i++)
        bias[i] = stof(delSpaces(tokens[i]));

    //coeffs
    coeffs = std::vector<std::vector<float>>(nbOriginalFilters, std::vector<float>(nbApproximatedFilters * nbChannels));
    int row = 0;
    while (getline(fic, lineread)) {
        tokens.clear();
        Tokenize(lineread, tokens);
        for (int i = 0; i < nbApproximatedFilters * nbChannels; i++)
            coeffs[row][i] = stof(delSpaces(tokens[i]));

        if (++row == nbOriginalFilters)
            break;
    }

    filters = std::vector<cv::Mat> (nbApproximatedFilters * nbChannels * 2, cv::Mat(1, sizeFilters, CV_32FC1));
    row = 0;
    while (getline(fic, lineread)) 
    {
        tokens.clear();
        Tokenize(lineread, tokens);

        std::vector<float>r(sizeFilters);
        for (int i = 0; i < sizeFilters; i++)
            r[i] = stof(delSpaces(tokens[i]));

        filters[row] = cv::Mat(r).clone();

        if (++row == nbApproximatedFilters * nbChannels * 2)
            break;
    }
}

//Return gx gy magnitude
std::vector<cv::Mat> image_gradient(const cv::Mat &input_rgb_image)
{
    //the output
    std::vector<cv::Mat> gradImage(3);
    std::vector<cv::Mat> color_channels(3);
    std::vector<cv::Mat> gx(3);
    std::vector<cv::Mat> gy(3);

    // The derivative5 kernels
    cv::Mat d1 = (cv::Mat_ <float>(1, 5) << 0.109604, 0.276691, 0.000000, -0.276691, -0.109604);
    cv::Mat d1T = (cv::Mat_ <float>(5, 1) << 0.109604, 0.276691, 0.000000, -0.276691, -0.109604);
    cv::Mat p = (cv::Mat_ <float>(1, 5) << 0.037659, 0.249153, 0.426375, 0.249153, 0.037659);
    cv::Mat pT = (cv::Mat_ <float>(5, 1) << 0.037659, 0.249153, 0.426375, 0.249153, 0.037659);

    // split the channels into each color channel
    cv::split(input_rgb_image, color_channels);
    // prepare output
    for (int idx = 0; idx < 3; ++idx) {
        gradImage[idx].create(color_channels[0].rows, color_channels[0].cols, CV_32F);
    }
    //	return gradImage;

    // for each channel do the derivative 5 
    for (int idxC = 0; idxC < 3; ++idxC) {
        cv::sepFilter2D(color_channels[idxC], gx[idxC], CV_32F, d1, p, cv::Point(-1, -1), 0,
                cv::BORDER_REFLECT);
        cv::sepFilter2D(color_channels[idxC], gy[idxC], CV_32F, p, d1, cv::Point(-1, -1), 0,
                cv::BORDER_REFLECT);
        // since we do the other direction, just flip signs
        gx[idxC] = -gx[idxC];
        gy[idxC] = -gy[idxC];
    }

    // the magnitude image
    std::vector<cv::Mat> mag(3);
    for (int idxC = 0; idxC < 3; ++idxC) {
        cv::sqrt(gx[idxC].mul(gx[idxC]) + gy[idxC].mul(gy[idxC]), mag[idxC]);
    }

    // Keep only max from each color component (based on magnitude)
    float curVal, maxVal; int maxIdx;
    for (int i = 0; i < mag[0].rows; i++)
    {
        float* pixelin1[3];
        float* pixelin2[3];
        float* pixelin3[3];

        for (int idxC = 0; idxC < 3; ++idxC)
        {
            pixelin1[idxC] = gx[idxC].ptr<float>(i);  // point to first color in row
            pixelin2[idxC] = gy[idxC].ptr<float>(i);  // point to first color in row
            pixelin3[idxC] = mag[idxC].ptr<float>(i);  // point to first color in row
        }

        float* pixelout1 = gradImage[0].ptr<float>(i);  // point to first color in row
        float* pixelout2 = gradImage[1].ptr<float>(i);  // point to first color in row
        float* pixelout3 = gradImage[2].ptr<float>(i);  // point to first color in row

        for (int j = 0; j < mag[0].cols; j++)
        {
            maxIdx = 0;
            maxVal = 0;
            for (int idxC = 0; idxC < 3; ++idxC)
            {
                curVal = *pixelin3[idxC];
                if (maxVal < curVal) {
                    maxIdx = idxC;
                    maxVal = curVal;
                }
            }
            *pixelout1++ = *pixelin1[maxIdx] * 0.5 + 128.0;
            *pixelout2++ = *pixelin2[maxIdx] * 0.5 + 128.0;
            *pixelout3++ = *pixelin3[maxIdx];

            //next in
            for (int idxC = 0; idxC < 3; ++idxC)
            {
                pixelin1[idxC]++;
                pixelin2[idxC]++;
                pixelin3[idxC]++;
            }
        }
    }

    return gradImage;
}

std::vector<cv::Point3f> NonMaxSup(const cv::Mat &response)
{
    std::vector<cv::Point3f> res;

    for(int i=1; i < response.rows-1; ++i)
    {
        for(int j=1; j < response.cols-1; ++j)
        {
            bool bMax = true;

            for(int ii=-1; ii <= +1; ++ii)
            {
                for(int jj=-1; jj <= +1; ++jj)
                {
                    if (ii == 0 && jj == 0)
                        continue;
                    bMax &= response.at<float>(i,j) > response.at<float>(i+ii,j+jj);
                }
            }

            if (bMax)
            {
                res.push_back(cv::Point3f(j,i, response.at<float>(i,j)));
            }
        }            
    }

    return res;
}

float angle_from_gradiant(cv::Mat &image, cv::KeyPoint &kp)
{
    cv::Point pt = kp.pt / (kp.size / PATCH_SIZE);
    const uchar* corner = &image.at<uchar>(cvRound(pt.y - HALF_PATCH_SIZE), cvRound(pt.x - HALF_PATCH_SIZE));
    cv::Mat patch = cv::Mat::zeros(PATCH_SIZE, PATCH_SIZE, CV_8UC1);
    uchar * patch_corner = &patch.at<uchar>(0, 0);

    int step = (int)image.step1();

    for (int v = 0; v < PATCH_SIZE; ++v)
    {
        for (int u = 0; u < PATCH_SIZE; ++u)
        {
            int p = corner[u + v * step];
            patch_corner[u + v * patch.step] = p;
        }
    }
    cv::imshow("patch", patch);

    cv::Mat tmp = patch.clone();
    tmp.convertTo(tmp, CV_32F, 1/255.0);

    cv::Mat gx, gy; 
    cv::Scharr(tmp, gx, CV_32F, 1, 0, 3, 0, cv::BORDER_REFLECT_101);
    cv::Scharr(tmp, gy, CV_32F, 0, 1, 3, 0, cv::BORDER_REFLECT_101);

    cv::Mat img_mag, img_angle;
    cv::cartToPolar(gx, gy, img_mag, img_angle, true);

    float angle = 0;
    float max = 0;

    for (int i = 0; i < img_mag.size().width; i++)
    for (int j = 0; j < img_mag.size().height; j++)
    {
        float a = img_angle.at<float>(i, j);
        float m = img_mag.at<float>(i, j);

        if (m > max)
        {
            max = m;
            angle = a;
        }
    }
    return angle;
}

