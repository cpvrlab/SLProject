#include "tools.h"

void init_patch(std::vector<int> &umax)
{
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int          v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2  = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
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
        float scale = p.scale_factors[level];

        for (auto kp = kps.begin(); kp != kps.end(); ++kp)
            kp->pt *= scale;

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

cv::Mat to_grayscale(cv::Mat &img)
{
    int from_to[] = {0, 0};
    cv::Mat img_gray = cv::Mat(img.rows, img.cols, CV_8UC1);
    cv::mixChannels(&img, 1, &img_gray, 1, from_to, 1);
    return img_gray;
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

cv::KeyPoint get_middle_keypoint(cv::Mat image)
{
    float x = image.cols * 0.5;
    float y = image.rows * 0.5;
   
    cv::KeyPoint kp = cv::KeyPoint(x, y, 7);
    kp.angle = 0;
    kp.octave = 0;

    return kp;
}

void keypoint_angle(const cv::Mat& image, cv::KeyPoint &kp, const std::vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(kp.pt.y), cvRound(kp.pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d     = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    kp.angle = atan2((float)m_01, (float)m_10);
}

