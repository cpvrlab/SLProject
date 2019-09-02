#include "matching.h"
#include "app.h"


void match_keypoints_0(std::vector<int> &indexes,
                       std::vector<cv::KeyPoint> &kps1, std::vector<Descriptor> &desc1, 
                       std::vector<cv::KeyPoint> &kps2, std::vector<Descriptor> &desc2,
                       float thres)
{
    std::vector<int> distances;
    for (int i = 0; i < kps2.size(); i++)
    {
        indexes.push_back(-1);
        distances.push_back(INT_MAX);
    }

    for (int i = 0; i < kps1.size(); i++)
    {
        int min_dist       = INT_MAX;
        int min_dist2      = INT_MAX;
        int min_dist_index = -1;
        int min_dist2_index = -1;
        
        for (int j = 0; j < kps2.size(); j++)
        {
            int dist = hamming_distance(desc1[i], desc2[j]);

            if (dist < min_dist)
            {
                min_dist2      = min_dist;
                min_dist       = dist;
                min_dist_index = j;
            }
            else if (dist < min_dist2)
            {
                min_dist2 = dist;
                min_dist2_index = j;
            }
        }

        if (min_dist > thres)
            continue;

        //If there is no match yet for this point or the current point is best than the previous, add the matching
        if (indexes[min_dist_index] == -1 || distances[min_dist_index] > min_dist)
        {
            indexes[min_dist_index] = i;
            distances[min_dist_index] = min_dist;
        }
        else 
        {
            //Try to add its second best distance
            if (indexes[min_dist2_index] == -1 || distances[min_dist2_index] > min_dist2)
            {
                indexes[min_dist2_index] = i;
                distances[min_dist2_index] = min_dist2;
            }
        }
    }
}

#define HISTO_LENGTH 30

void match_keypoints_1(std::vector<int> &indexes,
                       std::vector<cv::KeyPoint> &kps1, std::vector<Descriptor> &desc1, 
                       std::vector<cv::KeyPoint> &kps2, std::vector<Descriptor> &desc2,
                       bool check_orientation,
                       float factor,
                       float nnratio,
                       float thres)
{
    std::vector<int> distances;
    std::vector<int> rot_hist[HISTO_LENGTH];

    for (int i = 0; i < kps2.size(); i++)
    {
        indexes.push_back(-1);
        distances.push_back(INT_MAX);
    }
    
    for (int i = 0; i < kps1.size(); i++)
    {
        cv::KeyPoint kp1    = kps1[i];

        int bestDist  = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx  = -1;
        int bestIdx2  = -1;

        for (int j = 0; j < kps2.size(); j++)
        {
            int dist = hamming_distance(desc1[i], desc2[j]);

            if (distances[j] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist  = dist;
                bestIdx2  = bestIdx;
                bestIdx  = j;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
                bestIdx2 = j;
            }
        }

        if (bestDist <= thres)
        {
            //Test if point has similar position
            float x = kps2[bestIdx].pt.x - kps2[bestIdx2].pt.x;
            float y = kps2[bestIdx].pt.y - kps2[bestIdx2].pt.y;
            float d = sqrt(x * x + y * y);

#if MERGE_SIMILAR_LOCATION == 1
            if (bestDist < nnratio * (float)bestDist2 || (d < 0.1 && kps2[bestIdx].octave != kps2[bestIdx2].octave))
#else
            if (bestDist < nnratio * (float)bestDist2)
#endif
            {
                indexes[bestIdx]          = i;
                distances[bestIdx]        = bestDist;

                if (check_orientation)
                {
                    float rot = kps1[i].angle - kps2[bestIdx].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    rot_hist[bin].push_back(bestIdx);
                }
            }
        }
    }

    if (check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        compute_three_maxima(rot_hist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (int j = 0; j < rot_hist[i].size(); j++)
            {
                int idx = rot_hist[i][j];
                if (indexes[idx] >= 0)
                {
                    indexes[idx] = -1;
                }
            }
        }
    }
}



