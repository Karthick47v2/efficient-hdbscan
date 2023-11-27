#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
// gcc -shared -o graph.so graph.c -O3 -march=native -lm -fopenmp -fPIC

typedef struct Subset
{
    unsigned long long parent;
    unsigned long long rank;
} Subset;

typedef struct Edge
{
    unsigned long long src;
    unsigned long long dst;
} Edge;

typedef struct MSTEdge
{
    unsigned long long src;
    unsigned long long dst;
    double weight;
} MSTEdge;

typedef struct Cluster
{
    unsigned long long *points;
    unsigned long long size;
    unsigned long long root;
    bool isCluster;
    double stability;
    double birthStability;
    double childrenStability;
    unsigned long long stabilityPoints;

} Cluster;

unsigned long long findParentIdx(Subset *subsets, unsigned long long i)
{
    while (subsets[i].parent != i)
    {
        subsets[i].parent = subsets[subsets[i].parent].parent;
        i = subsets[i].parent;
    }
    return subsets[i].parent;
}

void unifySets(Subset *subsets, unsigned long long x, unsigned long long y)
{
    unsigned long long xRoot = findParentIdx(subsets, x);
    unsigned long long yRoot = findParentIdx(subsets, y);

    if (xRoot != yRoot)
    {
        if (subsets[xRoot].rank > subsets[yRoot].rank)
        {
            subsets[yRoot].parent = xRoot;
        }
        else if (subsets[yRoot].rank > subsets[xRoot].rank)
        {
            subsets[xRoot].parent = yRoot;
        }
        else
        {
            subsets[yRoot].parent = xRoot;
            subsets[xRoot].rank++;
        }
    }
}

int compareEdges(const void *a, const void *b)
{
    return ((struct MSTEdge *)a)->weight > ((struct MSTEdge *)b)->weight;
}

int *constructHierarchy(struct MSTEdge *edges, unsigned long long numEdges, int minClusterPoints)
{
    qsort(edges, numEdges, sizeof(struct MSTEdge), compareEdges);
    unsigned long long numPoints = numEdges + 1;

    struct Subset *subsets = (struct Subset *)malloc(numPoints * sizeof(struct Subset));
    struct Cluster *clusters = (struct Cluster *)malloc(numPoints * 2 * sizeof(struct Cluster));

    for (unsigned long long i = 0; i < numPoints; i++)
    {
        subsets[i].parent = i;
        subsets[i].rank = 0;
    }

    for (unsigned long long i = 0; i < numPoints; i++)
    {
        clusters[i].points = (unsigned long long *)malloc(sizeof(unsigned long long));
        clusters[i].points[0] = i;
        clusters[i].size = 1;
        clusters[i].root = i;

        clusters[i].stability = 0;
        clusters[i].stabilityPoints = 0;
        clusters[i].childrenStability = 0;
        clusters[i].birthStability = 0;
    }

    for (unsigned long long i = numPoints; i < 2 * numPoints; i++)
    {
        clusters[i].stability = 0;
        clusters[i].stabilityPoints = 0;
        clusters[i].childrenStability = 0;
        clusters[i].birthStability = 0;
    }

    unsigned long long clusterID = numPoints;
    for (unsigned long long i = 0; i < numEdges; i++)
    {
        unsigned long long rootSrc = findParentIdx(subsets, edges[i].src);
        unsigned long long rootDst = findParentIdx(subsets, edges[i].dst);

        if (rootSrc != rootDst)
        {
            while (clusters[rootSrc].root >= numPoints)
            {
                rootSrc = clusters[rootSrc].root;
            }

            while (clusters[rootDst].root >= numPoints)
            {
                rootDst = clusters[rootDst].root;
            }
            unsigned long long newClusterSize = clusters[rootSrc].size + clusters[rootDst].size;

            clusters[rootSrc].isCluster = clusters[rootSrc].size >= minClusterPoints;
            clusters[rootDst].isCluster = clusters[rootDst].size >= minClusterPoints;

            if (clusters[rootSrc].isCluster == clusters[rootDst].isCluster)
            {
                unsigned long long *newClusterPoints = (unsigned long long *)malloc(newClusterSize * sizeof(unsigned long long));
                memcpy(newClusterPoints, clusters[rootSrc].points, clusters[rootSrc].size * sizeof(unsigned long long));
                memcpy(newClusterPoints + clusters[rootSrc].size, clusters[rootDst].points, clusters[rootDst].size * sizeof(unsigned long long));

                clusters[clusterID].stability += (clusters[rootSrc].size + clusters[rootDst].size) * (1 / edges[i].weight);
                clusters[clusterID].stabilityPoints += (clusters[rootSrc].size + clusters[rootDst].size);

                if (!clusters[rootSrc].isCluster)
                {
                    free(clusters[rootSrc].points);
                    free(clusters[rootDst].points);

                    clusters[rootSrc].size = 0;
                    clusters[rootDst].size = 0;
                }
                else
                {
                    clusters[rootSrc].birthStability = (1 / edges[i].weight);
                    clusters[rootDst].birthStability = (1 / edges[i].weight);
                }

                clusters[clusterID].root = clusters[rootSrc].root;
                clusters[rootSrc].root = clusterID;
                clusters[rootDst].root = clusterID;

                clusters[clusterID].size = newClusterSize;
                clusters[clusterID].points = newClusterPoints;

                clusterID++;
            }
            else
            {
                unsigned long long validCluster, invalidCluster;
                if (clusters[rootSrc].isCluster)
                {
                    validCluster = rootSrc;
                    invalidCluster = rootDst;
                }
                else
                {
                    validCluster = rootDst;
                    invalidCluster = rootSrc;
                }

                clusters[validCluster].stability += clusters[invalidCluster].size * (1 / edges[i].weight);
                clusters[validCluster].stabilityPoints += clusters[invalidCluster].size;

                clusters[validCluster].size = newClusterSize;
                clusters[validCluster].points = (unsigned long long *)realloc(clusters[validCluster].points, newClusterSize * sizeof(unsigned long long));
                memcpy(clusters[validCluster].points + clusters[validCluster].size - clusters[invalidCluster].size, clusters[invalidCluster].points, clusters[invalidCluster].size * sizeof(unsigned long long));

                clusters[invalidCluster].size = 0;
                free(clusters[invalidCluster].points);
                clusters[invalidCluster].root = validCluster;
            }

            unifySets(subsets, edges[i].src, edges[i].dst);
        }
    }

    // #pragma omp parallel for schedule(static, 2)
    for (unsigned long long i = numPoints; i < clusterID; i++)
    {
        if (clusters[i].isCluster)
        {
            clusters[i].stability -= clusters[i].birthStability * clusters[i].stabilityPoints;
            clusters[clusters[i].root].childrenStability += clusters[i].stability;
        }
    }

    for (unsigned long long i = numPoints; i < clusterID; i++)
    {
        if (clusters[i].isCluster)
        {
            unsigned long long temp = i;
            while ((clusters[temp].size != numPoints) && (clusters[temp].stability < clusters[i].childrenStability))
            {
                if (temp == i)
                {
                    clusters[clusters[temp].root].childrenStability += (clusters[temp].childrenStability - clusters[temp].stability);
                    clusters[temp].stability = clusters[temp].childrenStability;
                    clusters[temp].isCluster = false;
                }
                temp = clusters[temp].root;
            }

            if (clusters[i].birthStability < 1e-6)
            {
                clusters[i].isCluster = false;
            }

            if (clusters[i].stability < 1e-6)
            {
                clusters[i].isCluster = false;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (unsigned long long i = numPoints; i < clusterID; i++)
    {
        if (clusters[i].isCluster)
        {
            unsigned long long temp = i;

            while (clusters[temp].size != numPoints)
            {
                if (clusters[clusters[temp].root].isCluster)
                {
                    clusters[i].isCluster = false;
                }
                temp = clusters[temp].root;
            }
        }

        // if (clusters[i].isCluster)
        // {
        //     printf("ClusterID %lld\n", i);
        //     printf("Cluster size %lld\n", clusters[i].size);
        //     printf("IsCluster %d\n", clusters[i].isCluster);
        //     printf("Birth Stability %f\n", clusters[i].birthStability);
        //     printf("Stability %f\n", clusters[i].stability);
        //     printf("Children Stability %f\n", clusters[i].childrenStability);
        //     printf("Root %lld\n", clusters[i].root);
        //     // printf("Points: ");
        //     // for (unsigned long long j = 0; j < clusters[i].size; j++)
        //     // {
        //     //     printf(" %lld", clusters[i].points[j]);
        //     // }
        //     // clusterDataPoints += clusters[i].size;
        //     // numClusers++;
        //     printf("\n\n");
        // }
    }

    // printf("\n\n Noise: %lld\n\n", numPoints - clusterDataPoints);

    int *cluster_labels = (int *)malloc(numPoints * sizeof(int));
    memset(cluster_labels, -1, numPoints * sizeof(int));
    int label = 0;
    for (unsigned long long i = numPoints; i < clusterID; i++)
    {
        if (clusters[i].isCluster)
        {
            for (unsigned long long j = 0; j < clusters[i].size; j++)
            {
                cluster_labels[clusters[i].points[j]] = label;
            }
            label++;
        }
    }

    // Clean up memory
    free(subsets);
    return cluster_labels;
}

void calc_mst(unsigned long long V, double *data, int minClusterPoints)
{
    Subset *subsets = malloc(V * sizeof(Subset));
    Edge *nearest = calloc(V, sizeof(Edge));
    unsigned long long *rootParent = (unsigned long long *)calloc(V, sizeof(unsigned long long));

    MSTEdge *mstEdges = calloc(V - 1, sizeof(MSTEdge));

    // initialize each element as a separate disjoint set
    // #pragma omp parallel for schedule(static, 32)
    for (unsigned long long i = 0; i < V; i++)
    {
        subsets[i].parent = i;
        subsets[i].rank = 0;
        rootParent[i] = -1;
        nearest[i].src = -1;
    }

    unsigned long long noOfTrees = V;

    unsigned long long v = 0;

    while (noOfTrees > 1)
    {

        for (unsigned long long i = 0; i < V; i++)
        {
            for (unsigned long long j = i + 1; j < V; j++)
            {
                if (data[i * V + j] != 0)
                {
                    // if 2 elements have the same parent at a point, then they will always have the same parent so
                    // no need to compute
                    if (rootParent[i] != -1 &&
                        (rootParent[i] == rootParent[j]))
                    {
                        continue;
                    }

                    // find parent
                    unsigned long long root1 = findParentIdx(subsets, i);
                    unsigned long long root2 = findParentIdx(subsets, j);

                    // ignore if the same parent (same disjoint set)
                    if (root1 == root2)
                    {
                        rootParent[i] = root1;
                        rootParent[j] = root2;
                        continue;
                    }

                    if (nearest[root1].src == -1 || data[nearest[root1].src * V + nearest[root1].dst] > data[i * V + j])
                    {
                        nearest[root1].src = i;
                        nearest[root1].dst = j;
                    }
                    if (nearest[root2].src == -1 || data[nearest[root2].src * V + nearest[root2].dst] > data[i * V + j])
                    {
                        nearest[root2].src = i;
                        nearest[root2].dst = j;
                    }
                }
            }
        }

        for (unsigned long long i = 0; i < V; i++)
        {
            if (nearest[i].src == -1)
            {
                continue;
            }

            unsigned long long root1 = findParentIdx(subsets, nearest[i].src);
            unsigned long long root2 = findParentIdx(subsets, nearest[i].dst);

            if (root1 == root2)
            {
                nearest[i].src = -1;
                continue;
            }

            mstEdges[v].src = nearest[i].src;
            mstEdges[v].dst = nearest[i].dst;
            mstEdges[v].weight = data[nearest[i].src * V + nearest[i].dst];
            nearest[i].src = -1;

            // unify trees/disjoint sets
            unifySets(subsets, root1, root2);
            noOfTrees--;
            v++;
        }
    }
    free(nearest);
    free(subsets);
    free(rootParent);

    constructHierarchy(mstEdges, V - 1, minClusterPoints);
}
