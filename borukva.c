/**
 * @file borukva.c
 * @author Karthick T. Sharma
 * @brief Borukva's algorithm to find Minimum Spanning Tree (MST)
 * @date 2022-11-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/**
 * @brief Structure to hold details of graph's edges
 *
 */
struct Edge
{
    long long src; /**< Source of edge */
    long long dst; /**< Destination of edge */
    double weight; /**< Weight of the edge */
};

/**
 * @brief Structure to hold graph elements
 *
 */
struct Graph
{
    long long V;        /**< Number of vertexes */
    long long E;        /**< Number of edges*/
    struct Edge *edges; /**< Array of edges to represent graph*/
};

/**
 * @brief Structre to hold subset (for union-find op - in disjoint sets)
 *
 */
struct Subset
{
    long long parent; /**< Root of disjoint set*/
    long long rank;   /**< Rank of disjoint set*/
};

/**
 * @brief Find root/parent index for disjoint set.
 *
 * Traverse through parent of elements to find root parent of disjoint set and use path
 * compression technique eliminate traversing (first time).
 *
 * @param subsets Disjoint sets
 * @param i Element/Vertex
 * @return int Parent of disjoint set
 */
int findParentIdx(struct Subset subsets[], long long i)
{
    while (subsets[i].parent != i)
    {
        subsets[i].parent = subsets[subsets[i].parent].parent;
        i = subsets[i].parent;
    }

    return subsets[i].parent;
}

/**
 * @brief Perform union operation between two disjoint sets.
 *
 * Find root parent of both disjoint sets and unify based on rank of the disjoint sets.
 *
 * @param subsets Disjoint sets
 * @param x Element/Vertex of one disjoint set
 * @param y Element/Vertex of another disjoint set
 */
void unifySets(struct Subset subsets[], long long x, long long y)
{
    // Find parent of disjoint sets
    long long xRoot = findParentIdx(subsets, x);
    long long yRoot = findParentIdx(subsets, y);

    if (xRoot != yRoot)
    {
        // Union by rank
        if (subsets[xRoot].rank > subsets[yRoot].rank)
        {
            subsets[yRoot].parent = xRoot;
        }
        else if (subsets[yRoot].rank > subsets[xRoot].rank)
        {
            subsets[xRoot].parent = yRoot;
        }
        // If rank same then make on as root and increment rank
        else
        {
            subsets[yRoot].parent = xRoot;
            subsets[xRoot].rank++;
        }
    }
}

/**
 * @brief Constructor styled function to generate graph
 *
 * @param graph Pointer to graph's memory location
 * @param V Number of vertices
 * @param E Number of edges
 */
void generateGraph(struct Graph *graph, long long V, long long E)
{
    graph->V = V;
    graph->E = E;
    graph->edges = (struct Edge *)calloc(E, sizeof(struct Edge));

    if (graph->edges == NULL)
    {
        fprintf(stderr, "\nError: Out of Memory\n\n");
        exit(-1);
    }
}

/**
 * @brief Construct minimum spanning tree from graph
 *
 * Use Borukva's MST algorithm to construct minimum spanning tree. Initially set all vertex as
 * seperate disjoint sets, find minimum cost edge for all vertex and unionionize those sets. Do
 * until only one disjoint set remains (minimum spanning tree)
 *
 * @param graph graph that needed for MST
 */
void borukva(struct Graph *restrict graph)
{
    long long V = graph->V;
    long long E = graph->E;
    struct Edge *restrict edges = graph->edges;
    struct Subset *restrict subsets = (struct Subset *)malloc(V * sizeof(struct Subset));

    // index of nearest edges
    long long *restrict nearest = (long long *)calloc(V, sizeof(long long));
    // --------------------------------------------------------------- check mem-computation tradeoff
    long long *restrict rootParent = (long long *)calloc(V, sizeof(long long));

    if (subsets == NULL || rootParent == NULL)
    {
        fprintf(stderr, "\nError: Out of Memory\n\n");
        exit(-1);
    }

    // initialize each element as separate disjoint set
    for (int i = 0; i < V; i++)
    {
        subsets[i].parent = i;
        subsets[i].rank = 0;
        rootParent[i] = -1;
        nearest[i] = -1;
    }

    // Number of trees - initially all vertexes are considered as separate trees / disjoint sets
    long long noOfTrees = V;
    // MST cost - used for debugging
    double mstCost = 0;

    // Unify all disjoint sets
    while (noOfTrees > 1)
    {
        // update cheapest edge for each vertex
        for (long long i = 0; i < E; i++)
        {
            // if 2 elements have same parent at a point then it will always have same parent so
            // no need to compute
            if (rootParent[edges[i].src] != -1 &&
                (rootParent[edges[i].src] == rootParent[edges[i].dst]))
            {
                continue;
            }

            // find parent
            long long root1 = findParentIdx(subsets, edges[i].src);
            long long root2 = findParentIdx(subsets, edges[i].dst);

            // ignore if same parent (same disjoint set)
            if (root1 == root2)
            {
                rootParent[edges[i].src] = root1;
                rootParent[edges[i].dst] = root2;
                continue;
            }

            // update nearest edge
            if (nearest[root1] == -1 || edges[nearest[root1]].weight > edges[i].weight)
            {
                nearest[root1] = i;
            }
            if (nearest[root2] == -1 || edges[nearest[root2]].weight > edges[i].weight)
            {
                nearest[root2] = i;
            }
        }

        // add to final MST
        for (long long i = 0; i < V; i++)
        {
            if (nearest[i] == -1)
            {
                continue;
            }

            long long root1 = findParentIdx(subsets, edges[nearest[i]].src);
            long long root2 = findParentIdx(subsets, edges[nearest[i]].dst);

            if (root1 == root2)
            {
                nearest[i] = -1;
                continue;
            }

            mstCost += edges[nearest[i]].weight;
            nearest[i] = -1;

            // unify trees/disjoint sets
            unifySets(subsets, root1, root2);
            noOfTrees--;
        }
    }

    free(nearest);
    free(subsets);
    free(rootParent);
    printf("Cost of MST : %lf\n", mstCost);
}

/**
 * @brief Main function to drive program
 *
 * @return int termination status code
 */
int main()
{
    FILE *file = fopen("test-1000000.txt", "r");

    if (!file)
    {
        fprintf(stderr, "\nCan't open file test.txt\n");
    }

    // Test using synthetic data
    long long V = 0;
    long long E = 0;

    char line[100];

    while (fgets(line, sizeof(line), file))
    {
        if (V == 0)
        {
            V = atoll(line);
        }
        else if (E == 0)
        {
            E = atoll(line);
            break;
        }
    }

    int iterations = 10;
    double startTime = omp_get_wtime();

    struct Graph graph;
    generateGraph(&graph, V, E);

    printf("V: %lld \nE: %lld \n", V, E);

    long long k = 0;

    while (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, " ");

        graph.edges[k].src = atoll(token);
        token = strtok(NULL, " ");
        graph.edges[k].dst = atoll(token);
        token = strtok(NULL, " ");
        graph.edges[k].weight = strtod(token, NULL);
        k++;
    }

    fclose(file);

    for (int i = 0; i < iterations; i++)
    {
        borukva(&graph);
    }

    free(graph.edges);

    printf("%fs\n", (omp_get_wtime() - startTime) / iterations);
    return 0;
}

// gcc -pg -fopenmp -o main borukva.c
// ./main
// gprof main gmon.out > report.txt
