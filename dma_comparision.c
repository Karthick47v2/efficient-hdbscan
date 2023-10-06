#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int R = 5000, C = 5000;

void p1()
{
    int r = R, c = C;
    int *ptr = malloc((r * c) * sizeof(int));

    for (int i = 0; i < r * c; i++)
        ptr[i] = i + 1;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            i = i;

    free(ptr);
}

void p2()
{
    int r = R, c = C, i, j, count;
    int *arr[r];
    for (i = 0; i < r; i++)
        arr[i] = (int *)malloc(c * sizeof(int));
    count = 0;
    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            arr[i][j] = ++count; // Or *(*(arr+i)+j) = ++count

    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            i = i;
    for (int i = 0; i < r; i++)
        free(arr[i]);
}

void p3()
{
    int r = R, c = C, i, j, count;
    int **arr = (int **)malloc(r * sizeof(int *));
    for (i = 0; i < r; i++)
        arr[i] = (int *)malloc(c * sizeof(int));
    count = 0;
    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            arr[i][j] = ++count; // OR *(*(arr+i)+j) = ++count
    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            i = i;
    for (int i = 0; i < r; i++)
        free(arr[i]);

    free(arr);
}

void p4()
{
    int r = R, c = C, len = 0;
    int *ptr, **arr;
    int count = 0, i, j;

    len = sizeof(int *) * r + sizeof(int) * c * r;
    arr = (int **)malloc(len);

    ptr = (int *)(arr + r);

    for (i = 0; i < r; i++)
        arr[i] = (ptr + c * i);

    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            arr[i][j] = ++count; // OR *(*(arr+i)+j) = ++count

    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            i = i;
}

void p5()
{
    int row = R, col = C, i, j, count;
    int(*arr)[row][col] = malloc(sizeof *arr);
    count = 0;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            (*arr)[i][j] = ++count;

    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            i = i;

    free(arr);
}

void p6()
{
    int row = R, col = C, i, j, count;

    int(*arr)[col] = calloc(row, sizeof *arr);

    count = 0;
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            arr[i][j] = ++count;

    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
            i = i;

    free(arr);
}

int main()
{
    clock_t start, end;
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p1();
    }
    printf("P1 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p2();
    }
    printf("P2 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p3();
    }
    printf("P3 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p4();
    }
    printf("P4 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p5();
    }
    printf("P5 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);
    start = clock();
    for (int i = 0; i < 5; i++)
    {
        p6();
    }
    printf("P6 : %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    return 0;
}

/**
 * @brief
 *
 *
 * FOR R,C = 10
 *
P1 : 0.000040
P2 : 0.000021
P3 : 0.000020
P4 : 0.000018
P5 : 0.000012
P6 : 0.000013
 *

 For R,C = 100

P1 : 0.000918
P2 : 0.001026
P3 : 0.001043
P4 : 0.001141
P5 : 0.000195
P6 : 0.000202

 For R, C = 1000
 P1 : 0.049703
P2 : 0.021999
P3 : 0.022050
P4 : 0.029102
P5 : 0.021981
P6 : 0.021825


for R, C = 10000
P1 : 2.651322
P2 : 2.788159
P3 : 2.790241
P4 : 2.700808
P5 : 2.582411
P6 : 2.665806

for R, C = 50000


 */