#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void minHeapify(int* array, int size, int index) {
    int smallest = index;
    int left = 2 * index + 1;
    int right = 2 * index + 2;

    if (left < size && array[left] < array[smallest])
        smallest = left;

    if (right < size && array[right] < array[smallest])
        smallest = right;

    if (smallest != index) {
        swap(&array[index], &array[smallest]);
        minHeapify(array, size, smallest);
    }
}

int kthSmallest(int* array, int size, int k) {
    // Build a min-heap from the first k elements.
    for (int i = k / 2 - 1; i >= 0; i--) {
        minHeapify(array, k, i);
    }

    // Continue the process for the remaining elements.
    for (int i = k; i < size; i++) {
        if (array[i] < array[0]) {
            swap(&array[i], &array[0]);
            minHeapify(array, k, 0);
        }
    }

    // The kth smallest element is at the root of the min-heap.
    return array[k-1];
}

int main() {
    int array[] = {3, 2, 1, 15, 5, 4, 45};
    int size = sizeof(array) / sizeof(array[0]);
    int k = 4; // Change this to the desired kth value.

    if (k > 0 && k <= size) {
        int kthSmallestElement = kthSmallest(array, size, k);
        printf("The %dth smallest element is: %d\n", k, kthSmallestElement);
    } else {
        printf("Invalid value of k\n");
    }

    return 0;
}


// #include <stdio.h>
// #include <stdlib.h>

// typedef struct MinHeap {
//     int *array;
//     int capacity;
//     int size;
// } MinHeap;

// MinHeap* createMinHeap(int capacity) {
//     MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
//     minHeap->capacity = capacity;
//     minHeap->size = 0;
//     minHeap->array = (int*)malloc(capacity * sizeof(int));
//     return minHeap;
// }

// void swap(int *a, int *b) {
//     int temp = *a;
//     *a = *b;
//     *b = temp;
// }

// void minHeapify(MinHeap* minHeap, int index) {
//     int smallest = index;
//     int left = 2 * index + 1;
//     int right = 2 * index + 2;

//     if (left < minHeap->size && minHeap->array[left] < minHeap->array[smallest])
//         smallest = left;

//     if (right < minHeap->size && minHeap->array[right] < minHeap->array[smallest])
//         smallest = right;

//     if (smallest != index) {
//         swap(&minHeap->array[index], &minHeap->array[smallest]);
//         minHeapify(minHeap, smallest);
//     }
// }

// int extractMin(MinHeap* minHeap) {
//     if (minHeap->size == 0) {
//         printf("Heap is empty\n");
//         return -1;
//     }

//     if (minHeap->size == 1) {
//         minHeap->size--;
//         return minHeap->array[0];
//     }

//     int root = minHeap->array[0];
//     minHeap->array[0] = minHeap->array[minHeap->size - 1];
//     minHeap->size--;
//     minHeapify(minHeap, 0);

//     return root;
// }

// int getKthSmallest(MinHeap* minHeap) {
//     // The kth smallest element is at the root of the minHeap.
//     if (minHeap->size < minHeap->capacity){
//         minHeap->capacity = minHeap->size;
//     }

//     for(int i = 0; i < minHeap->capacity; i++){
//         int root = minHeap->array[0];
//         minHeap->array[0] = minHeap->array[minHeap->size - 1];
//         minHeap->size--;
//         minHeapify(minHeap, 0);
//     }

//     return minHeap->array[0];
// }

// void insert(MinHeap* minHeap, int key) {
//     if (minHeap->size < minHeap->capacity) {
//         int i = minHeap->size;
//         minHeap->size++;
//         minHeap->array[i] = key;

//         int parent = (i - 1) / 2;

//         while (i > 0 && minHeap->array[parent] > minHeap->array[i]) {
//             swap(&minHeap->array[i], &minHeap->array[parent]);
//             i = parent;
//             parent = (i - 1) / 2;
//         }
//     }
//     else if (key < minHeap->array[0]) {
//         // If the key is smaller than the current smallest in the minHeap,
//         // replace the smallest element (root) with the new key and heapify.
//         minHeap->array[0] = key;
//         minHeapify(minHeap, 0);
//     }
// }

// int main() {
//     MinHeap* minHeap = createMinHeap(3);
//     insert(minHeap, 3);
//     insert(minHeap, 2);
//     insert(minHeap, 1);
//     insert(minHeap, 15);
//     insert(minHeap, 5);
//     insert(minHeap, 4);
//     insert(minHeap, 45);

//     printf("Minimum element: %d\n", getKthSmallest(minHeap));

//     return 0;
// }