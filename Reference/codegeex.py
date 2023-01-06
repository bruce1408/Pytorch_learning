#write a quick sort function in python which is efficient and easy to understand
import math
#find the median of a list of numbers
#(i.e. find the middle number and partition the list around that number)

def partition(a, low, high):
    """
    quick sort funciton 
    :param a: array to be sorted
    """
    lowest = low
    highest = high

    a_pivot = a[low]
    i = low + 1
    #swap the pivot with the element at the first position
    while i <= highest:
        if a[i] < a_pivot:
            #swap the ith element with the lowest element
            lowest = lowest + 1
            a[i], a[lowest] = a[lowest], a[i]
            i = i + 1
        elif a[i] > a_pivot:
            #swap the ith element with the highest element
            highest = highest - 1
            a[i], a[highest] = a[highest], a[i]
        else:
            #skip the ith element
            i = i + 1   

    a[low], a[lowest] = a[lowest], a[low]
    return lowest
#recursive function to sort a given array

def quick_sort(a, low, high):
    #base case
    if low >= high:
        return
    #find the pivot position
    mid = partition(a, low, high)
    #recursively sort the left side
    quick_sort(a, low, mid-1)
    #recursively sort the right side
    quick_sort(a, mid+1, high)


def quick_sort_recursion(a, low, high):
    """
    recursive quick sort
    :param a: array to be sorted
    """
    if low >= high:
        return
    #find the pivot position
    mid = partition(a, low, high)
    #recursively sort the left side
    quick_sort_recursion(a, low, mid-1)
    #recursively sort the right side
    quick_sort_recursion(a, mid+1, high)
    

#read a file line by line

def get_file_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line

#read a file line by line and concatenate
def get_file_lines_concat(filename):
    with open(filename) as f:
        
