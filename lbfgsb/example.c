#include <stdio.h>
#include "wrapper.h"
 
int main(void)
{
    puts("This is a shared library test...");
    foo();
    puts("This is the end...");
    return 0;
}
