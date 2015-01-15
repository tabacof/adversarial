#ifndef wrapper_h__
#define wrapper_h__

#include <string.h>

#define NMAX 1024
#define MMAX 17

extern void setulb_(int *n, int *m,double *x,double *l,double *u,int *nbd, double *f,
					double *g,double *factr,double *pgtol,double *wa,int *iwa,
					char *task, int *len_task, int *iprint, int *lsave,int *isave,double *dsave);

extern void foo(void);

#endif  // wrapper_h__
