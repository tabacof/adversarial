#include "wrapper.h"

#include <stdio.h>
#include <math.h>

double f_eval(int n, double *x)
{
	int i;
	double f = .25*(x[0]-1.0)*(x[0]-1.0);
	double power;
	for (i = 1; i < n; i++) {
		power = x[i] - x[i - 1]*x[i - 1];
		power *= power;
		f += power;
	}
	return 4.0*f;
}
         
void g_eval(int n, double *x, double *g)
{
	int i;
	double t1, t2;
	t1 = x[1] - x[0]*x[0];
	g[0] = 2.0*(x[0] - 1.0) - 16.0*x[0]*t1;
	for (i = 1; i < n - 1; i++) {
		t2 = t1;
		t1 = x[i + 1]-x[i]*x[i];
		g[i] = 8*t2 - 16.0*x[i]*t1;
	}
	g[n - 1] = 8.0*t1;
}

         
void foo(void)
{
	char task[61];
	int lsave[4];
	int len_task, n, m, iprint, nbd[NMAX], iwa[3*NMAX], isave[44];
	double f, factr, pgtol, x[NMAX], l[NMAX], u[NMAX], g[NMAX], dsave[29];
	double wa[2*MMAX*NMAX + 5*NMAX + 11*MMAX*MMAX + 8*MMAX];
	int i;
		
	iprint = 1;
	factr=1.0e+7;
    pgtol=1.0e-5;
      
	n = 25;
	m = 5;
     
    for (i = 0; i < n; i +=2) {
		nbd[i] = 2;
		l[i] = 1.0;
		u[i] = 100.0;
	}
	
    for (i = 1; i < n; i +=2) {
		nbd[i] = 2;
		l[i] = -100.0;
		u[i] = 100.0;
	}

    for (i = 0; i < n; i++) {
		x[i] = 3.0;
	}

	strncpy(task, "START                                                       ", 61);
	len_task = strlen(task);
	i = 0;
	do {
		if (strncmp(task, "FG", 2) == 0 || strncmp(task, "START", 5) == 0) {
			f = f_eval(n, x);
			g_eval(n, x, g);
		}
		setulb_(&n,&m,x,l,u,nbd,&f,g,&factr,&pgtol,wa,iwa,task,&len_task,&iprint,lsave,isave,dsave);
		
	} while (strncmp(task, "FG", 2) == 0 || strncmp(task, "NEW_X", 5) == 0);
}
