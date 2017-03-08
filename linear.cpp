#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <omp.h>
//#include <Eigen/Core>
#include <atomic>
#include "linear.h"
#include "tron.h"
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		wa[i] = sparse_operator::dot(s, xi);
		
		wa[i] = C[i]*D[i]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);
		
		wa[i] = C[I[i]]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

// A coordinate descent algorithm for 
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
// 
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i, 
//  C^m_i = 0 if m != y_i, 
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i 
//
// Given: 
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that 
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;

			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter)
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
// 
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[i], xi, w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			const schar yi = y[i];
			feature_node * const xi = prob->x[i];

			G = yi*sparse_operator::dot(w, xi)-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}


// A coordinate descent algorithm for 
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
// 
//  where Qij = xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given: 
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012   

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node * const xi = prob->x[i];
			G += sparse_operator::dot(w, xi);

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
				sparse_operator::axpy(d, xi, w);
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for 
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and 
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}
	
	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node * const xi = prob->x[i];
			ywTx = yi*sparse_operator::dot(w, xi);
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					sparse_operator::axpy(d_diff, x, b);
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					sparse_operator::axpy(-w[i], x, b);
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for 
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	double total_time =  0;
	double time_begin = clock();

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}


	while(newton_iter < max_newton_iter)
	{
		double time_begin = clock();
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
          break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		//info("iter %3d  #CD cycles %d\n", newton_iter, iter);

		total_time += (clock()-time_begin)/CLOCKS_PER_SEC;
		double v = 0;
		int nnz = 0;
		for(j=0; j<w_size; j++)
			if(w[j] != 0)
			{
				v += fabs(w[j]);
				nnz++;
			}
		for(j=0; j<l; j++)
			if(y[j] == 1)
				v += C[GETI(j)]*log(1+1/exp_wTx[j]);
			else
				v += C[GETI(j)]*log(1+exp_wTx[j]);
		info("%lf, %lf\n", v, total_time);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// The original proximal Newton method for  
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)
// Modified by Cho (removed the active subset selection trick)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr_new(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn, int max_newton_iter)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
//	int max_newton_iter = 100;
	int max_iter = 1;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};



	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			//XXX Meaning of xjneg_sum
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}
        double total_time = 0.0;
	while(newton_iter < max_newton_iter)
	{
        double newton_beg = clock();

		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<w_size; j++)
		{
			Hdiag[j] = nu; // diagonal of Hessian
			Grad[j] = 0; // gradient

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			
			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
			
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = w_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;
        double inner_beg = clock();

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			// Random permutation for each coordinate descent epoch
			for(j=0; j<w_size; j++)
			{
				int i = j+rand()%(w_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<w_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			// if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			//	break;

			QP_Gmax_old = QP_Gmax_new;
		}
        double search_beg = clock();
		if(iter >= max_iter)
			//info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;


                double newton_end = clock();
		//total_runtime += (clock()-timebegin)/CLOCKS_PER_SEC;

		// calculate objective value

		double v = 0;
		int nnz = 0;
		for(j=0; j<w_size; j++)
			if(w[j] != 0)
			{
				v += fabs(w[j]);
				nnz++;
			}
		for(j=0; j<l; j++)
			if(y[j] == 1)
				v += C[GETI(j)]*log(1+1/exp_wTx[j]);
			else
				v += C[GETI(j)]*log(1+exp_wTx[j]);
		total_time += (newton_end-newton_beg) / CLOCKS_PER_SEC;
		info("%lf, %lf\n", v, total_time);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

static void leverage_value(const problem *prob, double *D, double *leverage) {
  // X:
  int l = prob->l;
  int w_size = prob->n;
  double *XTX_diag = new double[w_size];
  double *feature_mean = new double[w_size];
  memset(leverage, 0, l * sizeof(double));
  memset(feature_mean, 0, w_size * sizeof(double));
  memset(XTX_diag, 0, w_size * sizeof(double));
  // calculate the feature_mean
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    while (xi->index != -1) {
      feature_mean[xi->index - 1] += xi->value;
      xi++;
    }
  }
  for (int i = 0; i < w_size; ++i) {
    feature_mean[i] /= l;
  }
  // calculate XTX_diag
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    while (xi->index != -1) {
      XTX_diag[xi->index - 1] += D[i] * (xi->value - feature_mean[xi->index - 1]) * (xi->value - feature_mean[xi->index - 1]);
      //XTX_diag[xi->index - 1] += D[i] * xi->value * xi->value;
      xi++;
    }
  }
  // calculate leverage value
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    while (xi->index != -1) {
      leverage[i] += (xi->value - feature_mean[xi->index - 1]) * (xi->value - feature_mean[xi->index - 1]) / XTX_diag[xi->index - 1];
      xi++;
    }
  }
  // normalize leverage value to 1
  double sum = 0.0;
  for (int i = 0; i < l; ++i) {
    sum += leverage[i];
  }
  for (int i = 0; i < l; ++i) {
    leverage[i] = (leverage[i] / sum);
  }
  delete [] XTX_diag;
  delete [] feature_mean;
}


static void corr_mat(const problem *prob, double *mat) {
  int l = prob->l;
  int w_size = prob->n;
  memset(mat, 0, l * w_size * sizeof(int));
  double *nrm2 = new double[l];
  memset(nrm2, 0, l * sizeof(int));
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    while (xi->index != -1) {
      nrm2[i] += xi->value;
      xi++;
    }
  }
  for (int i = 0; i < l; ++i) {
    mat[i * l + i] = 1.0;
    for (int j = i+1; j < l; ++j) {
      feature_node *xi = prob->x[i];
      feature_node *xj = prob->x[j];
      double sum = 0.0;
      while (xi->index != -1 && xj->index != -1) {
        if (xi->index == xj->index) {
          sum += xi->value * xj->value;
          xi++; xj++;
        }
        else if (xi->index > xj->index) {
          xj++;
        }
        else xi++;
      }
      mat[i * l + j] = sum / nrm2[i];
      mat[l * l + i] = sum / nrm2[j];
    }
  }
  delete [] nrm2;
}

static void leverage_value3(const problem *prob, double *D, double *leverage, double *mat) {
  int l = prob->l;
  int w_size = prob->n;
  memset(leverage, 0, l * sizeof(double));
  for (int i = 0; i < l; ++i) {
    double sum = 0.0;
    for (int j = 0; j < l; ++j) {
      sum += D[j] * mat[i * l + j];
    }
    leverage[i] = sum / D[i];
  }
  // normalize
  double sum = 0.0;
  for (int i = 0; i < l; ++i) {
    sum += leverage[i];
  }
  for (int i = 0; i < l; ++i) {
    leverage[i] /= sum;
  }
}

static void leverage_value2(const problem *prob, double *D, double *leverage) {
  // X:
  int l = prob->l;
  int w_size = prob->n;
  double *feature_mean = new double[w_size];
  memset(leverage, 0, l * sizeof(double));
  memset(feature_mean, 0, w_size * sizeof(double));
  // calculate the feature_mean
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    while (xi->index != -1) {
      feature_mean[xi->index - 1] += xi->value;
      xi++;
    }
  }
  for (int i = 0; i < w_size; ++i) {
    feature_mean[i] /= l;
  }
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    double sum = 0.0;
    for (int j = 0; j < w_size; ++j) {
      if (xi->index - 1 == j) {
        //sum += (xi->value - feature_mean[j]) * (xi->value - feature_mean[j]);
        sum += xi->value * xi->value;
        xi++;
      }
      else {
        //sum += feature_mean[j] * feature_mean[j];
      }
    }
    leverage[i] = D[i] * sum;
  }
  double sum = 0.0;
  for (int i = 0; i < l; ++i) {
    sum += leverage[i];
  }
  for (int i = 0; i < l; ++i) {
    leverage[i] = 0.5 * leverage[i] / sum + 0.5 / l;
    //printf("%lf\n", leverage[i]);
  }
}

static void sample(const double *leverage, int l, int *index, int n) {
  double *cumsum = new double[l];
  // calculate cumsum of leverage, inplace
  cumsum[0] = leverage[0];
  for (int i = 1; i < l; ++i) {
    cumsum[i] = cumsum[i-1] + leverage[i-1];
  }
  cumsum[l-1] = 1.0;
  for (int i = 0; i < n; ++i) {
    double choose = (double)rand() / (1.0 + RAND_MAX);
    // binary search
    int left = 0, right = l-1;
    while (left < right - 1) {
      int pivotal = (left + right) / 2;
      if (cumsum[pivotal] >= choose) {
        right = pivotal;
      }
      else {
        left = pivotal;
      }
    }
    index[i] = right;
  }
  delete [] cumsum;
}

static double prox(double x, double eta, double k) {
  if (x - k > eta) return x - k - eta;
  else if (x - k < -eta) return x - k + eta;
  else return 0.0;
}

static double proxn2(double x, double eta, double k, int n) {
  if (n == 1) return prox(x, eta, k);
  else return prox(proxn2(x, eta, k, n - 1), eta, k);
}

static double proxn(double x, double eta, double k, int n) {
  if (abs(k) <= eta) {
    if (x > n * (eta + k))
      return x - n * (eta + k);
    else if (x > n * (k - eta))
      return 0.0;
    else
      return x - n * (k - eta);
  }
  else if (k > eta) {
    double min_i = (x + eta - k) / (eta + k);
    if (min_i < 0)
      return x - n * (k - eta);
    else if (min_i >= n)
      return x - n * (k + eta);
    else {
      int min_i_ = (int)min_i;
      if (x - k + eta - min_i_ * (k + eta) < 2 * eta)
        return (1 + min_i_ - n) * (k - eta);
      else
        return x - (1 + min_i_) * (k + eta) + (1 + min_i_ - n) * (k - eta);
    }
  }
  else {
    double min_i = (x - n * (k - eta)) / (eta - k);
    if (min_i < 0)
      return x - n * (k - eta);
    else if (min_i >= n)
      return x - n * (eta + k);
    else {
      int min_i_ = (int)min_i;
      if (x + (min_i_ - n) * (k - eta) < 2 * eta)
        return -min_i_ * (eta + k);
      else
        return x + (min_i_ - n) * (k - eta) - 2 * eta - min_i_ * (eta + k);
    }
  }
}

static double recover(double w, int delay, int now, double eta, double k) {
  if (delay == now) return w;
  return proxn(w, eta, k, now - delay);
}

static void solve_l1r_lr_admm(const problem *prob, 
                              double *w, 
                              double eps, 
                              double Cp, 
                              double Cn, 
                              int max_newton_iter) {
  int l = prob->l;
  int w_size = prob->n;
  schar *y = new schar[l];
  
  for (int i = 0; i < l; ++i) {
    if (prob->y[i] > 0) y[i] = 1;
    else y[i] = -1;
  }

  
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_proxsagapp(const problem *prob_col,
                                  double *w,
                                  double eps,
                                  double Cp,
                                  double Cn,
                                  int max_newton_iter,
                                  const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s, newton_iter = 0, iter = 0;
  int saga_outer = 3;
  int max_saga_iter = l * 0.04;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];
  double *D = new double[l];

  double *w_saga = new double[w_size];
  double *xTw_saga = new double[l];
  double *xTw = new double[l];
  double *sum_grad = new double[w_size];
  double *leverage = new double[l];
  int *index = new int[max_saga_iter];
  int *delay = new int[w_size];
  //double eta = 0.0000003;
  double eta = 0.0001;
  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
      w_saga[j] = 0;
      sum_grad[j] = 0;
      delay[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
        y[j] = 1;
    else
        y[j] = -1;
    xTw_saga[j] = 0;
    xTw[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
      feature_node *x = prob_col->x[j];
      while(x->index != -1)
      {
          xTw_saga[x->index-1] += w_saga[j] * x->value;
          x++;
      }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }

  for(j=0; j<l; j++)
  {
      double exp_wTx = exp(xTw_saga[j]);
      double tau_tmp = 1 / (1 + exp_wTx);
      D[j] = C[GETI(j)] * exp_wTx * tau_tmp * tau_tmp;
  }
  // info("Begin newton iteration\n");
  double time = 0.0;
  while (newton_iter < max_newton_iter) {
    double newton_beg = clock();
    // solve with SAGA++
    for (int saga_outer_iter = 0; saga_outer_iter < saga_outer; ++saga_outer_iter) {
    for (int saga_iter = 0; saga_iter < max_saga_iter; ++saga_iter) {
      // randomly choose element \in (0,...,l-1)
      int choose = rand() % l;
      // calculate difference of gradient for element chosen
      double xjTw_saga = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        int idx = x->index - 1;
        w_saga[idx] = recover(w_saga[idx], delay[idx], saga_iter, eta, eta * sum_grad[idx]);
        xjTw_saga += x->value * w_saga[idx];
        x++;
      }
      double tmp = D[choose] * (xjTw_saga - xTw_saga[choose]);
      // update w_saga
      x = prob->x[choose]; // reset x
      while (x->index != -1) {
        int idx = x->index - 1;
        delay[idx] = saga_iter + 1;
        double descent = w_saga[idx] - eta * (sum_grad[idx] + l * x->value * tmp);
        sum_grad[idx] += tmp * x->value;
        if (descent > eta) {
          w_saga[idx] = descent - eta;
        }
        else if (descent < -eta) {
          w_saga[idx] = descent + eta;
        }
        else {
          w_saga[idx] = 0.0;
        }
        x++;
      }
      // update xTw_saga
      xTw_saga[choose] = xjTw_saga;
    }
    for (j = 0; j < w_size; ++j) {
      w_saga[j] = recover(w_saga[j], delay[j], max_saga_iter, eta, eta * sum_grad[j]);
      delay[j] = 0;
    }
     // recalculate
     memset(xTw_saga, 0, sizeof(double) * l);
     memset(sum_grad, 0, sizeof(double) * w_size);
     // re-calculate xTw_saga
     for (j = 0; j < l; ++j) {
       feature_node *x = prob->x[j];
       double sum = 0.0;
       while (x->index != -1) {
         int idx = x->index - 1;
         sum += w_saga[idx] * x->value;
         x++;
       }
       xTw_saga[j] = sum;
     }
     // re-calculate sum_grad
     for (j = 0; j < l; ++j) {
       feature_node *x = prob->x[j];
       double tmp = 0.0;
       tmp = C[GETI(j)];
       if (y[j] > 0) {
         tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
       } else {
         tmp *= (1.0 - 1.0 / (1.0 + exp(xTw_saga[j])));
       }
       while (x->index != -1) {
         sum_grad[x->index - 1] += x->value * tmp;
         x++;
       }
     }
    }
    // update D
    for (j = 0; j < l; ++j) {
      double exp_xTw_saga = exp(xTw_saga[j]);
      D[j] = C[GETI(j)] * exp_xTw_saga / (1.0 + exp_xTw_saga) / (1.0 + exp_xTw_saga);
    }
    time += (clock() - newton_beg) / CLOCKS_PER_SEC;
    int nnz = 0;
    // function value and sparsity
    double value = 0;
    for (j = 0; j < w_size; ++j) {
      if (w_saga[j] != 0) {
        value += fabs(w_saga[j]);
        nnz++;
      }
    }
    for (j = 0; j < l; j++)
      if (y[j] == 1)
        value += C[GETI(j)] * log(1 + 1 / exp(xTw_saga[j]));
      else
        value += C[GETI(j)] * log(1 + exp(xTw_saga[j]));
    info("%lf, %lf\n", value, time);
    newton_iter++;
  }
  delete [] y;
  delete [] D;
  delete [] w_saga;
  delete [] xTw_saga;
  delete [] xTw;
  delete [] sum_grad;
  delete [] leverage;
}


#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_proxsaga(const problem *prob_col,
                                  double *w,
                                  double eps,
                                  double Cp,
                                  double Cn,
                                  int max_newton_iter,
                                  const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s, newton_iter = 0, iter = 0;
  int max_saga_iter = l / 100;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];
  double *D = new double[l];

  double *w_saga = new double[w_size];
  double *xTw_saga = new double[l];
  double *xTw = new double[l];
  double *sum_grad = new double[w_size];
  double *leverage = new double[l];
  int *index = new int[max_saga_iter];
  double eta = 0.00004;

  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
      w_saga[j] = 0;
      sum_grad[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
        y[j] = 1;
    else
        y[j] = -1;
    xTw_saga[j] = 0;
    xTw[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
      feature_node *x = prob_col->x[j];
      while(x->index != -1)
      {
          xTw_saga[x->index-1] += w_saga[j] * x->value;
          x++;
      }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }

  for(j=0; j<l; j++)
  {
      double exp_wTx = exp(xTw_saga[j]);
      double tau_tmp = 1 / (1 + exp_wTx);
      D[j] = C[GETI(j)] * exp_wTx * tau_tmp * tau_tmp;
  }
  info("Begin newton iteration\n");
  while (newton_iter < max_newton_iter) {
    leverage_value(prob, D, leverage);
    sample(leverage, l, index, max_saga_iter);
    // solve with SAGA
    for (int saga_iter = 0; saga_iter < max_saga_iter; ++saga_iter) {
      // randomly choose element \in (0,...,l-1)
      //int choose = rand() % l;
      int choose = index[saga_iter];
      // calculate difference of gradient for element chosen
      double xjTw_saga = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        xjTw_saga += x->value * w_saga[x->index - 1];
        x++;
      }
      double tmp = D[choose] * (xjTw_saga - xTw_saga[choose]);
      // update w_saga
      x = prob->x[choose]; // reset x
      double descent = 0.0;
      for (j = 0; j < w_size; ++j) {
        if (x->index - 1 == j) {
          // x[choose][j] non-zero
          descent = w_saga[j] - eta * (l * tmp * x->value + sum_grad[j]);
          sum_grad[j] += tmp * x->value;
          x++;
        } else {
          descent = w_saga[j] - eta * sum_grad[j];
        }

        // do proximal mapping
        if (descent > eta) {
          w_saga[j] = descent - eta;
        } else if (descent < -eta) {
          w_saga[j] = descent + eta;
        } else {
          w_saga[j] = 0.0;
        }
      }
      // update xTw_saga
      xTw_saga[choose] = xjTw_saga;
    }
    // clear xTw_saga
    for (j = 0; j < l; ++j) {
      xTw_saga[j] = 0;
    }
    // re-calculate xTw_saga
    for (j = 0; j < w_size; ++j) {
      //printf("%lf ", w_saga[j]);
      sum_grad[j] = 0;
      feature_node *x = prob_col->x[j];
      while (x->index != -1) {
        int ind = x->index - 1;
        xTw_saga[ind] += w_saga[j] * x->value;
        x++;
      }
    }
    //printf("\n");
    // re-calculate sum_grad
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
      } else {
        tmp *= (1.0 - 1.0 / (1.0 + exp(xTw_saga[j])));
      }
      while (x->index != -1) {
        sum_grad[x->index - 1] += x->value * tmp;
        x++;
      }
    }
    // update D
    for (j = 0; j < l; ++j) {
      double exp_xTw_saga = exp(xTw_saga[j]);
      D[j] = C[GETI(j)] * exp_xTw_saga / (1.0 + exp_xTw_saga) / (1.0 + exp_xTw_saga);
    }
    int nnz = 0;
    // function value and sparsity
    double value = 0;
    for (j = 0; j < w_size; ++j) {
      if (w_saga[j] != 0) {
        value += fabs(w_saga[j]);
        nnz++;
      }
    }
    for (j = 0; j < l; j++)
      if (y[j] == 1)
        value += C[GETI(j)] * log(1 + 1 / exp(xTw_saga[j]));
      else
        value += C[GETI(j)] * log(1 + exp(xTw_saga[j]));
    info("func=%f sparsity %d/%d\n", value, nnz, w_size);
    newton_iter++;
  }
  delete [] y;
  delete [] D;
  delete [] w_saga;
  delete [] xTw_saga;
  delete [] xTw;
  delete [] sum_grad;
  delete [] leverage;
}


static void solve_l1r_lr_proxsvrg_opt(const problem *prob_col,
                                  double *w,
                                  double eps,
                                  double Cp,
                                  double Cn,
                                  int max_newton_iter,
                                  const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s, newton_iter = 0, iter = 0;
  int max_iter = 10;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;
  double w_norm, w_norm_new;
  double QP_Gmax_old = INF;
  double Gmax_new, QP_Gnorm1_new;
  double delta, negsum_xTd, cond;

  schar *y = new schar[l];
  double *intercept = new double[l];
  double *slope = new double[l];
  // parameters for svrg
  double *grad_snapshot = new double[w_size];
  double *xTw = new double[l];
  int *delay = new int[w_size];
  double eta = 0.00001;
  int svrg_iter_max = 2;
  int svrg_inner_max = l * 1;
  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
    w[j] = 0;
    grad_snapshot[j] = 0;
    delay[j] = 0;
  }
  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw[j] = 0;
  }

  double time = 0.0;
  while (newton_iter < max_newton_iter) {
    double begin = clock();
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double sum = 0.0, exp_sum = 0.0;
      while (x->index != -1) {
        sum += x->value * w[x->index - 1];
        x++;
      }
      exp_sum = exp(sum);
      if (y[j] > 0) {
        intercept[j] = -Cp / (1.0 + exp_sum) - Cp * sum * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
        slope[j] = Cp * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
      } else {
        intercept[j] = Cn * exp_sum / (1.0 + exp_sum) - Cn * sum * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
        slope[j] = Cn * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
      }
    }
    // solve with SVRG
    for (int svrg_iter = 0; svrg_iter < svrg_iter_max; ++svrg_iter) {
      // calculate the full gradient
      memset(grad_snapshot, 0, sizeof(double) * w_size);
      for (int j = 0; j < l; ++j) {
        feature_node *x = prob->x[j];
        double sum = 0.0;
        while (x->index != -1) {
          sum += x->value * w[x->index - 1];
          x++;
        }
        xTw[j] = sum;
        x = prob->x[j];
        while (x->index != -1) {
          int idx = x->index - 1;
          grad_snapshot[idx] += (intercept[j] + slope[j] * sum) * x->value;
          x++;
        }
      }
      // SVRG inner loop
      for (int svrg_inner_iter = 0; svrg_inner_iter < svrg_inner_max; ++svrg_inner_iter) {
        // randomly choose one sample
        int choose = rand() % l;
        // calculate xTw_svrg_inner
        double sum = 0.0;
        feature_node *x = prob->x[choose];
        while (x->index != -1) {
          int ind = x->index - 1;
          w[ind] = recover(w[ind], delay[ind], svrg_inner_iter, eta, eta * grad_snapshot[ind]);
          sum += w[ind] * x->value;
          x++;
        }
        // update w_svrg_inner using new sample
        x = prob->x[choose];
        double tmp = l * slope[choose] * (sum - xTw[choose]);
        while (x->index != -1) {
          int idx = x->index - 1;
          delay[idx] = svrg_inner_iter + 1;
          double descent = w[idx] - eta * (grad_snapshot[idx] + x->value * tmp);
          if (descent > eta)
            w[idx] = descent - eta;
          else if (descent < -eta)
            w[idx] = descent + eta;
          else
            w[idx] = 0.0;
          x++;
        }
      }
      for (int j = 0; j < w_size; ++j) {
        w[j] = recover(w[j], delay[j], svrg_inner_max, eta, eta * grad_snapshot[j]);
        delay[j] = 0;
      }
    }
    newton_iter++;
    time += (clock() - begin) / CLOCKS_PER_SEC;
    // function value
    double value = 0;
    // copy w_svrg to w
    for (j = 0; j < w_size; ++j) {
      value += fabs(w[j]);
    }
    for (j = 0; j < l; j++)
      if (y[j] == 1)
        value += C[GETI(j)] * log(1 + 1 / exp(xTw[j]));
      else
        value += C[GETI(j)] * log(1 + exp(xTw[j]));
    info("%lf, %lf\n", value, time);
  }
}


static void solve_l1r_lr_proxsvrg(const problem *prob_col,
                                  double *w,
                                  double eps,
                                  double Cp,
                                  double Cn,
                                  int max_newton_iter,
                                  const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s, newton_iter = 0, iter = 0;
  int max_iter = 10;
  int max_num_linesearch = 20;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;
  double w_norm, w_norm_new;
  double QP_Gmax_old = INF;
  double Gmax_new, QP_Gnorm1_new;
  double delta, negsum_xTd, cond;

  schar *y = new schar[l];
  double *intercept = new double[l];
  double *slope = new double[l];
  // parameters for svrg
  double *grad_snapshot = new double[w_size];
  double *xTw = new double[l];
  double eta = 1.0e-4;
  int svrg_iter_max = 2;
  int svrg_inner_max = l * 0.1;
  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
    w[j] = 0;
    //w_svrg[j] = 0;
    //w_svrg_inner[j] = 0;
    grad_snapshot[j] = 0;
  }
  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw[j] = 0;
  }

  info("Begin newton iteration\n");
  double time = 0.0;
  while (newton_iter < max_newton_iter) {
    double begin = clock();
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double sum = 0.0, exp_sum = 0.0;
      while (x->index != -1) {
        sum += x->value * w[x->index - 1];
        x++;
      }
      exp_sum = exp(sum);
      if (y[j] > 0) {
        intercept[j] = -Cp / (1.0 + exp_sum) - Cp * sum * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
        slope[j] = Cp * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
      } else {
        intercept[j] = Cn * exp_sum / (1.0 + exp_sum) - Cn * sum * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
        slope[j] = Cn * exp_sum / (1.0 + exp_sum) / (1.0 + exp_sum);
      }
    }
    // solve with SVRG
    for (int svrg_iter = 0; svrg_iter < svrg_iter_max; ++svrg_iter) {
      // calculate the full gradient
      memset(grad_snapshot, 0, sizeof(double) * w_size);
      for (int j = 0; j < l; ++j) {
        feature_node *x = prob->x[j];
        double sum = 0.0;
        while (x->index != -1) {
          sum += x->value * w[x->index - 1];
          x++;
        }
        xTw[j] = sum;
        x = prob->x[j];
        while (x->index != -1) {
          int idx = x->index - 1;
          grad_snapshot[idx] += (intercept[j] + slope[j] * sum) * x->value;
          x++;
        }
      }
      // SVRG inner loop
      for (int svrg_inner_iter = 0; svrg_inner_iter < svrg_inner_max; ++svrg_inner_iter) {
        // randomly choose one sample
        int choose = rand() % l;
        // calculate xTw_svrg_inner
        double sum = 0.0;
        feature_node *x = prob->x[choose];
        while (x->index != -1) {
          int ind = x->index - 1;
          sum += w[ind] * x->value;
          x++;
        }
        // update w_svrg_inner using new sample
        x = prob->x[choose];
        for (j = 0; j < w_size; ++j) {
          double gradient = 0.0, descent = 0.0;
          if (j == x->index - 1) {
            gradient = grad_snapshot[j] + l * x->value * slope[choose] * (sum - xTw[choose]);
            x++;
          } else {
            gradient = grad_snapshot[j];
          }
          descent = w[j] - eta * gradient;
          // proximal update
          if (descent > eta)
            w[j] = descent - eta;
          else if (descent < -eta)
            w[j] = descent + eta;
          else
            w[j] = 0;
        }
      }
    }
    newton_iter++;
    time += (clock() - begin) / CLOCKS_PER_SEC;
    for (int j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double sum = 0.0;
      while (x->index != -1) {
        sum += x->value * w[x->index - 1];
        x++;
      }
      xTw[j] = sum;
    }
    // function value
    double value = 0;
    // copy w_svrg to w
    for (j = 0; j < w_size; ++j) {
      value += fabs(w[j]);
    }
    for (j = 0; j < l; j++)
      if (y[j] == 1)
        value += C[GETI(j)] * log(1 + 1 / exp(xTw[j]));
      else
        value += C[GETI(j)] * log(1 + exp(xTw[j]));
    info("%.15lf, %.5lf\n", value, time);
  }
}


#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_svrg_opt(const problem *prob_col,
                              double *w,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_svrg_iter,
                              const problem *prob) {
  int l = prob_col->l, w_size = prob_col->n;
  schar *y = new schar[l];
  int max_inner_iter = 2.5 * l;
  double eta = 1.0e-7;
  double *w_svrg = new double[w_size];
  int *delay = new int[w_size];
  double *grad_snapshot = new double[w_size];
  double *xTw_svrg = new double[l];
  double C[3] = {Cn, 0, Cp};

  memset(w_svrg, 0, sizeof(double)*w_size);
  memset(delay, 0, sizeof(int) * w_size);
  memset(xTw_svrg, 0, sizeof(double)*l);

  for (int i = 0; i < l; ++i) {
    y[i] = prob->y[i] > 0 ? 1 : -1;
  }
  double time = 0.0;
  // SVRG outer iteration
  for (int i = 0; i < max_svrg_iter; ++i) {
    double begin = clock();
    // calculate full gradient
    memset(grad_snapshot, 0, sizeof(double)*w_size);
    for (int j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp(xTw_svrg[j]));
      } else {
        tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_svrg[j]));
      }
      while (x->index != -1) {
        grad_snapshot[x->index - 1] += tmp * x->value;
        x++;
      }
    }
    // svrg inner iteration
    for (int inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
      int choose = rand() % l;
      // calculate sample gradient
      // re-calculate xTw_svrg_inner[choose]
      double xTw_choose = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        int idx = x->index - 1;
        w_svrg[idx] = recover(w_svrg[idx], delay[idx], inner_iter, eta, eta * grad_snapshot[idx]);
        xTw_choose += w_svrg[x->index - 1] * x->value;
        x++;
      }
      x = prob->x[choose]; // reset x
      double tmp = C[GETI(choose)] * (1.0 / (exp(xTw_svrg[choose]) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
      while (x->index != -1) {
        int idx = x->index - 1;
        delay[idx] = inner_iter + 1;
        double descent = w_svrg[idx] - eta * (grad_snapshot[idx] + l * x->value * tmp);
        if (descent > eta) {
          w_svrg[idx] = descent - eta;
        } else if (descent < -eta) {
          w_svrg[idx] = descent + eta;
        } else {
          w_svrg[idx] = 0.0;
        }
        x++;
      }
    }
    // recover w
    for (int j = 0; j < w_size; ++j) {
      w_svrg[j] = recover(w_svrg[j], delay[j], max_inner_iter, eta, eta * grad_snapshot[j]);
      delay[j] = 0;
    }
    //eta *= 0.9;
    // update xTw_svrg
    for (int j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      xTw_svrg[j] = 0;
      while (x->index != -1) {
        xTw_svrg[j] += w_svrg[x->index - 1] * x->value;
        x++;
      }
    }
    time += (clock() - begin) / CLOCKS_PER_SEC;
    // display function value
    double v = 0;
    int nnz = 0;
    for(int j=0; j < w_size; j++)
      if(w_svrg[j] != 0)
      {
        v += fabs(w_svrg[j]);
        nnz++;
      }
    for(int j=0; j < l; j++)
      if(y[j] == 1)
        v += C[GETI(j)]*log(1+1/exp(xTw_svrg[j]));
      else
        v += C[GETI(j)]*log(1+exp(xTw_svrg[j]));
    printf("%lf, %lf\n", v, time, (double)nnz/w_size);
  }
  memcpy(w, w_svrg, sizeof(double) * w_size);
  delete [] w_svrg;
  delete [] grad_snapshot;
  delete [] xTw_svrg;
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_svrg(const problem *prob_col,
                              double *w,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_svrg_iter,
                              const problem *prob) {
  int l = prob_col->l, w_size = prob_col->n;
  schar *y = new schar[l];
  int max_inner_iter = 0.01 * l;
  double eta = 1.0e-7;
  double *w_svrg_old = new double[w_size];
  double *w_svrg = new double[w_size];
  double *w_svrg_inner = new double[w_size];
  double *grad_snapshot = new double[w_size];
  double *xTw_svrg = new double[l];
  double C[3] = {Cn, 0, Cp};

  memset(w_svrg_old, 0, sizeof(double)*w_size);
  memset(w_svrg, 0, sizeof(double)*w_size);
  memset(w_svrg_inner, 0, sizeof(double)*w_size);
  memset(grad_snapshot, 0, sizeof(double)*w_size);
  memset(xTw_svrg, 0, sizeof(double)*l);

  for (int i = 0; i < l; ++i) {
    y[i] = prob->y[i] > 0 ? 1 : -1;
  }
  double time = 0.0;
  // SVRG outer iteration
  for (int i = 0; i < max_svrg_iter; ++i) {
    double begin = clock();
    // calculate full gradient
    memset(grad_snapshot, 0, sizeof(double) * w_size);
    for (int j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp(xTw_svrg[j]));
      } else {
        tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_svrg[j]));
      }
      while (x->index != -1) {
        grad_snapshot[x->index - 1] += tmp * x->value;
        x++;
      }
    }
    // svrg inner iteration
    for (int inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
      int choose = rand() % l;
      // calculate sample gradient
      // re-calculate xTw_svrg_inner[choose]
      double xTw_choose = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        xTw_choose += w_svrg_inner[x->index - 1] * x->value;
        x++;
      }
      x = prob->x[choose]; // reset x
      // update w_svrg_inner
      for (int j = 0; j < w_size; ++j) {
        double descent = 0.0;
        if (x->index - 1 == j) {
          double tmp = C[GETI(choose)] * x->value * (1.0 / (exp(xTw_svrg[choose]) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
          descent = w_svrg_inner[j] - eta * (l * tmp + grad_snapshot[i]);
          x++;
        } else {
          descent = w_svrg_inner[j] - eta * grad_snapshot[j];
        }
        // proximal mapping
        if (descent > eta) {
          w_svrg_inner[j] = descent - eta;
        } else if (descent < -eta) {
          w_svrg_inner[j] = descent + eta;
        } else {
          w_svrg_inner[j] = 0;
        }
      }
    }
    double end_inner = clock();

    
    // FISTA basic
    //for (int j = 0; j < w_size; ++j) {
      //w_svrg_inner[j] = (2 * i - 1) / (i + 1) * w_svrg_inner[j] - (i - 2) / (i + 1) * w_svrg[j];
    //}
    // copy w_svrg_inner to w_svrg
    memcpy(w_svrg, w_svrg_inner, sizeof(double) * w_size);

    // update xTw_svrg
    for (int j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      xTw_svrg[j] = 0;
      while (x->index != -1) {
        xTw_svrg[j] += w_svrg[x->index - 1] * x->value;
        x++;
      }
    }
    time += (clock() - begin) / CLOCKS_PER_SEC;
    // display function value
    if (i % 2 == 0) {
    double v = 0;
    int nnz = 0;
    for(int j=0; j < w_size; j++)
      if(w_svrg[j] != 0)
      {
        v += fabs(w_svrg[j]);
        nnz++;
      }
    for(int j=0; j < l; j++)
      if(y[j] == 1)
        v += C[GETI(j)]*log(1+1/exp(xTw_svrg[j]));
      else
        v += C[GETI(j)]*log(1+exp(xTw_svrg[j]));
    printf("%lf, %lf\n", v, time);
    }
  }
  delete [] w_svrg;
  delete [] w_svrg_inner;
  delete [] grad_snapshot;
  delete [] xTw_svrg;
}

static void permute(int *index, int len) {
  for (int i = 0; i < len; ++i) {
    int choose = rand() % len;
    // swap i with choose
    int tmp = index[i];
    index[i] = index[choose];
    index[choose] = tmp;
  }
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_saga_opt(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int data_access,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;

  int max_saga_iter = data_access * l;
  int count_cycle = 1.5 * l;
  
  double nu = 1e-12;
  double inner_eps = 1;

  schar *y = new schar[l];
  double *w_tmp = new double[w_size];
  double *xTw_saga = new double[l];
  double *xTw_saga_ = new double[l];
  int *delay = new int[w_size];
  int *index = new int[l];
  double *sum_grad = new double[w_size];
  double eta = 1e-7;

  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
    delay[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_saga[j] = 0;
    index[j] = j;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_saga[x->index-1] += w_saga[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  double time = 0.0, begin;
  for (int saga_iter = 0; saga_iter < max_saga_iter; ++saga_iter) {
    begin = clock();
    if (saga_iter % l == 0) permute(index, l);
    // choose sample
    //int choose = rand() % l;
    int choose = index[saga_iter % l];
    // calculate gradient
    double xTw_choose = 0.0;
    feature_node *x = prob->x[choose];
    while (x->index != -1) {
      int idx = x->index - 1;
      w_saga[idx] = recover(w_saga[idx], delay[idx], saga_iter, eta, eta * sum_grad[idx]);
      xTw_choose += x->value * w_saga[idx];
      x++;
    }
    // update sum_grad and xTw_saga
    x = prob->x[choose];
    double tmp = C[GETI(choose)] * (1.0 / (exp(xTw_saga[choose]) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
    while (x->index != -1) {
      int idx = x->index - 1;
      delay[idx] = saga_iter + 1;
      double descent = w_saga[idx] - eta * (sum_grad[idx] + l * x->value * tmp);
      sum_grad[idx] += tmp * x->value;
      if (descent > eta) {
        w_saga[idx] = descent - eta;
      } else if (descent < -eta) {
        w_saga[idx] = descent + eta;
      } else {
        w_saga[idx] = 0.0;
      }
      x++;
    }
    xTw_saga[choose] = xTw_choose;
    time += (clock() - begin) / CLOCKS_PER_SEC;
    if ((saga_iter + 1) % count_cycle == 0) {
      // recover w_tmp
      for (j = 0; j < w_size; ++j) {
        w_tmp[j] = recover(w_saga[j], delay[j], saga_iter, eta, eta * sum_grad[j]);
      }
      // function value and sparsity
      for (int i = 0; i < l; ++i) {
        xTw_saga_[i] = 0.0;
      }
      for(int i=0; i < w_size; i++)
      {
        feature_node *x = prob_col->x[i];
        while(x->index != -1)
        {
          xTw_saga_[x->index-1] += w_tmp[i] * x->value;
          x++;
        }
      }
      double value = 0;
      for (int i = 0; i < w_size; ++i) {
        if (w_tmp[i] != 0) {
          value += fabs(w_tmp[i]);
        }
      }
      for (int i = 0; i < l; i++)
        if (y[i] == 1)
          value += C[GETI(i)] * log(1 + 1 / exp(xTw_saga_[i]));
        else
          value += C[GETI(i)] * log(1 + exp(xTw_saga_[i]));
      info("%lf, %lf\n", value, time);
    }
  }
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_saga(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;

  max_saga_iter = 20;// * l;
  int timing_period = 0.10*l;

  double nu = 1e-12;
  double inner_eps = 1;

  schar *y = new schar[l];

  double *xTw_saga = new double[l];
  double *xTw_saga_ = new double[l];
  double *sum_grad = new double[w_size];
  double *full_grad = new double[w_size];
  double *est_grad = new double[w_size];
  double eta = 1.0e-3;

  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
    full_grad[j] = 0;
    est_grad[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_saga[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_saga[x->index-1] += w_saga[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (j == 3) printf("tmp %lf xTw_saga %lf\n", tmp, xTw_saga[j]);
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    if (j == 3) printf("tmp %lf xTw_saga %lf\n", tmp, xTw_saga[j]);
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  printf("%lf %lf\n", sum_grad[0], xTw_saga[0]);
  double time = 0.0, begin;
  for (int saga_iter = 0; saga_iter < max_saga_iter; ++saga_iter) {
    printf("%lf\n", w_saga[0]);
    begin = clock();
    // choose sample
    int choose = rand() % l;
    // calculate gradient
    double xTw_choose = 0.0;
    feature_node *x = prob->x[choose];
    while (x->index != -1) {
      xTw_choose += x->value * w_saga[x->index - 1];
      x++;
    }
    // update sum_grad and xTw_saga
    x = prob->x[choose];
    for (int i = 0; i < w_size; ++i) {
      if (i <= 2) printf("%lf\n", sum_grad[i]);
      double descent = 0.0;
      if (x->index - 1 == i) {
        double tmp = C[GETI(choose)] * x->value * (1.0 / (exp(xTw_saga[choose]) + 1.0)
                                                   - 1.0 / (exp(xTw_choose) + 1.0));
        descent = w_saga[i] - eta * (sum_grad[i] + l * tmp);
        est_grad[i] = sum_grad[i] + l * tmp; 
        sum_grad[i] += tmp;
        x++;
      } else {
        descent = w_saga[i] - eta * sum_grad[i];
        est_grad[i] = sum_grad[i];
      }
      // update w_saga
      if (descent > eta) {
        w_saga[i] = descent - eta;
      } else if (descent < -eta) {
        w_saga[i] = descent + eta;
      } else {
        w_saga[i] = 0;
      }
    }
    xTw_saga[choose] = xTw_choose;
    time += (clock() - begin) / CLOCKS_PER_SEC;
    if ((1 + saga_iter) % timing_period == 0) {
    // function value
    // function value and sparsity
    for (int i = 0; i < l; ++i) {
      xTw_saga_[i] = 0.0;
    }
    for(int i=0; i < w_size; i++)
    {
      feature_node *x = prob_col->x[i];
      while(x->index != -1)
      {
        xTw_saga_[x->index-1] += w_saga[i] * x->value;
        x++;
      }
    }
    double value = 0;
    for (int i = 0; i < w_size; ++i) {
      if (w_saga[i] != 0) {
        value += fabs(w_saga[i]);
      }
    }
    for (int i = 0; i < l; i++)
      if (y[i] == 1)
        value += C[GETI(i)] * log(1 + 1 / exp(xTw_saga_[i]));
      else
        value += C[GETI(i)] * log(1 + exp(xTw_saga_[i]));
    info("%lf, %lf\n", value, time);
    }
  }
}


#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_asyncsagapp2(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int max_saga_inner = 0.1 * l;
  int max_threads = 8;
  double lambda;
  double eta = 0.0000003; // realsim
  omp_set_num_threads(max_threads);

  std::atomic_flag flag = ATOMIC_FLAG_INIT;
  schar *y = new schar[l];
  for (int j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
  }

  //struct atomic_flag flag;
  //atomic_flag_clear(&flag);

  double *w_global = new double[w_size];
  double *XTw_global = new double[l];
  double *sumGrad_global = new double[w_size];

  memset(w_global, 0, sizeof(double) * w_size);
  memset(XTw_global, 0, sizeof(double) * l);
  memset(sumGrad_global, 0, sizeof(double) * w_size);
  
  double C[3] = {1, 0, 1};
  
  // thread private data
  int work = l;
  int more = 0;
  double **w_private = new double*[max_threads];
  double **XTw = new double*[max_threads];
  double **sumGrad = new double*[max_threads];
  int stage = 0; 
  for (int i = 0; i < max_threads; ++i) {
    w_private[i] = new double[w_size];
    memset(w_private[i], 0, sizeof(double) * w_size);
      XTw[i] = new double[l];
      memset(XTw[i], 0, sizeof(double) * l);
    sumGrad[i] = new double[w_size];
    memset(sumGrad[i], 0, sizeof(double) * w_size);
  }
  #pragma omp parallel
  {
    int max_outer = max_saga_iter;
    int tid = omp_get_thread_num();
    double *w_inner = w_private[tid];
    double *XTw_inner = XTw[tid];
    double *sumGrad_inner = sumGrad[tid];
    int stage_private = 0;
    struct drand48_data buffer;
    srand48_r(tid, &buffer);
    for (int i = 0; i < l; ++i) {
      feature_node *xi = prob->x[i];
      double tmp = 0.0;
      tmp = C[GETI(i)];
      if (y[i] > 0) {
        tmp *= -0.5;
      } else {
        tmp *= 0.5;
      }
      while (xi->index != -1) {
        sumGrad_inner[xi->index - 1] += xi->value * tmp;
        xi++;
      }
    }
    
    while(max_outer-- > 0) {
      for (int epoch = 0; epoch < max_saga_inner; ++epoch) {
        // update independently
        long int choose = 0;
        lrand48_r(&buffer, &choose);
        choose = choose % l;
        // calculate x_choose*w
        double sum = 0.0;
        feature_node *x_choose = prob->x[choose];
        while (x_choose->index != -1) {
          sum += x_choose->value * w_inner[x_choose->index - 1];
          x_choose++;
        }
        // get gradient
        double tmp = C[GETI(choose)] * (1.0 / (1.0 + exp(XTw_inner[choose])) - 1.0 / (1.0 + exp(sum)));
        x_choose = prob->x[choose];
        for (int i = 0; i < w_size; ++i) {
          double descent = 0.0;
          if (x_choose->index - 1 == i) {
            descent = w_inner[i] - eta * (sumGrad_inner[i] + l * x_choose->value * tmp);
            sumGrad_inner[i] += tmp * x_choose->value;
            x_choose++;
          }
          else descent = w_inner[i] - eta * sumGrad_inner[i];
          if (descent > eta) { w_inner[i] = descent - eta; }
          else if (descent < -eta) { w_inner[i] = descent + eta; }
          else { w_inner[i] = 0.0; }
        }
        XTw_inner[choose] = sum;
      }

      stage_private++;
      if (stage_private > stage) lambda = 0.3;
      else if (stage_private < stage) lambda = 0.7;
      else lambda = 0.5;
      // update w_global
     
      for (int i = 0; i < w_size; ++i) {
        double tmp = 0.0;
        #pragma omp atomic read
        tmp = w_global[i];
        #pragma omp atomic write
        w_global[i] = w_inner[i]; 
        w_inner[i] = lambda * tmp + (1.0 - lambda) * w_inner[i];
      }
      
      for (int i = 0; i < l; ++i) {
        feature_node *xi = prob->x[i];
        double sum = 0.0;
        while (xi->index != -1) {
          sum += xi->value * w_inner[xi->index - 1];
          xi++;
        }
        XTw_inner[i] = sum;
      }
      memset(sumGrad_inner, 0, sizeof(double) * w_size);
      for (int i = 0; i < l; ++i) {
        feature_node *xi = prob->x[i];
        double tmp_p = 0.0, tmp_n = 0.0;
        tmp_p = -Cp / (1.0 + exp(XTw_inner[i]));
        tmp_n = Cn * (1.0 - 1.0 / (1.0 + exp(XTw_inner[i])));
        while (xi->index != -1) {
          if (y[i] > 0)
            sumGrad_inner[xi->index - 1] += tmp_p * xi->value;
          else
            sumGrad_inner[xi->index - 1] += tmp_n * xi->value;
          xi++;
        }
      }
    } // END while
    
  } // END parallel block
  memset(XTw_global, 0, sizeof(double) * l);
  for (int i = 0; i < l; ++i) {
    feature_node *xi = prob->x[i];
    double sum = 0.0;
    while (xi->index != -1) {
      sum += xi->value * w_global[xi->index - 1];
      xi++;
    }
    XTw_global[i] = sum;
  }
  double value = 0;
  for (int i = 0; i < w_size; ++i) {
    if (w_global[i] != 0) {
      value += fabs(w_global[i]);
    }
  }
  for (int i = 0; i < l; i++)
    if (y[i] == 1)
      value += C[GETI(i)] * log(1.0 + 1.0 / exp(XTw_global[i]));
    else
      value += C[GETI(i)] * log(1.0 + exp(XTw_global[i]));
  printf("Final: %lf\n", value);
}


#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_asyncsagapp(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;
  int max_saga_outer = max_saga_iter;
  int max_saga_inner = 2 * l;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];

  double *xTw_saga = new double[l];
  double *sum_grad = new double[w_size];
  int *delay = new int[w_size];
  //double eta = 1.0e-7; // covtype
  double eta = 0.00007; // realsim
  double C[3] = {1, 0, 1};
  int max_threads = 10;
  omp_set_num_threads(max_threads);
  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
    delay[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_saga[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_saga[x->index-1] += w_saga[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  struct drand48_data *buffer = new struct drand48_data[max_threads];
  memset(buffer, 0, sizeof(struct drand48_data) * max_threads);
  for (int i = 0; i < max_threads; ++i)
    srand48_r(i, &buffer[i]);
  double end_inner;
  int counter = 0;
  double total_time = 0.0;
  struct timespec begin, end;
  for (int saga_outer_iter = 0; saga_outer_iter < max_saga_outer; ++saga_outer_iter) {
  #pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    #pragma omp single
    {
      clock_gettime(CLOCK_MONOTONIC, &begin);
      counter = 0;
    }
    #pragma omp for
    for (int saga_inner_iter = 0; saga_inner_iter < max_saga_inner; ++saga_inner_iter) { 
      int nThread = omp_get_num_threads();
      // choose sample
      long choose = 0;
      lrand48_r(buffer + tid, &choose);
      choose = choose % l;
      // calculate gradient
      double xTw_choose = 0.0;
      const feature_node *x = prob->x[choose];
      while (x->index != -1) {
        int idx = x->index - 1;
        double w_saga_tmp, delay_tmp, counter_tmp;
        #pragma omp atomic read
        delay_tmp = delay[idx];
        #pragma omp atomic read
        w_saga_tmp = w_saga[idx];
        #pragma omp atomic read
        counter_tmp = counter;
        w_saga_tmp = recover(w_saga_tmp, delay_tmp, counter_tmp, eta, eta * sum_grad[idx]);
        #pragma omp atomic write
        w_saga[idx] = w_saga_tmp;
        #pragma omp atomic write
        delay[idx] = counter_tmp;
        xTw_choose += x->value * w_saga_tmp;
        x++;
      }
      // update sum_grad and xTw_saga
      x = prob->x[choose];
      double xtw_old = 0.0;
      #pragma omp atomic read
      xtw_old = xTw_saga[choose];
      double tmp = C[GETI(choose)] * (1.0 / (exp(xtw_old) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
      while (x->index != -1) {
        int idx = x->index - 1, counter_tmp;
        double descent = 0.0, w_saga_tmp, sum_grad_tmp;
        #pragma omp atomic read
        counter_tmp = counter;
        #pragma omp atomic read
        w_saga_tmp = w_saga[idx];
        #pragma omp atomic read
        sum_grad_tmp = sum_grad[idx];
        descent = w_saga_tmp - eta * (sum_grad_tmp + l * x->value * tmp);
        sum_grad_tmp += tmp * x->value;
        if (descent > eta) 
          w_saga_tmp = descent - eta;
        else if (descent < -eta) 
          w_saga_tmp = descent + eta; 
        else
          w_saga_tmp = 0.0;
        #pragma omp atomic write
        sum_grad[idx] = sum_grad_tmp;
        #pragma omp atomic write
        w_saga[idx] = w_saga_tmp;
        #pragma omp atomic write
        delay[idx] = counter_tmp + 1;
        x++;
      }
      #pragma omp atmic write
      xTw_saga[choose] = xTw_choose;
      #pragma omp atomic update
      counter++;
    }
    #pragma omp for
    for (j = 0; j < w_size; ++j) {
      w_saga[j] = recover(w_saga[j], delay[j], max_saga_inner, eta, eta * sum_grad[j]);
      delay[j] = 0;
    }
    #pragma omp single 
    {
      // recalculate
      memset(xTw_saga, 0, sizeof(double) * l);
      memset(sum_grad, 0, sizeof(double) * w_size);
    }
    #pragma omp for
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      while (x->index != -1) {
        xTw_saga[j] += w_saga[x->index - 1] * x->value;
        x++;
      }
    }
    #pragma omp for
    for (j = 0; j < w_size; ++j) {
      feature_node *x = prob_col->x[j];
      double tmp = 0.0;
      while (x->index != -1) {
        if (y[x->index - 1] > 0) {
          tmp = -Cp / (1.0 + exp(xTw_saga[x->index - 1])) * x->value; 
        }
        else {
          tmp = Cn * (1.0 - 1.0 / (1.0 + exp(xTw_saga[x->index - 1]))) * x->value;
        }
        sum_grad[j] += tmp;
        x++;
      }
    }
    #pragma omp single
    {
      clock_gettime(CLOCK_MONOTONIC, &end);
      total_time += end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1.0e9;
      double value = 0;
      for (int i = 0; i < w_size; ++i) {
        if (w_saga[i] != 0) {
          value += fabs(w_saga[i]);
        }
      }
      for (int i = 0; i < l; i++)
        if (y[i] == 1)
          value += C[GETI(i)] * log(1 + 1 / exp(xTw_saga[i]));
        else
          value += C[GETI(i)] * log(1 + exp(xTw_saga[i]));
      info("%lf, %lf\n", value, total_time);
   }
   }
 }
}


#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_sgd(const problem *prob_col,
                             double *w_sgd,
                             double eps,
                             double Cp,
                             double Cn,
                             int max_sgd_epoch,
                             const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  schar *y = new schar[l];
  double *xTw = new double[l];
  int *delay = new int[w_size];
  double eta = 1.0e-3;
  double C[3] = {1, 0, 1};
  int iter_per_epoch = 0.5 * l;
  int max_sgd_iter = iter_per_epoch * max_sgd_epoch;
  
  for (int j = 0; j < w_size; ++j) {
    w_sgd[j] = 0;
    delay[j] = 0;
  }

  for (int j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
  }
  double time = 0.0;
  for (int iter = 0; iter < max_sgd_iter; ++iter) {
    double begin = clock();
    int choose = rand() % l;
    // calculate gradient
    double xTw_choose = 0.0;
    feature_node *x = prob->x[choose];
    while (x->index != -1) {
      int idx = x->index - 1;
      w_sgd[idx] = recover(w_sgd[idx], delay[idx], iter, eta, 0.0);
      xTw_choose += x->value * w_sgd[idx];
      x++;
    }
    x = prob->x[choose];
    double coef = 0.0;
    if (y[choose] > 0) {
      coef = -Cp / (1.0 + exp(xTw_choose));
    }
    else {
      coef = Cn * (1.0 - 1.0 / (1.0 + exp(xTw_choose)));
    }
    while (x->index != -1) {
      int idx = x->index - 1;
      delay[idx] = iter + 1;
      double descent = w_sgd[idx] - eta * coef * x->value * l;
      if (descent > eta)
        w_sgd[idx] = descent - eta;
      else if (descent < -eta) 
        w_sgd[idx] = descent + eta;
      else
        w_sgd[idx] = 0.0;
      x++;
    }
    /*
    for (int j = 0; j < w_size; ++j) {
      double descent = 0.0;
      if (x->index - 1 == j) {
        descent = w_sgd[j] - eta * coef * x->value * l;
        x++;
      }
      else {
        descent = w_sgd[j];
      }
      if (descent > eta)
        w_sgd[j] = descent - eta;
      else if (descent < -eta)
        w_sgd[j] = descent + eta;
      else
        w_sgd[j] = 0.0;
    }
    */
    if ((iter+1) % iter_per_epoch == 0) {
      // recover w
      for (int j = 0; j < w_size; ++j) {
        w_sgd[j] = recover(w_sgd[j], delay[j], iter+1, eta, 0.0);
        delay[j] = iter+1;
      }
      eta *= 0.9;
    }
    time += (clock() - begin) / CLOCKS_PER_SEC;
    if ((iter+1) % iter_per_epoch == 0) {
      // calculate xTw  
      for (int j = 0; j < l; ++j) {
        feature_node *xj = prob->x[j];
        double sum = 0.0;
        while (xj->index != -1) {
          int idx = xj->index - 1;
          sum += xj->value * w_sgd[idx];
          xj++;
        }
        xTw[j] = sum;
      }
      double value = 0;
      for (int i = 0; i < w_size; ++i) {
        if (w_sgd[i] != 0) {
          value += fabs(w_sgd[i]);
        }
      }
      for (int i = 0; i < l; i++)
        if (y[i] == 1)
          value += C[GETI(i)] * log(1 + 1 / exp(xTw[i]));
        else
          value += C[GETI(i)] * log(1 + exp(xTw[i]));
      info("%lf, %lf\n", value, time);
    }
  }
}

static void shuffle(int *vec, int len, int n) {
  for (int i = 0; i < n; ++i) {
    int choose = rand() % len;
    // swap i with choose
    int tmp = vec[choose];
    vec[choose] = vec[i];
    vec[i] = tmp;
  }
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_sagapp_opt(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;
  int max_saga_outer = max_saga_iter;
  int max_saga_inner = 2.5 * l;
  schar *y = new schar[l];
  int *index = new int[l];
  double *exp_xTw_saga = new double[l];
  double *sum_grad = new double[w_size];
  int *delay = new int[w_size];
  double time = 0.0;
  //double eta = 1.0e-6; // mnist
  double eta = 1.0e-7; // realsim
  //double eta = 3.0e-7;
  double C[3] = {Cn, 0, Cp};
  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
    delay[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    exp_xTw_saga[j] = 1.0;
    index[j] = j;
  }
  for (int saga_outer_iter = 0; saga_outer_iter < max_saga_outer; ++saga_outer_iter) {
  double begin = clock();
    memset(sum_grad, 0, sizeof(double) * w_size);
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp_xTw_saga[j]);
      } else {
        tmp *= 1.0 - 1.0 / (1.0 + exp_xTw_saga[j]);
      }
      while (x->index != -1) {
        sum_grad[x->index - 1] += x->value * tmp;
        x++;
      }
    }
    //shuffle(index, l, max_saga_inner);
    for (int saga_inner_iter = 0; saga_inner_iter < max_saga_inner; ++saga_inner_iter) { 
      // choose sample
      int choose = rand() % l;
      //int choose = index[saga_inner_iter];
      // calculate gradient
      double exp_xTw_choose = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        int idx = x->index - 1;
        w_saga[idx] = recover(w_saga[idx], delay[idx], saga_inner_iter, eta, eta * sum_grad[idx]);
        exp_xTw_choose += x->value * w_saga[idx];
        x++;
      }
      exp_xTw_choose = exp(exp_xTw_choose);
      // update sum_grad and xTw_saga
      x = prob->x[choose];
      double tmp = C[GETI(choose)] * (1.0 / (exp_xTw_saga[choose] + 1.0) - 1.0 / (exp_xTw_choose + 1.0));
      while (x->index != -1) {
        int idx = x->index - 1;
        delay[idx] = saga_inner_iter + 1;
        double descent = w_saga[idx] - eta * (sum_grad[idx] + l * x->value * tmp);
        sum_grad[idx] += tmp * x->value;
        if (descent > eta) {
          w_saga[idx] = descent - eta;
        } else if (descent < -eta) {
          w_saga[idx] = descent + eta;
        } else {
          w_saga[idx] = 0.0;
        }
        x++;
      }
      exp_xTw_saga[choose] = exp_xTw_choose;
    }
    for (j = 0; j < w_size; ++j) {
      w_saga[j] = recover(w_saga[j], delay[j], max_saga_inner, eta, eta * sum_grad[j]);
      delay[j] = 0;
    }
    // recalculate
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double sum = 0.0;
      while (x->index != -1) {
        sum += w_saga[x->index - 1] * x->value;
        x++;
      }
      exp_xTw_saga[j] = exp(sum);
    }
    time += (clock() - begin) / CLOCKS_PER_SEC;
        double value = 0;
        for (int i = 0; i < w_size; ++i) {
          if (w_saga[i] != 0) {
            value += fabs(w_saga[i]);
          }
        }
        for (int i = 0; i < l; i++)
          if (y[i] == 1)
            value += C[GETI(i)] * log(1 + 1 / exp_xTw_saga[i]);
          else
            value += C[GETI(i)] * log(1 + exp_xTw_saga[i]);
        info("%lf, %lf\n", value, time);
  
}
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_sagapp_batch(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;
  int max_saga_outer = max_saga_iter;
  int batch = 32;
  int max_saga_inner = 2 * l / batch;
  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];

  double *xTw_saga = new double[l];
  double *xTw_saga_ = new double[l];
  double *sum_grad = new double[w_size];
  int *delay = new int[w_size];
  //double eta = 1.0e-6; // mnist
  double eta = 0.0004; // realsim
  //double eta = 3.0e-7;
  double C[3] = {1, 0, 1};
  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
    delay[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_saga[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_saga[x->index-1] += w_saga[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  double time = 0.0;
  for (int saga_outer_iter = 0; saga_outer_iter < max_saga_outer; ++saga_outer_iter) {
    double begin = clock();
    for (int saga_inner_iter = 0; saga_inner_iter < max_saga_inner; ++saga_inner_iter) { 
      // choose sample
      int choose = rand() % (l - batch + 1);
      for (int k = choose; k < choose + batch; ++k) {
      // calculate gradient
      double xTw_choose = 0.0;
      feature_node *x = prob->x[k];
      while (x->index != -1) {
        int idx = x->index - 1;
        w_saga[idx] = recover(w_saga[idx], delay[idx], saga_inner_iter, eta, eta * sum_grad[idx]);
        xTw_choose += x->value * w_saga[idx];
        x++;
      }
      // update sum_grad and xTw_saga
      x = prob->x[k];
      double tmp = C[GETI(k)] * (1.0 / (exp(xTw_saga[k]) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
      while (x->index != -1) {
        int idx = x->index - 1;
        delay[idx] = saga_inner_iter + 1;
        double descent = w_saga[idx] - eta * (sum_grad[idx] + l * x->value * tmp);
        sum_grad[idx] += tmp * x->value;
        if (descent > eta) {
          w_saga[idx] = descent - eta;
        } else if (descent < -eta) {
          w_saga[idx] = descent + eta;
        } else {
          w_saga[idx] = 0.0;
        }
        x++;
      }
      xTw_saga[k] = xTw_choose;
      }
    }
    for (j = 0; j < w_size; ++j) {
      w_saga[j] = recover(w_saga[j], delay[j], max_saga_inner, eta, eta * sum_grad[j]);
      delay[j] = 0;
    }
    double end_inner = clock();
    // recalculate
    memset(xTw_saga, 0, sizeof(double) * l);
    memset(sum_grad, 0, sizeof(double) * w_size);
    for(j=0; j<w_size; j++)
    {
      feature_node *x = prob_col->x[j];
      while(x->index != -1)
      {
        xTw_saga[x->index-1] += w_saga[j] * x->value;
        x++;
      }
    }
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
      } else {
        tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
      }
      while (x->index != -1) {
        sum_grad[x->index - 1] += x->value * tmp;
        x++;
      }
    }
    time += (clock() - begin) / CLOCKS_PER_SEC; 
    double value = 0;
    for (int i = 0; i < w_size; ++i) {
      if (w_saga[i] != 0) {
        value += fabs(w_saga[i]);
      }
    }
    for (int i = 0; i < l; i++)
      if (y[i] == 1)
        value += C[GETI(i)] * log(1 + 1 / exp(xTw_saga[i]));
      else
        value += C[GETI(i)] * log(1 + exp(xTw_saga[i]));
    info("%lf, %lf\n", value, time);
  }
}

#undef GETI
#define GETI(i) (y[i]+1)
static void solve_l1r_lr_sagapp(const problem *prob_col,
                              double *w_saga,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_saga_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;
  int max_saga_outer = max_saga_iter;
  int max_saga_inner = 0.1 * l;
  long counter = 0;
  long max_count = 6;
  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];

  double *xTw_saga = new double[l];
  double *sum_grad = new double[w_size];
  double eta = 1.0e-4; // realsim
  double C[3] = {Cn, 0, Cp};

  for (j = 0; j < w_size; ++j) {
    w_saga[j] = 0;
    sum_grad[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_saga[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_saga[x->index-1] += w_saga[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
    }
    //if (j == 3) printf("tmp %lf xTw_saga %lf\n", tmp, xTw_saga[j]);
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  //printf("%lf, %lf\n", sum_grad[0], xTw_saga[0]);
  double time = 0.0;
  for (int saga_outer_iter = 0; saga_outer_iter < max_saga_outer; ++saga_outer_iter) {
    double begin = clock();
    for (int saga_inner_iter = 0; saga_inner_iter < max_saga_inner; ++saga_inner_iter) { 
      //printf("%lf\n", w_saga[0]);
      // choose sample
      int choose = rand() % l;
      // calculate gradient
      double xTw_choose = 0.0;
      feature_node *x = prob->x[choose];
      while (x->index != -1) {
        int idx = x->index - 1;
        xTw_choose += x->value * w_saga[idx];
        x++;
      }
      // update sum_grad and xTw_saga
      x = prob->x[choose];
      double tmp = C[GETI(choose)] * (1.0 / (exp(xTw_saga[choose]) + 1.0) - 1.0 / (exp(xTw_choose) + 1.0));
      
      for (int i = 0; i < w_size; ++i) {
        //if (i <= 2) printf("%lf\n", sum_grad[i]);
        double descent = 0.0;
        if (x->index - 1 == i) {
          descent = w_saga[i] - eta * (sum_grad[i] + l * x->value * tmp);
          sum_grad[i] += tmp * x->value;
          x++;
        }
        else {
          descent = w_saga[i] - eta * sum_grad[i];
        }
        // update w_saga
        if (descent > eta) {
          w_saga[i] = descent - eta;
        } else if (descent < -eta) {
          w_saga[i] = descent + eta;
        } else {
          w_saga[i] = 0;
        }
      }
      xTw_saga[choose] = xTw_choose;
    }
    // recalculate
    memset(xTw_saga, 0, sizeof(double) * l);
    memset(sum_grad, 0, sizeof(double) * w_size);
    for(j=0; j<w_size; j++)
    {
      feature_node *x = prob_col->x[j];
      while(x->index != -1)
      {
        xTw_saga[x->index-1] += w_saga[j] * x->value;
        x++;
      }
    }
    for (j = 0; j < l; ++j) {
      feature_node *x = prob->x[j];
      double tmp = 0.0;
      tmp = C[GETI(j)];
      if (y[j] > 0) {
        tmp *= -1.0 / (1.0 + exp(xTw_saga[j]));
      } else {
        tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_saga[j]));
      }
      while (x->index != -1) {
        sum_grad[x->index - 1] += x->value * tmp;
        x++;
      }
    }
    time += (clock() - begin) / CLOCKS_PER_SEC; 
    double value = 0;
    for (int i = 0; i < w_size; ++i) {
      if (w_saga[i] != 0) {
        value += fabs(w_saga[i]);
      }
    }
    for (int i = 0; i < l; i++)
      if (y[i] == 1)
        value += C[GETI(i)] * log(1 + 1 / exp(xTw_saga[i]));
      else
        value += C[GETI(i)] * log(1 + exp(xTw_saga[i]));
    if (counter % max_count == 0)
    info("%lf, %lf\n", value, time);
    counter++;
  }
}


static void solve_l1r_lr_sag(const problem *prob_col,
                              double *w_sag,
                              double eps,
                              double Cp,
                              double Cn,
                              int max_sag_iter,
                              const problem *prob) {
  int l = prob_col->l;
  int w_size = prob_col->n;
  int j, s;

  max_sag_iter = 20 * l;

  double nu = 1e-12;
  double inner_eps = 1;
  double sigma = 0.01;

  schar *y = new schar[l];

  double *xTw_sag = new double[l];
  double *xTw_sag_ = new double[l];
  double *sum_grad = new double[w_size];
  schar *sub_grad_w = new schar[l * w_size];
  double eta = 0.000001;

  double C[3] = {Cn, 0, Cp};
  for (j = 0; j < l * w_size; ++j) {
    sub_grad_w[j] = 0;
  }
  for (j = 0; j < w_size; ++j) {
    w_sag[j] = 0;
    sum_grad[j] = 0;
  }

  for (j = 0; j < l; ++j) {
    if (prob_col->y[j] > 0)
      y[j] = 1;
    else
      y[j] = -1;
    xTw_sag[j] = 0;
  }
  for(j=0; j<w_size; j++)
  {
    feature_node *x = prob_col->x[j];
    while(x->index != -1)
    {
      xTw_sag[x->index-1] += w_sag[j] * x->value;
      x++;
    }
  }
  for (j = 0; j < l; ++j) {
    feature_node *x = prob->x[j];
    double tmp = 0.0;
    tmp = C[GETI(j)];
    if (y[j] > 0) {
      tmp *= -1.0 / (1.0 + exp(xTw_sag[j]));
    } else {
      tmp *= 1.0 - 1.0 / (1.0 + exp(xTw_sag[j]));
    }
    while (x->index != -1) {
      sum_grad[x->index - 1] += x->value * tmp;
      x++;
    }
  }
  double time = 0.0, begin;
  int timing_period = 10000;
  for (int sag_iter = 0; sag_iter < max_sag_iter; ++sag_iter) {
    begin = clock();
    // choose sample
    int choose = rand() % l;
    // calculate gradient
    double xTw_choose = 0.0;
    feature_node *x = prob->x[choose];
    while (x->index != -1) {
      xTw_choose += x->value * w_sag[x->index - 1];
      x++;
    }
    // update sum_grad and xTw_sag
    x = prob->x[choose];
    for (int i = 0; i < w_size; ++i) {
      double descent = 0.0;
      double sub_grad = 0.0;
      if (w_sag[i] > 0) sub_grad = 1.0;
      else if (w_sag[i] < 0) sub_grad = -1.0;
      if (x->index - 1 == i) {
        double tmp = C[GETI(choose)] * x->value * (1.0 / (exp(xTw_sag[choose]) + 1.0)
                                                   - 1.0 / (exp(xTw_choose) + 1.0)) + (sub_grad - sub_grad_w[w_size*choose + i]);
        descent = w_sag[i] - eta * (sum_grad[i] + tmp);
        sum_grad[i] += tmp;
        x++;
      } else {
        descent = w_sag[i] - eta * (sum_grad[i] + sub_grad - sub_grad_w[w_size*choose + i]);
        sum_grad[i] += sub_grad - sub_grad_w[w_size*choose + i];
      }
      // update w_sag
      w_sag[i] = descent;
      sub_grad_w[w_size*choose + i] = sub_grad;
    }
    xTw_sag[choose] = xTw_choose;
    time += (clock() - begin) / CLOCKS_PER_SEC;
    if (sag_iter % timing_period == 0) {
      // function value
      // function value and sparsity
      for (int i = 0; i < l; ++i) {
        xTw_sag_[i] = 0.0;
      }
      for(int i=0; i < w_size; i++)
      {
        feature_node *x = prob_col->x[i];
        while(x->index != -1)
        {
          xTw_sag_[x->index-1] += w_sag[i] * x->value;
          x++;
        }
      }
      double value = 0;
      for (int i = 0; i < w_size; ++i) {
        if (w_sag[i] != 0) {
          value += fabs(w_sag[i]);
        }
      }
      for (int i = 0; i < l; i++)
        if (y[i] == 1)
          value += C[GETI(i)] * log(1 + 1 / exp(xTw_sag_[i]));
        else
          value += C[GETI(i)] * log(1 + exp(xTw_sag_[i]));
      info("%lf, %lf\n", value, time);
    }
  }
}


// transpose matrix X from row format to column format and SUBSAMPLE
static void transpose_subsample(const problem *prob, feature_node **x_space_ret, problem *prob_col, int *subsamples)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
		if ( subsamples[i]==1 )
		{
			feature_node *x = prob->x[i];
			while(x->index != -1)
			{
				nnz++;
				col_ptr[x->index]++;
				x++;
			}
		}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
		if (subsamples[i] == 1)
		{
			feature_node *x = prob->x[i];
			while(x->index != -1)
			{
				int ind = x->index-1;
				x_space[col_ptr[ind]].index = i+1; // starts from 1
				x_space[col_ptr[ind]].value = x->value;
				col_ptr[ind]++;
				x++;
			}
		}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}



// Proximal Newton method with sampled Hessian for  
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)
// Modified by Cho (removed the active subset selection trick)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr_sampled(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn, int max_newton_iter, const problem *prob)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
//	int max_newton_iter = 100;
	int max_iter = 10;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];

	int *samples = new int[l];
	int nn = 5; // sample_rate = 1/5
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
        double newton_beg = clock();
        Gmax_new = 0;
		Gnorm1_new = 0;

		// Compute gradient 
		for(j=0; j<w_size; j++)
		{
			Grad[j] = 0; // gradient

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		// Subsamples
		int l_sub=0;
		for ( int ii=0 ; ii<l ; ii++ )
		{
			if (rand()%nn == 0) {
				l_sub++;
				samples[ii] = 1;
			} else {
				samples[ii] = 0;
			}
		}
		// Rescale hessian
		for ( int ii=0 ; ii<l ; ii++ )
			D[ii]*=nn;

		// Create subsampled X
		problem prob_sub;
		feature_node *x_space = NULL;
		transpose_subsample(prob, &x_space, &prob_sub, samples);

		// Compute Diagonal of Hessian
		for(j=0; j<w_size; j++)
		{
			Hdiag[j] = nu; // diagonal of Hessian

			double tmp = 0;
			x = prob_sub.x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				x++;
			}
		}

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = w_size;

		// Need to maintain Xd during the inner coordinate updates
		for(int i=0; i<l; i++)
			xTd[i] = 0;

        double inner_beg = clock();
		// optimize QP over wpd (w+d)
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			// Random permutation for each coordinate descent epoch
			for(j=0; j<w_size; j++)
			{
				int i = j+rand()%(w_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<w_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_sub.x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_sub.x[j];
				sparse_operator::axpy(z, x, xTd); // update xTd: z*x+xTd, x is sparse
			}

			iter++;

			// if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			//	break;

			QP_Gmax_old = QP_Gmax_new;
		}
        double search_beg = clock();
		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		// Recompute xTd for whole training set
		for(int i=0; i<l; i++)
			xTd[i] = 0;
		for ( int j=0 ; j<w_size ; j++ )
		{
			double dnow = wpd[j]-w[j];
			x = prob_col->x[j];
			sparse_operator::axpy(dnow, x, xTd);
		}

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

        double newton_end = clock();
		// calculate objective value

		double v = 0;
		int nnz = 0;
		for(j=0; j<w_size; j++)
			if(w[j] != 0)
			{
				v += fabs(w[j]);
				nnz++;
			}
		for(j=0; j<l; j++)
			if(y[j] == 1)
				v += C[GETI(j)]*log(1+1/exp_wTx[j]);
			else
				v += C[GETI(j)]*log(1+exp_wTx[j]);

		info("Iter %3d #CD cycles %d obj %lf sparsity %d/%d iter %lf init %lf inner %lf search %lf\n",
             newton_iter, iter, v, nnz, w_size,
             (newton_end-newton_beg)/CLOCKS_PER_SEC,
             (inner_beg-newton_beg)/CLOCKS_PER_SEC,
             (search_beg-inner_beg)/CLOCKS_PER_SEC,
             (newton_end-search_beg)/CLOCKS_PER_SEC);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
	delete [] samples;
}


// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;

			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		case PROX_NEWTON_L1LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr_new(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter);
			//solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn); 
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case PROX_NEWTON_SUB_CD:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr_sampled(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
	        case PROX_NEWTON_SAGA:
	        {
	          problem prob_col;
	          feature_node *x_space = NULL;
	          transpose(prob, &x_space ,&prob_col);
	          solve_l1r_lr_proxsaga(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
	          delete [] prob_col.y;
	          delete [] prob_col.x;
	          delete [] x_space;
	          break;
	        }
	        case PROX_NEWTON_SAGAPP:
	        {
	           problem prob_col;
	           feature_node *x_space = NULL;
	           transpose(prob, &x_space ,&prob_col);
	           solve_l1r_lr_proxsagapp(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
	           delete [] prob_col.y;
	           delete [] prob_col.x;
	           delete [] x_space;
	           break;
	        }
		case PROX_NEWTON_SVRG:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr_proxsvrg(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case PROX_NEWTON_SVRG_OPT:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr_proxsvrg_opt(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
    case SVRG:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_svrg(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SVRG_OPT:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_svrg_opt(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SAGA:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_saga(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SAGA_OPT:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_saga_opt(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SAGAPP:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_sagapp(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SAGAPP_OPT:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_sagapp_opt(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case ASAGAPP:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_asyncsagapp(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    case SGD:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_sgd(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
    
    
    case SAG:
    {
      problem prob_col;
      feature_node *x_space = NULL;
      exit(0);
      transpose(prob, &x_space ,&prob_col);
      solve_l1r_lr_sag(&prob_col, w, primal_solver_tol, Cp, Cn, param->max_iter, prob);
      delete [] prob_col.y;
      delete [] prob_col.x;
      delete [] x_space;
      break;
    }
		default:
			fprintf(stderr, "ERROR: unknown solver_type %d\n", param->solver_type);
			break;
	}
}

// Calculate the initial C for parameter selection
static double calc_start_C(const problem *prob, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	max_xTx = 0;
	for(i=0; i<prob->l; i++)
	{
		xTx = 0;
		feature_node *xi=prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC)
		min_C = 1.0 / (2 * prob->l * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
}


//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, model_->w, 0, 0);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
		}
		else
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;
				
				if(param->init_sol != NULL)
					for(i=0;i<w_size;i++)
						model_->w[i] = param->init_sol[i];
				else
					for(i=0;i<w_size;i++)
						model_->w[i] = 0;

				train_one(&sub_prob, param, model_->w, weighted_C[0], weighted_C[1]);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					if(param->init_sol != NULL)
						for(j=0;j<w_size;j++)
							w[j] = param->init_sol[j*nr_class+i];
					else
						for(j=0;j<w_size;j++)
							w[j] = 0;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

void find_parameter_C(const problem *prob, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate)
{
	// variables for CV
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int, l);
	double *target = Malloc(double, prob->l);
	struct problem *subprob = Malloc(problem,nr_fold);

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;

		subprob[i].bias = prob->bias;
		subprob[i].n = prob->n;
		subprob[i].l = l-(end-begin);
		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
		subprob[i].y = Malloc(double,subprob[i].l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}

	}

	*best_rate = 0;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C = start_C;

	while(param1.C <= max_C)
	{
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		for(i=0; i<nr_fold; i++)
		{
			int j;
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			struct model *submodel = train(&subprob[i],&param1);

			int total_w_size;
			if(submodel->nr_class == 2)
				total_w_size = subprob[i].n;
			else
				total_w_size = subprob[i].n * submodel->nr_class;

			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(j=0; j<total_w_size; j++)
				{
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
					prev_w[i][j] = submodel->w[j];
				}
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}
			else
			{
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}

			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		for(i=0; i<prob->l; i++)
			if(target[i] == prob->y[i])
				++total_correct;
		double current_rate = (double)total_correct/prob->l;
		if(current_rate > *best_rate)
		{
			*best_C = param1.C;
			*best_rate = current_rate;
		}

		info("log2c=%7.2f\trate=%g\n",log(param1.C)/log(2.0),100.0*current_rate);
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;
		param1.C = param1.C*ratio;
	}

	if(param1.C > max_C && max_C > start_C) 
		info("warning: maximum C reached.\n");
	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
		free(prev_w[i]);
	}
	free(prev_w);
	free(subprob);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
                        continue;
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
	 	if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
		if (fscanf(fp, "\n") !=0)
		{
			fprintf(stderr, "ERROR: fscanf failed to read the model\n");
			EXIT_LOAD_MODEL()
		}
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx) 
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else 
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if(param->init_sol != NULL)
		free(param->init_sol);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

/*	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL
		&& param->solver_type != L2R_L2LOSS_SVR
		&& param->solver_type != L2R_L2LOSS_SVR_DUAL
		&& param->solver_type != L2R_L1LOSS_SVR_DUAL)
		return "unknown solver type";
*/
	if(param->init_sol != NULL 
		&& param->solver_type != L2R_LR && param->solver_type != L2R_L2LOSS_SVC)
		return "Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}



