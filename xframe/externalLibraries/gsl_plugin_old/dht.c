#include "dht.h"
#include <stdlib.h>
#include <complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_dht.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

int
dht_loop(const gsl_dht *restrict t, const gsl_complex *restrict f_in, gsl_complex *restrict f_out, const gsl_complex prefactor)
{
  size_t m;
  size_t i;
  double Y;      
  for(m=0; m<t->size; m++)
    {
      gsl_complex sum = gsl_complex_rect(0.0,0.0);
      
      for(i=0; i<m; i++)
	{
	  /*	  printf("real in= %f \n",real_f_in[i]);
		  printf("imag in= %f \n",imag_f_in[i]);*/
	  Y = t->Jjj[m*(m+1)/2 + i] / t->J2[i+1];
	  sum=gsl_complex_add(sum,gsl_complex_mul_real(f_in[i],Y));
	}
      for(i=m;i<t->size; i++)
	{
	  Y = t->Jjj[i*(i+1)/2 + m] / t->J2[i+1];
	  sum=gsl_complex_add(sum,gsl_complex_mul_real(f_in[i],Y));
	}
      f_out[m]=gsl_complex_mul(prefactor,sum);
    }
  return GSL_SUCCESS;
}



int
dht_2D(size_t n_orders,const gsl_dht *restrict*restrict t, const double (*f_in)[t[0]->size][2], const double (*f_out)[t[0]->size][2], gsl_const_restrict_complex_packed_array prefactor)
{
  size_t order;
  const gsl_complex *restrict local_prefactor=(const gsl_complex *restrict)prefactor;
  for(order=0;order<n_orders; order++)
    {
      dht_loop(t[order],(gsl_complex*)f_in[order],(gsl_complex *)f_out[order],local_prefactor[order]);
    }
  return GSL_SUCCESS;
}



int
dht_bare(const gsl_dht *restrict*restrict t, gsl_const_restrict_complex_packed_array f_in, gsl_restrict_complex_packed_array f_out, gsl_complex prefactor)
{
  printf("alive");
  dht_loop(t[0],(gsl_complex*)f_in,(gsl_complex*)f_out, prefactor);
  return GSL_SUCCESS;
}



int test2(size_t size_y,const gsl_complex *restrict data)
{
  size_t y;
  for(y=0;y<size_y;y++)
    {
      //	  printf("real in %f \n",data[x*size_y*2+y*2]);
      //	  printf("imag in %f \n",data[x*size_y*2+y*2+1]);
      printf("real in %f \n",GSL_REAL(data[y]));
      //	  printf("imag in %f \n",data[x][2*y+1]); 
    }
  return 0;
}

int test(size_t size_x,size_t size_y,const double (*restrict data)[size_y][2])
{
  size_t x;
  for(x=0;x<size_x;x++)
    {
      test2(size_y,(gsl_complex *)data[x]);
    }
  return GSL_SUCCESS;
}


