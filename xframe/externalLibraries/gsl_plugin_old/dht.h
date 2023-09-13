/* dht/gsl_dht.h
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* Author:  G. Jungman
 */
#include <stdlib.h>
#include <complex.h>
#include <gsl/gsl_dht.h>
#include <gsl/gsl_complex.h>

#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif

__BEGIN_DECLS

typedef double *restrict gsl_restrict_complex_packed_array;
typedef const double *restrict gsl_const_restrict_complex_packed_array;

int dht_2D(const size_t n_orders,const gsl_dht *restrict*restrict t, const double (*f_in)[t[0]->size][2], const double (*f_out)[t[0]->size][2],gsl_const_restrict_complex_packed_array prefactor);

int dht_bare(const gsl_dht *restrict*restrict t, gsl_const_restrict_complex_packed_array f_in, gsl_restrict_complex_packed_array f_out, gsl_complex prefactor);

int test(size_t size_x, size_t size_y,const double (*restrict data)[size_y][2]);

__END_DECLS


