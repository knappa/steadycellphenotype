#include<assert.h>

/*
 * basic mod 3 math routines
 */

int mod3reduce(int n)
{
  while(n < 0) { n += 3; }
  while(n > 3) { n -= 3; }
  return n;
}

int mod3pow(int base, int exponent) 
{
  base = mod3reduce(base);

  switch (base)
    {
    case 0:
      return 0; /* I decide that 0^0=0 */
    case 1:
      return 1;
    default: /* base == 2 */
      if(exponent == 0) return 1;
      else if(exponent == 1) return base;
      else
      {
        int half_power = mod3pow(base,exponent/2);
        if(exponent % 2 == 0) return mod3reduce(half_power*half_power);
        else return mod3reduce(base*half_power*half_power);
      }
    }
}

int mod3max(int a, int b)
{
  a = mod3reduce(a);
  b = mod3reduce(b);

  return a > b ? a : b;
}

int mod3min(int a, int b)
{
  a = mod3reduce(a);
  b = mod3reduce(b);

  return a > b ? b : a;
}

/* defined by 0 -> 2, 1 -> 1, 2 -> 0. */
int mod3not(int a)
{
  a = mod3reduce(a);

  return 2 - a;
}

/* helper function as in the PLoS article, doi:10.1371/journal.pcbi.1005352.t003 pg 16/24 */
int mod3continuity(int control, int evaluated)
{
  control = mod3reduce(control);
  evaluated = mod3reduce(evaluated);

  if( evaluated > control ) return control + 1;
  else if( evaluated < control ) return control - 1;
  else return control;
}
