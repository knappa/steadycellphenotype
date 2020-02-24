#define INITIAL_SIZE 256

typedef struct {
  u_int32_t *array;
  size_t size;
  size_t max_length;
} path_length_count;

void init_counts(path_length_count *plc)
{
  plc->array = (u_int32_t *) malloc(INITIAL_SIZE * sizeof(u_int32_t));
  plc->size = INITIAL_SIZE;
  plc->max_length = 0;
  /* TODO: check, does malloc zero memory? If so, this is unneeded. */
  for(int idx = 0; idx < INITIAL_SIZE; idx ++)
    { (plc->array)[idx] = 0; }
}

void update_counts(path_length_count *plc, int new_length)
{
  /* If the array is too small, make it bigger by repeated doubling */
  int orig_size = plc->size;
  while (plc->size <= new_length)
    {
      plc->size *= 2;
      plc->array = (u_int32_t *) realloc(plc->array, plc->size * sizeof(u_int32_t) );
    }
  /* init to zero any new regions */
  for(int idx = orig_size; idx < plc->size; idx++)
    { (plc->array)[idx] = 0; }

  /* record the length */
  (plc->array)[new_length]++;

  /* update max length */
  plc->max_length = (plc->max_length > new_length) ? plc->max_length : new_length;

}

void print_length_distribution(path_length_count *plc)
{
  printf(" \"length-dist\" : [");
  if( plc->max_length >= 0 )
    {
      printf("%u", (plc->array)[0]);
      for(int idx = 1; idx <= plc->max_length; idx++)
	{ printf(", %u", (plc->array)[idx]); }
    }
  printf("]");
}

void free_length_count(path_length_count *plc)
{
  free(plc->array);
  plc->array = NULL;
  plc->size = 0;
  plc->max_length = 0;
}
