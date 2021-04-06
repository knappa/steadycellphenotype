#include <stdbool.h>

#define FILTER_SIZE_EXPONENT 16
#define FILTER_SIZE 65536
#define LOG_2_OF_64 6

/* filter for finding repeat in sequences */
u_int64_t sequence_filter[FILTER_SIZE];
/* filter for states which have been visited */
u_int64_t state_filter[FILTER_SIZE];

void flush_filter(u_int64_t* filter)
{
  for(int i = 0; i < FILTER_SIZE; i++)
    { filter[i] = 0; }
}

bool is_marked(u_int64_t* filter, unsigned long int hash_one, unsigned long int hash_two, unsigned long int hash_three)
{
  bool poss_present = true;

  int index, bit;

  index = (hash_one >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_one & (( 1 << LOG_2_OF_64) - 1);
  poss_present = poss_present && (filter[index] & (1 << bit));

  index = (hash_two >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_two & (( 1 << LOG_2_OF_64) - 1);
  poss_present = poss_present && (filter[index] & (1 << bit));

  index = (hash_three >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_three & (( 1 << LOG_2_OF_64) - 1);
  poss_present = poss_present && (filter[index] & (1 << bit));

  return poss_present;
}

void mark(u_int64_t* filter, unsigned long int hash_one, unsigned long int hash_two, unsigned long int hash_three)
{
  int index, bit;
  
  index = (hash_one >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_one & (( 1 << LOG_2_OF_64) - 1);
  filter[index] = filter[index] | (1 << bit);

  index = (hash_two >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_two & (( 1 << LOG_2_OF_64) - 1);
  filter[index] = filter[index] | (1 << bit);

  index = (hash_three >> LOG_2_OF_64) % FILTER_SIZE;
  bit   = hash_three & (( 1 << LOG_2_OF_64) - 1);
  filter[index] = filter[index] | (1 << bit);
}
