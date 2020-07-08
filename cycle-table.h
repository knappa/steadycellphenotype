#include <stdbool.h>
#include <stdint.h>
#include "length-count-array.h"
#include <inttypes.h>

#define TABLE_SIZE 65536

unsigned __int128 cycle_ids[TABLE_SIZE];
u_int64_t cycle_lengths[TABLE_SIZE];
u_int64_t cycle_counts[TABLE_SIZE];
path_length_count length_counts[TABLE_SIZE];

/* returns true if the cycle is new */
bool record_cycle(unsigned __int128 hash_code, int cycle_length, int path_length_to_cycle)
{
  
  int index = (int) ( hash_code % TABLE_SIZE );
  
  /* forward chained */
  int forward_chaining_length = 0;

  do {
  
    if( cycle_counts[index] == 0 )
      {
	/* new entry */
	cycle_ids[index] = hash_code;
	cycle_lengths[index] = cycle_length;
	cycle_counts[index] = 1;
	init_counts(&length_counts[index]);
	update_counts(&length_counts[index], path_length_to_cycle);
	return true;
      }
    else if( cycle_ids[index] == hash_code )
      {
	cycle_counts[index] += 1;
	update_counts(&length_counts[index], path_length_to_cycle);
	return false;
      }

    forward_chaining_length++;
    index++;
    if(index >= TABLE_SIZE) index = 0; /* wrap around */

  } while (forward_chaining_length < TABLE_SIZE);

  printf("Cycle table full!\n");
  return true; /* I guess? */
}

void print128(unsigned __int128 value)
{
  printf("0x%"PRIX64, (uint64_t)(value>>64));
  printf("%016"PRIX64, (uint64_t)value);
}

void print_cycle_counts(int runs)
{
  printf("\"counts\":[");
  bool first_cycle = true;
  for(int index=0; index < TABLE_SIZE; index++)
    {
      if( cycle_counts[index] > 0 )
      {
	    if(first_cycle) {
	        first_cycle = false;
	    } else {
	        printf(",");
	    }
	    printf("{ \"id\":\"");
	    print128(cycle_ids[index]);
	    printf("\", ");

	    printf("\"length\":%lu, \"count\":%lu, ",
		   cycle_lengths[index], cycle_counts[index]);
	    
	    float percent = (100.0*cycle_counts[index])/runs;
	    if(percent > 0.01) {
	      printf("\"percent\":%.2f, ", percent);
	    } else {
	      printf("\"percent\":%.2e, ", percent);
	    }
	    print_length_distribution(&length_counts[index]);
	    printf("}\n");
      }
    }
  printf("]}\n");
}

