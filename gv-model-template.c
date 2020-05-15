#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<stdbool.h>
#include "mod3ops.h"
#include "link-table.h"

#define NUM_RUNS {num_runs}

{update_functions}

unsigned __int128 compute_int_rep({typed_param_list})
{{
  unsigned __int128 accumulator = 0;
{accumulate_hash_one}
  return accumulator;
}}

int main(int argc, char** argv)
{{
  /* initialize random number generator */
  srandom(time(0));

  init_table();
  
  for(int runs = 0; runs < NUM_RUNS; runs++)
  {{

    /* declare and initialize variables */
{declare_variables}
{initialize_variables}

    /* initialize hashes */
    unsigned __int128 current_int_rep, next_int_rep;
    current_int_rep = compute_int_rep({param_list});

    /* computational loop */
    int step = 0;
    do
      {{

        /* compute next iteration, storing in temp variables */
{compute_next8}

        /* copy temp variables back */
{copy8}

        /* compute the rep */
        next_int_rep = compute_int_rep({param_list});

	/* insert into table */
	record_link(current_int_rep, next_int_rep, step);

	/* advance representative to next state */
	current_int_rep = next_int_rep;
	step++;
	
      }}
    while (!is_present(current_int_rep));
  }}
  
  print_table_summary();
  
  return 0;
  
}}
