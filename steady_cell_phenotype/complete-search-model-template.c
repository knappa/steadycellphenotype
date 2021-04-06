#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<stdbool.h>
#include "mod3ops.h"
#include "bloom-filter.h"
#include "cycle-table.h"

#define NUM_RUNS {num_runs}

{update_functions}

unsigned __int128 compute_int_rep({typed_param_list})
{{
  unsigned __int128 accumulator = 0;
{accumulate_hash_one}
  return accumulator;
}}

unsigned long int compute_hash_one({typed_param_list})
{{
  unsigned long int accumulator = 0;
{accumulate_hash_one}
  return accumulator;
}}

unsigned long int compute_hash_two({typed_param_list})
{{
  unsigned long int accumulator = 0;
{accumulate_hash_two}
  return accumulator;
}}

unsigned long int compute_hash_three({typed_param_list})
{{
  unsigned long int accumulator = 0;
{accumulate_hash_three}
  return accumulator;
}}


int main(int argc, char** argv)
{{

  bool first_cycle = true;
    
  printf("{{ \"complete_search\": true, \"cycles\": [");

  {state_for_loops_head}

  flush_filter(sequence_filter);

  /* declare variables */
  {declare_variables}
  /* declare hash variables */
  unsigned long int hash_one, hash_two, hash_three;

	/* initialize variables */
{fixed_variable_initialization}
 	/* initialize hashes */
        hash_one   = compute_hash_one  ({param_list});
	hash_two   = compute_hash_two  ({param_list});
	hash_three = compute_hash_three({param_list});

    int transition_count = 0;

    /* computational loop */
    do
      {{
        transition_count++;

        /* record state in bloom filters */
        mark(sequence_filter, hash_one, hash_two, hash_three);
	mark(state_filter, hash_one, hash_two, hash_three);

        /* compute next iteration, storing in temp variables */
{compute_next8}

        /* copy temp variables back */
{copy8}

        /* compute the hashes */
        hash_one   = compute_hash_one  ({param_list});
        hash_two   = compute_hash_two  ({param_list});
        hash_three = compute_hash_three({param_list});

      }}
    while (!is_marked(sequence_filter, hash_one, hash_two, hash_three));

    /* we have hit periodicity, re-run to find cycle length and hash value */
{variable_stash}

    int cycle_length = 0;
    unsigned __int128 cycle_hash = compute_int_rep({param_list});
    
    do
      {{
        cycle_length++;
	unsigned __int128 next_state_hash = compute_int_rep({param_list});
        cycle_hash = cycle_hash < next_state_hash ? cycle_hash : next_state_hash;

        /* compute next iteration, storing in temp variables */
{compute_next8}

        /* copy temp variables back */
{copy8}
      }}
    while( cycle_length < transition_count &&
           {neq_check} );

    /* determine if we really detected a cycle */
    if( cycle_length == transition_count &&
        {neq_check} )
      {{
	continue;
      }}

    /* record this cycle */
    int path_length_to_cycle = transition_count - cycle_length;
    bool new_cycle = record_cycle(cycle_hash, cycle_length, path_length_to_cycle);

    if(new_cycle)
      {{
	if(first_cycle) {{
	  first_cycle = false;
	}} else {{
	  printf(",");
	}}
	printf("{{ \"length\":%u, ", cycle_length);
	printf("\"id\":\"");
	print128(cycle_hash);
	printf("\", ");
	
        /* print the newly found cycle */
        printf("\"cycle\": [");
	bool first_state = true;
        for(int count = 0; count < cycle_length; count++ )
          {{
            /* compute next iteration, storing in temp variables */
{compute_next12}

            /* copy temp variables back */
{copy12}

            /* print the state */
            if(first_state) {{
	      first_state = false;
	    }} else {{
	      printf(",");
	    }}
{print_state}
          }}
        printf("]}}\n");
      }}
    
    {state_for_loops_tail}

  printf(" ],");
  print_cycle_counts(NUM_RUNS);
  
  return 0;
  
}}
