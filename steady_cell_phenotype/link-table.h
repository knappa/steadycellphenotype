#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

typedef struct {
  unsigned __int128 source;
  unsigned __int128 target;
  unsigned int step;
  bool occupied;
} link;

#define INIT_TABLE_SIZE 65536
size_t table_size = INIT_TABLE_SIZE;
link* links;
size_t num_recorded = 0;

void init_table()
{
  links = (link*) calloc(INIT_TABLE_SIZE, sizeof(link));
}


void record_link(unsigned __int128 source, unsigned __int128 target, unsigned int step)
{

  /* if we will exceed 2/3 of table, rebalance */
  if( (1+num_recorded)*3 > table_size*2 )
    {
      /* stash old size and double new size */
      size_t old_size = table_size;
      table_size = 2 * old_size;

      /* stash old table and allocate new one */
      link* old_links = links;
      links = (link*) calloc(table_size, sizeof(link));

      num_recorded = 0;
      
      /* copy old table contents */
      for(size_t index = 0; index < old_size; index++)
	{
	  if( old_links[index].occupied )
	    {
	      record_link(old_links[index].source, old_links[index].target, old_links[index].step);
	    }
	}

      free(old_links);
    }
  
  size_t index = (size_t) ( source % table_size );
  
  /* forward chained */
  int forward_chaining_length = 0;

  do {
  
    if( !links[index].occupied )
      {
	/* new entry */
	links[index].source = source;
	links[index].target = target;
	links[index].occupied = true;
	links[index].step = step;
	num_recorded++;
	return;
      }
    else if( links[index].source == source )
      {
	printf("rewriting a link?\n");
	if( links[index].target != target )
	  { printf("with a different destination?!?\n"); }
	return;
      }

    forward_chaining_length++;
    index++;
    if(index >= table_size) index = 0; /* wrap around */

  } while (forward_chaining_length < table_size);

  printf("table full?! should not happen\n");
}

bool is_present(unsigned __int128 source)
{
  size_t index = (size_t) ( source % table_size );
  
  /* forward chained */
  int forward_chaining_length = 0;

  do {
  
    if( !links[index].occupied )
      { return false; }
    else if( links[index].source == source )
      {	return true; }

    forward_chaining_length++;
    index++;
    if(index >= table_size) index = 0; /* wrap around */

  } while (forward_chaining_length < table_size);

  return false;
}

void print128(unsigned __int128 value)
{
  printf("0x%"PRIX64, (uint64_t)(value>>64));
  printf("%016"PRIX64, (uint64_t)value);
}

void print_table_summary(void)
{
  printf("{\"edges\":[");
  bool first_edge = true;
  for(int index=0; index < table_size; index++)
    {
      if( links[index].occupied )
	{
	  if( !first_edge )
	    { printf(" , "); }
	  else
	    { first_edge=false; }
	  
	  printf("{\"source\": ");
	  printf("\""); print128(links[index].source); printf("\"");

	  printf(" , \"target\": ");
	  printf("\""); print128(links[index].target); printf("\"");

      printf(" , \"step\": %u", links[index].step);

	  printf(" }");
	}
    }
  printf("]}\n");
}

