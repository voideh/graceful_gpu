#include<stdio.h>
#include <sys/time.h>
struct timeval start, end;
void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(int NUMNODES) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%d Nodes: %f ms\n", NUMNODES,elapsed); 
}

void init(int NUMNODES) {
  printf("***************** %d **********************\n", NUMNODES);
  starttime();
}

void finish(int NUMNODES) {
  endtime(NUMNODES);
  printf("***************************************************\n");
}

