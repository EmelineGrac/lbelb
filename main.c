#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "pixel_operations.h"
#include <err.h>

void wait_for_keypressed(void) {
  SDL_Event             event;
  // Infinite loop, waiting for event
  for (;;) {
    // Take an event
    SDL_PollEvent( &event );
    // Switch on event type
    switch (event.type) {
    // Someone pressed a key -> leave the function
    case SDL_KEYDOWN: return;
    default: break;
    }
  // Loop until we got the expected event
  }
}

void init_sdl(void) {
  // Init only the video part
  if( SDL_Init(SDL_INIT_VIDEO)==-1 ) {
    // If it fails, die with an error message
    errx(1,"Could not initialize SDL: %s.\n", SDL_GetError());
  }
  // We don't really need a function for that ...
}

SDL_Surface* load_image(char *path) {
  SDL_Surface          *img;
  // Load an image using SDL_image with format detection
  img = IMG_Load(path);
  if (!img)
    // If it fails, die with an error message
    errx(3, "can't load %s: %s", path, IMG_GetError());
  return img;
}

SDL_Surface* display_image(SDL_Surface *img) {
  SDL_Surface          *screen;
  // Set the window to the same size as the image
  screen = SDL_SetVideoMode(img->w, img->h, 0, SDL_SWSURFACE|SDL_ANYFORMAT);
  if ( screen == NULL ) {
    // error management
    errx(1, "Couldn't set %dx%d video mode: %s\n",
         img->w, img->h, SDL_GetError());
  }
 
  /* Blit onto the screen surface */
  if(SDL_BlitSurface(img, NULL, screen, NULL) < 0)
    warnx("BlitSurface error: %s\n", SDL_GetError());
 
  // Update the screen
  SDL_UpdateRect(screen, 0, 0, img->w, img->h);
 
  // wait for a key
  wait_for_keypressed();
 
  // return the screen for further uses
  return screen;
}

int main(int argc, char *argv [])
{
	if(argc < 2)
		errx(1, "pas d'image");
	float sum;
	init_sdl();
	SDL_Surface* img = load_image(argv[1]);
	display_image(img);
	load_image(argv[1]);
	for(int y = 0; y < img->h; y++)
	{
		for(int x = 0; x < img->w; x++)
		{
			Uint32 p = getpixel(img, x, y);
			Uint8 r, g, b;
			SDL_GetRGB(p, img->format, &r, &g, &b);
			sum = 0.3*(float)r + 0.59*(float)g + 0.11*(float)b;
			// binarize the pictures
			if(sum >= 128)
				sum = 255;
			else
				sum = 0;
			p = SDL_MapRGB(img->format, sum, sum, sum);
			putpixel(img, x, y, p);
		}
	}
	
	//MAKE THE ARRAY WITH 0 AND 1 ( WHITE PIXEL = 0 AND BLACK PIXEL = 1)
	int * array = NULL;
	array = malloc(sizeof(int*) * (img->h) * (img->w));
	int *arrayX = array;
	 for(int y = 0; y < img->h; ++y)                                      
   	 {                                                        
		 for(int x = 0; x < img->w; ++x)                                 
		 {                                    
			   Uint32 p = getpixel(img, x, y);                         
	                   Uint8 r, g, b;                                         
	                   SDL_GetRGB(p, img->format, &r, &g, &b);
	                   if(r == 255)
	        	            *arrayX = 0;
	                   else
	                            *arrayX = 1;
	                         ++arrayX;
	                 }
	   } 
	// HERE INITIALIS OF THE MEMORY	
	int **listLigne = NULL;
	listLigne = malloc(sizeof(int*) * (img->h));
    	int **debutListLigne=listLigne; 
	int cpt = 1, n = 1, b = 0, h = 0;
	if ( listLigne == NULL)
        {
        printf(" Out of memory!\n");
        exit(1);
        }
	int *tabListX = NULL;
	tabListX = malloc(sizeof(int) * ((img->h) * (img->w)));
	arrayX = array;
	for(int y = 0; y < img->h; ++y)
	{
		if(cpt){
		  	b = 0;

			//CHECK IF WE HAVE A LINE OF DARK PIXEL (DARK PIXEL = 1)
                	for(int x = 0 ; x < img->w; ++x)
                	{
				if(*arrayX != 0)
					++b;
				++arrayX;
                	}
			if(b > 0)
			{
				cpt = 0;
			}
		}
		else
		{
		  	b = 0;
			if(n)
			{
				*listLigne = tabListX;
				n = 0;
			}
				//IF WE FOUND A LINE OF DARK PIXEL:
				//FOR EACH PIXEL PLACE IT IN A NEW TAB 
				for(int xTab = 0; xTab < img->w; xTab++)
                		{
				  	if(*arrayX == 0)
						++b;
					   if(b == (img->w) )
					   { 
			      		      n = 1;
		      			      cpt=1;
					      ++tabListX;
					      for(int u = 0; u < img->w; ++u){
						   *tabListX = 24;                
						    ++tabListX;                   
					      }
					      *tabListX = 42; 
					      ++tabListX;
					      ++listLigne;
					   }
					   else
					   {
					   	*tabListX = *arrayX;
					  	++tabListX;
					   	++arrayX;
					   }
                		}
				++h;	
        	}
	}
	++listLigne;
	*listLigne = NULL;
	//DISPLAY OF THE ARRAY OF THE LINE
	printf( "\n" );//Display
	int *debutdebut = *debutListLigne;
	int i = 0;
	while(*debutdebut != 24){
	        printf("%d", *debutdebut);//Display	
		if(i == (img->w-1)){
	 		 printf("\n");//Display
	  		i = 0;
		}
		else
			++i;
		++debutdebut;
		}
	printf("\n");//Display
	//Char extraction 
	int **listChar = NULL;
	listChar = malloc(sizeof(int*) * (img->h) * (img->w));
    	while (*debutListLigne != NULL){ //All line
        	int j=0, Bool=1, deb = 0;
        	while (j < img->w){  //One line
            		int i = j;
            		if(Bool){//Search for the beginning of the char
                   		 while(*((*debutListLigne)+i) == 0){ 
                    			i+=img->w;                
                    		}
                    		if(*((*debutListLigne)+i) == 24){ 
                        		j++;                   
                    		}	
                    		else{            
                    			deb=j; //Find it
                    			Bool=0;
                    			j++;
                    		}        
            		}
            		else{  //Search for the end of the char
                	  while(*((*debutListLigne)+i)==0){ 
                	          i+=img->w;  
                	  }		
                	  if(*((*debutListLigne)+i)!=24){
                    		  j++;
                	  }	
                	  else{
                	    int fin=j; //find it 
                	    //Full the array of char 
			    int *tabCharX = NULL;
			    tabCharX= malloc(sizeof(int)*((img->h) * (img->w)));
			    i=0;
                	    while(*((*debutListLigne)+i)!=24 ){
                    	         for(int k=deb;k<fin;k++){
                       		    *tabCharX = *((*debutListLigne + i+k));
				    printf("%d",*tabCharX);//Display
                 		    ++tabCharX;                       
                    		 }
		   		   printf("\n");//Display 
		   		   i+=img->w;
                	    }
			    //Put 42 at the end of the array 
                	    *tabCharX = 42;       
			    printf("\n");//Display
	       		    j=fin+1;
                	    Bool=1;
                	    ++listChar;
                	  }
           		}
       		} 
		++debutListLigne;
   	}
    	++listChar;
  	*listChar = NULL;
	printf("\n");//Display
	printf("End");
       	printf("\n");

        display_image(img);
        SDL_FreeSurface(img);
	
	return 0;
}
