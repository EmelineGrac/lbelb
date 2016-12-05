
 # include "create_array.h"

int* makeArray(/*char *argv [],*/ SDL_Surface *img){
         //init_sdl();
         //SDL_Surface* img = load_image(argv[2]);
         //load_image(argv[2]);
         //MAKE THE ARRAY WITH 0 AND 1 ( WHITE PIXEL = 0 AND BLACK PIXEL = 1)
         int *array = NULL;
         array = malloc(sizeof(int) * ((img->h) * (img->w)));
         int *arrayX = array;
          for(int y = 0; y < img->h; ++y)
          {
                  for(int x = 0; x < img->w; ++x)
                  {
                            Uint32 p = getpixel(img, x, y);
                            Uint8 r, g, b;
                            SDL_GetRGB(p, img->format, &r, &g, &b);
                            if(r >= 128)
                                     *arrayX = 0;
                            else
                                     *arrayX = 1;
                                  ++arrayX;
                  }
            }

   return array;
 }

int* makeArrayW1B0(/*char *argv [],*/ SDL_Surface *img){
         //MAKE THE ARRAY WITH 0 AND 1 ( WHITE PIXEL = 1 AND BLACK PIXEL = 0)
    float sum;
    int *array = NULL;
    array = malloc(sizeof(int) * ((img->h) * (img->w)));
    int *arrayX = array;
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
            p = getpixel(img, x, y);
            SDL_GetRGB(p, img->format, &r, &g, &b);
            if(r >= 128)
              *arrayX = 0;
            else
              *arrayX = 1;
            ++arrayX;
		}
	}
   return array;
 }

int** segmentation(int* array/*, char *argv []*/, SDL_Surface *img){
         //init_sdl();
         //SDL_Surface* img = load_image(argv[2]);
         //load_image(argv[2]);

         // HERE INITIALIS OF THE MEMORY
         int **listLigne = NULL;
         listLigne = malloc(sizeof(int*) * (img->h));
         int **debutListLigne=listLigne;
         int cpt = 1, n = 1, b = 0, h = 0;
         if ( listLigne == NULL)
         {
         printf(" Out of memory!\n");
         return NULL;
         }
         int *tabListX = NULL;
         tabListX = malloc(sizeof(int) * ((img->h) * (img->w)));
         int *arrayX = array;
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
	 int **debutlistChar = listChar;
	 int prem=2,temp=100,comp=0,lost=1;
         if (temp && comp)
	 {}
	 while (*debutListLigne != NULL){ //All line
                 int j=0, Bool=1, deb = 0,espace=0;

                 while (j < img->w){  //One char
                         int i = j;

			 if(Bool){//Search for the beginning of the char
                                  while(*((*debutListLigne)+i) == 0){
                                         i+=img->w;
                                 }
                                 if(*((*debutListLigne)+i) == 24){
                                         j++;
					 comp++;
				 }
                                 else{
                                         deb=j; //Find it
                                         Bool=0;
                                         j++;
					//printf("\ncomp = %d et temp=%d\n",comp,temp);
					 if(comp>temp)
					 {
					 espace=1;
					 }
					 if (prem == 0 && lost ==1)
			 		{
			   		temp = comp+1;
			   		lost=0;
			 		}
					comp=0;
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
			     if(espace)
			     {
			       printf("espace\n");
			       int *tabCharX = NULL;
			       tabCharX= malloc(sizeof(int)*((img->h) * (img->w)));
			       *tabCharX = 0;
			       ++tabCharX;
			       *tabCharX = 42;
			       ++tabCharX;
			       ++listChar;
			     	espace=0;
			     }
                             int fin=j; //find it
                             //Full the array of char
                             int *tabCharX = NULL;
                             tabCharX= malloc(sizeof(int)*((img->h) * (img->w)));
			     int *tabCharXdebut = tabCharX;
                             i=0;
			     prem--;
                             while(*((*debutListLigne)+i)!=24 ){
                                  for(int k=deb;k<fin;k++){
                                     *tabCharX = *((*debutListLigne + i+k));
                                     printf("%d",*tabCharX);//Display
                                     ++tabCharX;
                                  }
				  //put 20 at the end of a ligne
				    *tabCharX = 20;
				    ++tabCharX;
                                    printf("\n");//Display
                                    i+=img->w;
                             }
                             //Put 42 at the end of the array
                             *tabCharX = 42;
                             printf("\n");//Display
			     //printf("%d", *tabCharX);
                             //printf("\n");//Display
                             j=fin+1;
                             Bool=1;
			     *listChar = tabCharXdebut;
                             ++listChar;
                           }
                         }
                 }
                 ++debutListLigne;
         }
         ++listChar;
         *listChar = NULL;
	 return debutlistChar;
         //free(listChar);
         //free(listLigne);
         //free(tabListX);
 }

int* tabLetter(int* array){
  int w = 0;
  int h = 0;
  int b = 1;
  /*for(size_t y = 0; *array != 42; ++y){
    for(size_t x = 0; *array != 20; ++x){
      if(b)
	++w;
      ++array;
    }
    ++array;
    ++h;
    b = 0;
  }*/
  for(int i = 0; array[i] != 42; ++i){
    if(array[i] == 20){
      ++h;
      b = 0;
    }
    else{
      if(b)
      	++w;
    }
  }
  int* newArray = calloc(w * h, sizeof(int));
  int cpt = 0;
  for(int i = 0; array[i] != 42; ++i){
    if(array[i] != 20){
      newArray[cpt] = array[i];
      ++cpt;
    }
  }
  printf("value of h: %d \n", h);
  printf("value of w: %d \n", w);
  //struct tab;
  //int z = 0;
  int *tab = calloc(20 * 20, sizeof(int));
  //int divH = 0;
  //int divW = 0;
  double xRatio = w / (double)20;
  double yRatio = h / (double)20;
  double px, py;
  for(int i = 0; i < 20; ++i){
    for(int j = 0; j < 20; ++j){
      px = floor(j * xRatio);
      py = floor(i * yRatio);
      //if( array[(int)((py * w) + px)] != 20)
      tab[(i * 20) + j] = newArray[(int)((py * w) + px) ];
    }
  }
  return tab;
  /*if(h <= 20){
    divH = 20 - h;
    if(w <= 20){
      divW = 20 - w;
      for(int j = divH - divH/2; j < (int)(h - divH/2); ++j){
    	for(int i = divW - divW/2; i < (int)(w - divW/2); ++i){
      	  tab[i + j] = *array;
      	    ++array;
  	}
      }
    }
    else{
        divW = w / 20;
	if(w / 20 > (float)divW){
	  ++divW;
	}
	int cpt = 0;
	for(int j = divH - divH/2; j < (int)(h - divH/2); ++j){
    	  for(int i = 0; i < (int)w; ++i){
	    if(cpt == divW){
      	      tab[i + j] = *array;
	      cpt = 0;
	      }
	    else{
	      ++cpt;
	    }
      	    ++array;
  	  }
        }
    }
  }
  else{
    divH = h / 20;
    if(h / 20 > (float)divH)
      ++divH;
    if(w <= 20){
      divW = 20 - w;
    int cpth1 = 0;
    for(int j = 0; j < (int)h; ++j){
      if(cpth1 == divH){
      for(int i = divW - divW/2; i < (int)(w - divW/2); ++i){
      	tab[i + j] = *array;
      	++array;
  	}
      cpth1 = 0;
      //++array;
      }
      else{
	++cpth1;
      }
      //++array;
    }
    }
    else{
      divW = w / 20;
      if(w / 20 > (float)divW){
	++divW;
      }
      int cpth = 0;
      int cpt = 0;
	for(int j = 0; j < (int)h; ++j){
	  if(cpth == divH){
    	    for(int i = 0; i < (int)w; ++i){
	      if(cpt == divW){
      	        tab[i + j] = *array;
      	        ++array;
	        cpt = 0;
	      }
	      else{
	        ++cpt;
	      }
  	    }
	   cpth = 0;
	  }
	  else{
	    ++cpth;
	  }
	}
    }
  }*/
}
/*int* imgLetter(int* array){
  //Compute the dimension of the array
  size_t w = 0;
  size_t h = 0;
  int b = 1;
  for(size_t y = 0; *array != 42; ++y){
    for(size_t x = 0; *array != 20; ++x){
      if(b)
	++w;
      ++array;
    }
    ++h;
    b = 0;
  }
  //struct tab;
  int *tab = calloc(20 * 20, sizeof (int*));
  //Make a "zoom"
  float divW = 0;
  float divH = 0;
// TRUC QUI SERT A RIEN PEUT ETRE REMPLACER PAR - 20
  	divH = h/20;
	if(h <= 20)
		divH = 1 - divH;
	else
	  	divH -= 1;
	divH *= 20;

 	divW = w/20;
	if(w <= 20)
    		divW = 1 - divW;
	else
	  	divW -= 1;
   	divW *= 20;

  if(h <= 20){
     if(w <= 20){

       		for(int j = divH - divH/2; j < (h - divH/2); ++j){
    			for(int i = divW - divW/2; i < (w - divW/2); ++i){
      				tab[i + j] = *array;
      				++array;
  			}
  		}
  	}
	else{
	  //retrcicement en w
	}
  }
  else{
    //retrecicement en h

  }

}*/

/*typedef struct Image{
	int w,h;
	Pixel* dat;
}*/

/*SDL_Surface* NouvelleImage(int *array, SDL_Surface* img){
   SDL_Surface* newImg = img;
  size_t w = 0;
  size_t h = 0;
  int b = 1;
  for(size_t y = 0; *array != 42; ++y){
    for(size_t x = 0; *array != 20; ++x){
      if(b)
	++w;
      ++array;
    }
    ++h;
    b = 0;
  }
  size_t wRatio = w / img->w ;
  size_t hRatio = h / img->h;
  newImg = rotozoomSurfaceXY(newImg, 0, wRatio, hRatio, 0);
  for(int y = 0; y < hRatio; y++)
  {
    for(int x = 0; x < wRatio; x++)
      {
	 float sum = arrayp[x + y];
	 Uint32 p = SDL_MapRGB(newImg->format, sum, sum, sum);
         putpixel(newImg, x, y, p);
      }
   }
  wRatio = 20 / newImg->w;
  hRatio = 20 / newImg->h;
  newImg = rotozoomSurfavce(newImg, 0, wRatio, hRatio, 0);
  return newImg;
  //int SDL_SaveBMP(SDL_Surface *surface, const char *file);
}*/

/*SDL_Surface* redimension(int *arraym SDL_Surface* img){
  SDL_surface* newImg = img;
  size_t w = 0;
  size_t h = 0;
  int b = 1;
  for(size_t y = 0; *array != 42; ++y){
    for(size_t x = 0; *array != 20; ++x){
      if(b)
	++w;
      ++array;
    }
    ++h;
    b = 0;
  }
  size_t wRatio = 20 / img->w;
  size_t hRatio = 20 / img->h;
  newImg = rotozoomSurfaceXY(newImg, 0, wRatio, hRatio, 0);
}*/
