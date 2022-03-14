#ifndef HACKAGAMES_GEOMETRY_H
#define HACKAGAMES_GEOMETRY_H

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct Str_Float2 {
    float x;
    float y;
} Float2;

unsigned Color_red(unsigned color);
unsigned Color_green(unsigned color);
unsigned Color_blue(unsigned color);
unsigned Color_alpha(unsigned color);

void Float2_str(Float2 self, char* buffer);

float Float2_length(Float2 self);
float Float2_length2(Float2 self);

float Float2_distance(Float2 self, Float2 another);
float Float2_distance2(Float2 self, Float2 another);

float Float2_normalize(Float2 *self);

void Float2_repultion(Float2 *self, Float2 *another, float dist);

#endif