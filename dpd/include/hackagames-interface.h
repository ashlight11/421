#ifndef HACKAGAME_INTERFACE_H
#define HACKAGAME_INTERFACE_H

//-----------------------------------//
//--          Interface            --//
//-----------------------------------//
#include <pthread.h>

struct Str_Interface {
    char* name;
    Float2 camera, screenCenter;
    int frameWidth, frameHeight;
    float scale; // pixel per meter
    Organism* tabletop;
    Float2 cursor;
    pthread_t thread;
    bool isReady;
    // Mask to select the attributs to show on the IHM
    int attributMask_size; // mean that all the attributs would be displayed
    int* attributMask;
//    pthread_t thread;
};
typedef struct Str_Interface Interface;

//-----------------------------------//
//--   Constructor / Destructor    --//
//-----------------------------------//

/**  CONSTRUCT
 * 
 * @brief     Construc the @param self Interface with @param frameWidth, @param frameHeight and @param scale for camera.
 */
void Interface_construct(Interface * self, Organism* tabletop, int frameWidth, int frameHeight, float scale);

/**  NEW
 * 
 * @brief     Allocate the memory to store a Interface
 * @return    The pointer to the new Interface.
 */
Interface * Interface_new(Organism* tabletop, int frameWidth, int frameHeight, float scale);
Interface * Interface_newBasic(); // a new Interface with default attributs

/**  DISTROY
 * 
 * @brief    distroy the elements of a cell.
 * @param    self the Organism not to distroy.
 */
void Interface_distroy( Interface * self );

/**  DELETE
 * 
 * @brief    distroy and delete a cell.
 * @param    self the Organism not to delete.
 */
void Interface_delete( Interface * self );

//-----------------------------------//
//--       Camera managment        --//
//-----------------------------------//

// Camera Conversion
Vector2 Interface_pixelFromPosition(Interface * self, Float2 p); // > framFromTabletop
Float2 Interface_positionFromPixel(Interface * self, Vector2 p); // > tabletopFromFrame

// To String
char* Interface_str(Interface * self, char* buffer);

//-----------------------------------//
//--     Raylib IHM Interface      --//
//-----------------------------------//

// Game Interface
void Interface_startIHM(Interface* self);
void Interface_stopIHM(Interface* self);
bool Interface_IHMIsOpen(Interface* self);

// Configuration
int* Interface_setAttributMask_ofSize(Interface* self, int* mask, int mask_size);

// Rendering
void Interface_draw(Interface * self);
void Interface_drawBasis(Interface * self);
void Interface_drawCursor(Interface * self);
void Interface_drawTabletop(Interface * self, Organism * aTtop);
void Interface_drawCell(Interface * self, Organism * aOrganism);
void Interface_drawPiece(Interface * self, Float2 basis, Organism * piece);
void Interface_drawEdge(Interface * self, Organism * aOrganism, Organism * aTarget);

// Control
void Interface_control(Interface * self);
void Interface_controlCursor(Interface * self);
void Interface_controlCamera(Interface * self);

#endif