#include <stdio.h>
#include <stdlib.h>

#define size 9  //2n+1
#define target_i 5
#define target_j 3

float i_cord = 2;
float j_cord = 1;
int dir = 1;

int grid[size][size];

typedef struct node* NODE;
NODE head = NULL;
NODE tail = NULL;

struct node
{
    int x;
    int y;
    NODE next;
};

void enqueue(int x, int y)
{
    
    NODE new_node;
    new_node = (NODE)malloc(sizeof(struct node));

    new_node -> x = x;
    new_node -> y = y;
    new_node -> next = NULL;

    if (tail != NULL)
    {
        tail->next = new_node;
    }

    tail = new_node;
}

void dequeue()
{
    NODE temp = head;
    head = head->next;
    free(temp);
}

void print_queue()
{
    NODE temp = head;
    while (temp->next != NULL)
    {
        printf("(%d,%d)", temp->x, temp->y);
        temp = temp->next;
    }
    printf("\n");
}

void flood_fill(int x, int y)
{
    enqueue(x,y);
    head = tail;

    int steps = 1;
    int i,j;

    i = head->x;
    j = head->y;

    grid[i][j] = 1;
    
    while (head != NULL)
    {
        i = head->x;
        j = head->y;
        
        int l = grid[i+1][j];

        if(!grid[i-1][j]) //One step above
        {
            grid[i-1][j] = grid[i][j]+1;
            enqueue(i-1,j);
        }
        if(!grid[i+1][j]) //One step below
        {
            grid[i+1][j] = grid[i][j]+1;
            enqueue(i+1,j);
        }
        if(!grid[i][j+1]) //One step left
        {
            grid[i][j+1] = grid[i][j]+1;
            enqueue(i,j+1);
        }
        if(!grid[i][j-1]) //One step right
        {
            grid[i][j-1] = grid[i][j]+1;
            enqueue(i,j-1);
        }

        for (int i = 0; i<size; i ++)
            {
                for (int j = 0; j<size; j++)
                {
                    printf("%d  ", grid[i][j]);
                }
                printf("\n");
            } 

        dequeue();
        //print_queue();
    }
}

void initiate() // To be saved in memory later to reduce time complexity
{
    for (int i = 1; i<size-1; i ++)
    {
        for (int j = 1; j<size-1; j++)
        {
            grid[i][j] = 0;
        }
    }

    for (int i = 2; i<size-1; i = i+2)
    {
        for (int j = 2; j<size-1; j = j+2)
        {
            grid[i][j] = 69;
        }
    }

    for (int i = 0; i<size; i ++)
    {
        grid[0][i] = 69;
        grid[i][0] = 69;
        grid[size-1][i] = 69;
        grid[i][size-1] = 69;
    }
}

void reset() // Reset after each move and run flood fill again
{
  for (int i = 1; i<size-1; i ++)
    {
        for (int j = 1; j<size-1; j++)
        {
            if(grid[i][j] != 69)
            {
                grid[i][j] = 0;
            }
        }
    }  
}

int best_choice() // If two similar choices available case to be written
{
    int cell_i = (int) i_cord;
    int cell_j= (int) j_cord;

    int options[4];

    options[0] = grid[cell_i][cell_j+1]; // East 1
    options[1] = grid[cell_i+1][cell_j]; // South 2
    options[2] = grid[cell_i][cell_j-1]; // West 3
    options[3] = grid[cell_i-1][cell_j]; // North 4

    int min_index;
    int min = 69;

    for(int i = 0; i<4; i++)
    {
        if(options[i] < min)
        {
            min = options[i];
            min_index = i;
        }
    }

    return min_index + 1;
}

void forward() // Nehaal required
{

}

void orient()
{

}

void celebrate()
{

}

int isRight()
{   
    int cell_i = (int) i_cord;
    int cell_j= (int) j_cord;

    if (1/*Condition for wall*/)
    {
        switch (dir)
        {
            case 1:
                grid[cell_i+1][cell_j] = 69;
                break;
            case 2:
                grid[cell_i][cell_j-1] = 69;
                break;
            case 3:
                grid[cell_i-1][cell_j] = 69;
                break;
            case 4:
                grid[cell_i][cell_j+1] = 69;
                break;
        }
    }
}

int isLeft()
{
    int cell_i = (int) i_cord;
    int cell_j= (int) j_cord;

    if (1/*Condition for wall*/)
    {
        switch (dir)
        {
            case 1:
                grid[cell_i-1][cell_j] = 69;
                break;
            case 2:
                grid[cell_i][cell_j+1] = 69;
                break;
            case 3:
                grid[cell_i+1][cell_j] = 69;
                break;
            case 4:
                grid[cell_i][cell_j-1] = 69;
                break;
        }
    }
}

int isForward()
{
    int cell_i = (int) i_cord;
    int cell_j= (int) j_cord;

    if (1/*Condition for wall*/)
    {
        switch (dir)
        {
            case 1:
                grid[cell_i][cell_j+1] = 69;
                break;
            case 2:
                grid[cell_i+1][cell_j] = 69;
                break;
            case 3:
                grid[cell_i][cell_j-1] = 69;
                break;
            case 4:
                grid[cell_i-1][cell_j] = 69;
                break;
        }
    }
}

int main()
{
    initiate();
    
    
    for (int i = 1; i<6; i++)
    {
        grid[i][5] = 69;
    }
    for (int j = 1; j<3; j++)
    {
        grid[5][j] = 69;
    }

    flood_fill(target_i, target_j);

    for (int i = 0; i<size; i ++)
    {
        for (int j = 0; j<size; j++)
        {
            printf("%d   ", grid[i][j]);
        }
        printf("\n");
    } 

    printf("The best choice is %d \n", best_choice());
}


void setup ()
{
    initiate();
}

void loop ()
{
    int choice;
    while ((int) i_cord != target_i && (int) j_cord != target_j)
    {
        reset();

        isRight();
        isLeft();
        isForward();

        orient();

        flood_fill(target_i, target_j);
        choice = best_choice();
        
        forward(choice);
    }

    celebrate();
}