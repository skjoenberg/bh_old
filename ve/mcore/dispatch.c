#include <cphvb.h>
#include <cphvb_compute.h>
#include "dispatch.h"
#include <assert.h>
#include <pthread.h>

//The data type of cphvb_array extended with temporary info.
typedef struct
{
    CPHVB_ARRAY_HEAD
    //Saved original values
    cphvb_index      org_start;
    cphvb_index      org_shape[CPHVB_MAXDIM];

} dispatch_ary;

//Returns the offset based on the current block.
inline cphvb_intp get_offset(cphvb_intp block, cphvb_instruction *inst,
                             cphvb_intp nblocks)
{
    if(block == 0)
        return 0;

    //We compute the offset based on the output operand.
    dispatch_ary *ary = (dispatch_ary*) inst->operand[0];

    return ary->org_shape[0] / nblocks * block + //Whole blocks
           ary->org_shape[0] % nblocks;//The reminder.
}

//Returns the shape based on the current block.
inline cphvb_intp get_shape(cphvb_intp block, cphvb_instruction *inst,
                            cphvb_intp nblocks)
{
    dispatch_ary *ary = (dispatch_ary*) inst->operand[0];

    //We block over the most significant dimension
    //and the first block gets the reminder.
    if(block == 0)
    {
        return ary->org_shape[0] / nblocks +
               ary->org_shape[0] % nblocks;
    }
    else
    {
        return ary->org_shape[0] / nblocks;
    }
}

//Shared thread variables.
static pthread_t thd_ids[32];
static pthread_cond_t  thd_cond_wait     = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  thd_cond_finished = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t thd_mutex         = PTHREAD_MUTEX_INITIALIZER;
static int thd_wait         = 1;//0==start,1==wait
static int thd_finished[2]  = {0,0};//Whan it reaches nthd we are finished.
static int thd_exit         = 0;//1==exit,0=running
static cphvb_intp thd_nblocks;
static cphvb_intp thd_nthds;
static cphvb_intp thd_size;
static cphvb_instruction** thd_inst_bundle;
static computeloop *thd_traverses;
static int main_finished_idx = 0;//Finished index for main.

//The thread function.
static void *thd_do(void *msg)
{
    cphvb_intp myid = ((cphvb_intp)msg);
    int finished_idx = 0;//Finished index (swapped between iterations).
    while(1)
    {
        //Wait for start signal.
        pthread_mutex_lock(&thd_mutex);
        while(thd_wait)
            pthread_cond_wait(&thd_cond_wait,&thd_mutex);
        pthread_mutex_unlock(&thd_mutex);

        if(thd_exit)//Check if we should exit.
            break;

        cphvb_intp nthds = thd_nthds;
        cphvb_intp size = thd_size;
        cphvb_intp nblocks = thd_nblocks;
        cphvb_instruction** inst_bundle = thd_inst_bundle;
        computeloop *traverses = thd_traverses;

        if(nthds > nblocks)
            nblocks = nthds;//Minimum one block per thread.

        //Check if this thread has any work to do.
        if(myid < nthds)
        {
            //We will not block a single instruction into more blocks
            //than threads.
            if(thd_size == 1)
                if(nblocks > nthds)
                    nblocks = nthds;

            cphvb_intp length = nblocks / nthds; // Find this thread's length of work.
            cphvb_intp start = myid * length;    // Find this thread's start block.
            if(myid == nthds-1)
                length += nblocks % nthds;       // The last thread gets the reminder.

            //Clone the instruction and make new views of all operands.
            cphvb_instruction thd_inst[CPHVB_MAX_NO_INST];
            cphvb_array ary_stack[CPHVB_MAX_NO_INST*CPHVB_MAX_NO_OPERANDS];
            cphvb_intp ary_stack_count=0;
            for(cphvb_intp j=0; j<size; ++j)
            {
                thd_inst[j] = *inst_bundle[j];
                cphvb_intp nops = cphvb_operands(inst_bundle[j]->opcode);
                for(cphvb_intp i=0; i<nops; ++i)
                {
                    cphvb_array *ary_org = inst_bundle[j]->operand[i];
                    cphvb_array *ary = &ary_stack[ary_stack_count++];
                    *ary = *ary_org;

                    if(ary_org->base == NULL)//base array
                    {
                        ary->base = ary_org;
                    }

                    thd_inst[j].operand[i] = ary;//Save the operand.
                 }
            }

            //Handle one block at a time.
            for(cphvb_intp b=start; b<start+length; ++b)
            {
                //Update the operands to match the current block.
                for(cphvb_intp j=0; j<size; ++j)
                {
                    cphvb_instruction *inst = &thd_inst[j];
                    cphvb_intp nops = cphvb_operands(inst->opcode);
                    for(cphvb_intp i=0; i<nops; ++i)
                    {
                        dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
                        ary->start = ary->org_start + ary->stride[0] *
                                     get_offset(b, inst, nblocks);
                        ary->shape[0] = get_shape(b, inst, nblocks);
                    }
                }

                //Dispatch a block.
                for(cphvb_intp j=0; j<size; ++j)
                {
                    cphvb_instruction *inst = &thd_inst[j];
                    if(inst->operand[0]->shape[0] <= 0)
                        break;//Nothing to do.

                    inst->status = traverses[j](inst);
                    if(inst->status != CPHVB_SUCCESS)
                    {
                        //ret = CPHVB_PARTIAL_SUCCESS;
                        start = length;//Force a complete exit.
                        break;
                    }
                }
            }
        }

        //Signal that we are finished.
        pthread_mutex_lock(&thd_mutex);
        if(++thd_finished[finished_idx] == thd_nthds)//We are the last to finish.
        {
            finished_idx = (finished_idx+1)%2;//Swap the index.
            //Reset signals and signal all threads.
            thd_wait                   = 1;//0==start,1==wait
            thd_finished[finished_idx] = 0;//Whan it reaches nthd we are finished.
            pthread_cond_broadcast(&thd_cond_finished);
        }
        else//Lets wait for the others to finish.
        {
            while(thd_finished[finished_idx] < thd_nthds)
                pthread_cond_wait(&thd_cond_finished,&thd_mutex);
            finished_idx = (finished_idx+1)%2;//Swap the index.
        }
        pthread_mutex_unlock(&thd_mutex);
    }
    pthread_exit(NULL);
    
    //Fix compiler warning/error with VC
    return NULL;
}

//Initiate the dispather.
cphvb_error dispatch_init(void)
{
    char *env = getenv("CPHVB_NUM_THREADS");
    if(env != NULL)
        thd_nthds = atoi(env);
    else
        thd_nthds = 1;

    if(thd_nthds > 32)
    {
        fprintf(stderr,"CPHVB_NUM_THREADS greater than 32!\n");
        thd_nthds = 32;//MAX 32 thds.
    }

    //Create all threads.
    for(long i=0; i<thd_nthds; ++i)
        pthread_create(&thd_ids[i], NULL, thd_do, (void *) (i));
    return CPHVB_SUCCESS;
}

//Finalize the dispather.
cphvb_error dispatch_finalize(void)
{
    //Kill all threads.
    pthread_mutex_lock(&thd_mutex);
    thd_wait = 0;//Set the start signal.
    thd_exit = 1;//Set the exit signal.
    pthread_cond_broadcast(&thd_cond_wait);
    pthread_mutex_unlock(&thd_mutex);

    //Join all threads.
    for(int i=0; i<thd_nthds; ++i)
        pthread_join(thd_ids[i], NULL);
    return CPHVB_SUCCESS;
}

//Dispatch the bundle of instructions.
cphvb_error dispatch_bundle(cphvb_instruction** inst_bundle,
                            cphvb_intp size,
                            cphvb_intp nblocks)
{
    cphvb_error ret = CPHVB_SUCCESS;

    //Get all traverse function -- one per instruction.
    computeloop traverses[CPHVB_MAX_NO_INST];
    for(cphvb_intp j=0; j<size; ++j)
    {
        traverses[j] = cphvb_compute_get( inst_bundle[j] );
        if(traverses[j] == NULL)
        {
            inst_bundle[j]->status = CPHVB_OUT_OF_MEMORY;
            ret = CPHVB_INST_NOT_SUPPORTED;
            goto finish;
        }
    }
    //Save the original array information.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        cphvb_intp nops = cphvb_operands(inst->opcode);
        for(cphvb_intp i=0; i<nops; ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->org_start = ary->start;
            memcpy(ary->org_shape, ary->shape, ary->ndim * sizeof(cphvb_index));
        }
    }
    //Make sure that all array-data is allocated.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        cphvb_intp nops = cphvb_operands(inst->opcode);
        for(cphvb_intp i=0; i<nops; ++i)
        {
            if(cphvb_data_malloc(inst->operand[i]) != CPHVB_SUCCESS)
            {
                inst->status = CPHVB_OUT_OF_MEMORY;
                ret = CPHVB_PARTIAL_SUCCESS;
                goto finish;
            }
        }
    }

    pthread_mutex_lock(&thd_mutex);
        //Init shared thread variables.
        thd_nblocks      = nblocks;
        thd_size         = size;
        thd_inst_bundle  = inst_bundle;
        thd_traverses    = traverses;

        //Start all threads.
        thd_wait = 0;//False is the start signal.
        pthread_cond_broadcast(&thd_cond_wait);

        //Wait for them to finish.
        while(thd_finished[main_finished_idx] < thd_nthds)
            pthread_cond_wait(&thd_cond_finished,&thd_mutex);

        //Swap the finished index.
        main_finished_idx = (main_finished_idx+1)%2;
    pthread_mutex_unlock(&thd_mutex);

finish:
/*
    This is not needed anymore.
    //Restore the original arrays.
    for(cphvb_intp j=0; j<size; ++j)
    {
        cphvb_instruction *inst = inst_bundle[j];
        for(cphvb_intp i=0; i<cphvb_operands(inst->opcode); ++i)
        {
            dispatch_ary *ary = (dispatch_ary*) inst->operand[i];
            ary->start = ary->org_start;
            memcpy(ary->shape, ary->org_shape, ary->ndim * sizeof(cphvb_index));
        }
    }
*/
    return ret;
}