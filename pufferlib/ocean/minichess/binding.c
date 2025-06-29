#include "minichess.h"
#define Env MiniChess
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "win_rate", log->win_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
