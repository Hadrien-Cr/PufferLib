#include <time.h>
#include "minichess.h"
#include "puffernet.h"

void demo() {
    MiniChess env = {
    };
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);
    while (!WindowShouldClose()) {
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    close_client(env.client);
}

void test_performance(int timeout) {
    MiniChess env = {
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 1225;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    //demo();
    test_performance(10);
}
