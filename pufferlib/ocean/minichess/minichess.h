#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "raylib.h"

const int w_WIN = 1.0;
const int b_WIN = -1.0;
const int RANKS = 5;
const int FILES = 5;
const int PIECE_WIDTH = 90;
const int PIECE_HEIGHT = 90;

const int EMPTY = 0;
const int PAWN = 1;
const int KNIGHT = 2;
const int BISHOP = 3;
const int ROOK = 4;
const int QUEEN = 5;
const int KING = 6;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct MiniChess MiniChess;
struct MiniChess {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;
    Client* client;

    // Bit string representation from:
    //  https://www.chessprogramming.org/Bitboards
    uint64_t w_P; // Pawns
    uint64_t b_P;
    uint64_t w_N; // Knights
    uint64_t b_N;
    uint64_t w_B; // Bishops
    uint64_t b_B;
    uint64_t w_R; // Rooks
    uint64_t b_R;
    uint64_t w_Q; // Queens
    uint64_t b_Q;
    uint64_t w_K; // Kings
    uint64_t b_K;

    int prev_action;

    int color_to_play; // 1 for white, -1 for black
    int tick;
    int w_can_castle;
    int b_can_castle;
    int repetitions;
    int no_progress_count;
};

void allocate(MiniChess* env) {
    env->observations = (float*)calloc(5 * 5 * 12 + 7, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->rewards = (float*)calloc(1, sizeof(float));
}

void free_allocated(MiniChess* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
}

void c_close(MiniChess* env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
    free_allocated(env);
    free(env);
}

void add_log(MiniChess* env) {
    env->log.perf += env->rewards[0];
    env->log.score += env->rewards[0];
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += env->log.episode_length;
    env->log.n += 1;
}

void init(MiniChess* env) {
    env->log = (Log){0};
    env->tick = 0;
}

int get_piece_at(const MiniChess* game, int rank, int file) {
    if (rank < 0 || rank >= 5 || file < 0 || file >= 5) {
        return EMPTY;  // out of bounds
    }

    int square_index = rank * 5 + file;
    uint64_t mask = 1ULL << square_index;

    if (game->w_P & mask) return PAWN;
    if (game->b_P & mask) return -PAWN;
    if (game->w_K & mask) return KING;
    if (game->b_K & mask) return -KING;
    if (game->w_N & mask) return KNIGHT;
    if (game->b_N & mask) return -KNIGHT;
    if (game->w_B & mask) return BISHOP;
    if (game->b_B & mask) return -BISHOP;
    if (game->w_R & mask) return ROOK;
    if (game->b_R & mask) return -ROOK;
    if (game->w_Q & mask) return QUEEN;
    if (game->b_Q & mask) return -QUEEN;

    return EMPTY;
}

static inline void set_observation(float* obs, int index) {
    obs[index] = 1.0f;
}

// Helper: iterate over set bits in a bitboard
static inline void set_bits(float* obs, uint64_t bitboard, int piece_type, int color_offset) {
    while (bitboard) {
        int sq = __builtin_ctzll(bitboard);  // index of least significant set bit
        bitboard &= bitboard - 1;            // clear that bit

        int rank = sq / FILES;
        int file = sq % FILES;

        int obs_index = rank * FILES * 12 + file * 12 + color_offset + piece_type - 1;
        set_observation(obs, obs_index);
    }
}

void compute_observations(MiniChess* env) {
    float* obs = env->observations;
    memset(obs, 0, sizeof(float) * (RANKS * FILES * 12 + 6));

    // Player 1 (white): color_offset = 0
    set_bits(obs, env->w_P, PAWN, 0);
    set_bits(obs, env->w_N, KNIGHT, 0);
    set_bits(obs, env->w_B, BISHOP, 0);
    set_bits(obs, env->w_R, ROOK, 0);
    set_bits(obs, env->w_Q, QUEEN, 0);
    set_bits(obs, env->w_K, KING, 0);

    // Player 2 (black): color_offset = 6
    set_bits(obs, env->b_P, PAWN, 6);
    set_bits(obs, env->b_N, KNIGHT, 6);
    set_bits(obs, env->b_B, BISHOP, 6);
    set_bits(obs, env->b_R, ROOK, 6);
    set_bits(obs, env->b_Q, QUEEN, 6);
    set_bits(obs, env->b_K, KING, 6);

    // Extra features (1D section at the end)
    int base = RANKS * FILES * 12;
    obs[base + 0] = env->color_to_play == 1 ? 1.0f : 0.0f;
    obs[base + 1] = env->w_can_castle ? 1.0f : 0.0f;
    obs[base + 2] = env->b_can_castle ? 1.0f : 0.0f;
    obs[base + 3] = env->repetitions > 0 ? 1.0f : 0.0f;
    obs[base + 4] = env->no_progress_count > 0 ? 1.0f : 0.0f;
    obs[base + 5] = env->tick / 100.0f;
}

uint64_t square(int rank, int file) {
    return 1ULL << (rank * FILES + file);
}

void init_board(MiniChess* env) {
    env->w_P = square(1, 0) | square(1, 1) | square(1, 2) | square(1, 3) | square(1, 4);
    env->b_P = square(3, 0) | square(3, 1) | square(3, 2) | square(3, 3) | square(3, 4);
    env->w_N = square(0, 1);
    env->b_N = square(4, 1);
    env->w_B = square(0, 2);
    env->b_B = square(4, 2);
    env->w_R = square(0, 0);
    env->b_R = square(4, 0);
    env->w_Q = square(0, 3);
    env->b_Q = square(4, 3);
    env->w_K = square(0, 4);
    env->b_K = square(4, 4);
}

void c_reset(MiniChess* env) {
    env->log = (Log){0};
    env->terminals[0] = 0;
    init_board(env);
    compute_observations(env);
}

void c_step(MiniChess* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = env->actions[0];
    int source = action / 49;
    int move = action % 49;

    if (env->action_)
    compute_observations(env);
}


const Color BLACK_SQUARE = (Color){120, 100, 100, 255};
const Color WHITE_SQUARE = (Color){241, 241, 241, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    
    Texture2D w_P; // Pawns
    Texture2D b_P;
    Texture2D w_N; // Knights
    Texture2D b_N;
    Texture2D w_B; // Bishops
    Texture2D b_B;
    Texture2D w_R; // Rooks
    Texture2D b_R;
    Texture2D w_Q; // Queens
    Texture2D b_Q;
    Texture2D w_K; // Kings
    Texture2D b_K;

};

Client* make_client() {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = FILES * PIECE_WIDTH;
    client->height = RANKS * PIECE_WIDTH;

    InitWindow(client->width, client->height, "PufferLib Ray MiniChess");
    SetTargetFPS(60);

    client->w_P = LoadTexture("resources/minichess/wPawn.png");
    client->b_P = LoadTexture("resources/minichess/bPawn.png");
    client->w_N = LoadTexture("resources/minichess/wKnight.png");
    client->b_N = LoadTexture("resources/minichess/bKnight.png");
    client->w_B = LoadTexture("resources/minichess/wBishop.png");
    client->b_B = LoadTexture("resources/minichess/bBishop.png");
    client->w_R = LoadTexture("resources/minichess/wRook.png");
    client->b_R = LoadTexture("resources/minichess/bRook.png");
    client->w_Q = LoadTexture("resources/minichess/wQueen.png");
    client->b_Q = LoadTexture("resources/minichess/bQueen.png");
    client->w_K = LoadTexture("resources/minichess/wKing.png");
    client->b_K = LoadTexture("resources/minichess/bKing.png");

    return client;
}

void c_render(MiniChess* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = make_client();
    }

    Client* client = env->client;

    BeginDrawing();
    ClearBackground(WHITE_SQUARE);
    
    Texture2D piece_texture;
    Color bg_color;
    Color piece_color;
    int y_offset = client->height - PIECE_HEIGHT;

    for (int rank = 0; rank < RANKS; rank++) {
        for (int file = 0; file < FILES; file++) {
            int y = y_offset - rank * PIECE_HEIGHT;
            int x = file * PIECE_WIDTH;

            bg_color = (rank + file) % 2 == 1 ? BLACK_SQUARE: WHITE_SQUARE;
            DrawRectangle(x, y, PIECE_WIDTH, PIECE_HEIGHT, bg_color);
            Rectangle src = {0, 0, 45, 45};
            Rectangle dest = {x, y, PIECE_WIDTH, PIECE_HEIGHT};
            Vector2 origin = {0, 0 };

            int piece = get_piece_at(env, rank, file);
           
            if (piece > 0) {
                piece_color = (Color) WHITE;
                switch (piece) {
                    case PAWN: piece_texture = client->w_P; break;
                    case KNIGHT: piece_texture = client->w_N; break;
                    case BISHOP: piece_texture = client->w_B; break;
                    case ROOK: piece_texture = client->w_R; break;
                    case QUEEN: piece_texture = client->w_Q; break;
                    case KING: piece_texture = client->w_K; break;
                }
                DrawTexturePro(piece_texture, src, dest, origin, 0.0f, piece_color);
            } 
            else if (piece < 0) {
                piece_color = (Color) BLACK;
                switch (-piece) {
                    case PAWN: piece_texture = client->b_P; break;
                    case KNIGHT: piece_texture = client->b_N; break;
                    case BISHOP: piece_texture = client->b_B; break;
                    case ROOK: piece_texture = client->b_R; break;
                    case QUEEN: piece_texture = client->b_Q; break;
                    case KING: piece_texture = client->b_K; break;
                }
                DrawTexturePro(piece_texture, src, dest, origin, 0.0f, bg_color);
            }
        }
    }
    EndDrawing();
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}
