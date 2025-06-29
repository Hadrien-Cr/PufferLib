#include "minichess_utils.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <inttypes.h>

const int PIECE_WIDTH = 90;
const int PIECE_HEIGHT = 90;

typedef struct Log {
    float win_rate;
    float episode_length;
    float episode_return;
    float n;
} Log;

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

typedef struct MiniChess {
    Client* client;
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int episode_return;
    ZobristTable3D* zobrist_table;
    ChessBoard* board;
} MiniChess;


void init(MiniChess* env) {
    env->zobrist_table = (ZobristTable3D*)calloc(1, sizeof(ZobristTable3D));
    env->board = (ChessBoard*)calloc(1, sizeof(ChessBoard));
    zobrist_init(env->zobrist_table, BOARD_SIZE * BOARD_SIZE, PIECE_TYPES, 3, 5118);
    chessboard_allocate(env->board);
}

void allocate(MiniChess* env) {
    init(env);
    env->observations = (float*)calloc(BOARD_SIZE*BOARD_SIZE+6, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(MiniChess* env) {
    zobrist_destroy(env->zobrist_table);
    chessboard_destroy(env->board);
}

void free_allocated(MiniChess* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(MiniChess* env, int val) {
    env->log.win_rate += val;
    env->log.episode_length += env->episode_return;
    env->log.episode_length += env->board->move_number / 2;
    env->log.n += 1;
}

void compute_observations(MiniChess* env) {
    int offset = 6;
    for (int idx = 0; idx < BOARD_SIZE*BOARD_SIZE; idx++){
        Piece piece = env->board->pieces[idx];
        if (piece.type == pEmpty){
            continue;
        };
        env->observations[offset + idx] =  6 * piece.color * PIECE_TYPES + piece.type;
    }
}

void c_reset(MiniChess* env) {
    env->episode_return = 0;
    chessboard_init(env->zobrist_table, env->board);
    compute_and_cache_legal_actions(env->zobrist_table, env->board);
    compute_observations(env);
}

int env_move(MiniChess* env) {
    return sample_random_action(env->board);
}

void step_board(MiniChess* env, int action){
    apply_action(env->zobrist_table, env->board, action);
    compute_and_cache_legal_actions(env->zobrist_table, env->board);
}

void c_step(MiniChess* env) {
    int action = env->actions[0];
    env->rewards[0] = 0;

    if (!env->board->action_mask[action]){
        action = sample_random_action(env->board);
        env->rewards[0] -= 0.1;
    }

    step_board(env, action);
    GameResult result = chessboard_get_result(env->board);

    if (result == WhiteWin){
        env->rewards[0] += 1;
        env->episode_return += 1;
    }
    else if (result == BlackWin) {
        env->rewards[0] += -1;
        env->episode_return += 1;
    }
    if (result != NotEnded){
        add_log(env, (result + 1) / (float) 2);
        c_reset(env);
    }

    else{
        step_board(env, env_move(env));
        result = chessboard_get_result(env->board);

        if (result == WhiteWin){
            env->rewards[0] += 1;
            env->episode_return += 1;
        }
        else if (result == BlackWin) {
            env->rewards[0] += -1;
            env->episode_return += 1;
        }
        if (result != NotEnded){
            add_log(env, (result + 1) / (float) 2);
            c_reset(env);
        }
    }
    compute_observations(env);
} 


Piece get_piece_at(const MiniChess* env, Square sq) {
    return env->board->pieces[SquareToIndex(sq)];
}

const Color BLACK_SQUARE = (Color){120, 100, 100, 255};
const Color WHITE_SQUARE = (Color){241, 241, 241, 255};

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

Client* make_client(MiniChess* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = BOARD_SIZE * PIECE_WIDTH;
    client->height = BOARD_SIZE * PIECE_WIDTH;

    InitWindow(client->width, client->height, "PufferLib Ray MiniChess");
    SetTargetFPS(1);

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
        env->client = make_client(env);
    }

    Client* client = env->client;

    BeginDrawing();
    ClearBackground(WHITE_SQUARE);
    
    Texture2D piece_texture;
    Color bg_color;
    Color piece_color;
    int y_offset = client->height - PIECE_HEIGHT;
    
    for (int rank = 0; rank < BOARD_SIZE; rank++) {
        for (int file = 0; file < BOARD_SIZE; file++) {
            int y = y_offset - rank * PIECE_HEIGHT;
            int x = file * PIECE_WIDTH;
            bg_color = (rank + file) % 2 == 1 ? BLACK_SQUARE: WHITE_SQUARE;
            DrawRectangle(x, y, PIECE_WIDTH, PIECE_HEIGHT, bg_color);
            Rectangle src = {0, 0, 45, 45};
            Rectangle dest = {x, y, PIECE_WIDTH, PIECE_HEIGHT};
            Vector2 origin = {0, 0 };
            
            Square sq = {file,rank};
            Piece piece = get_piece_at(env, sq);
            
            if ((piece.color == cEmpty) || (piece.type == pEmpty)) {
                continue;
            }
            else if (piece.color == cWhite) {
                piece_color = (Color) WHITE;
                switch (piece.type) {
                    case pPawn: piece_texture = client->w_P; break;
                    case pKnight: piece_texture = client->w_N; break;
                    case pBishop: piece_texture = client->w_B; break;
                    case pRook: piece_texture = client->w_R; break;
                    case pQueen: piece_texture = client->w_Q; break;
                    case pKing: piece_texture = client->w_K; break;
                    case pEmpty: piece_texture = client->w_K; break;
                }
                DrawTexturePro(piece_texture, src, dest, origin, 0.0f, piece_color);
            } 
            else if (piece.color == cBlack) {
                piece_color = (Color) BLACK;
                switch (piece.type) {
                    case pPawn: piece_texture = client->b_P; break;
                    case pKnight: piece_texture = client->b_N; break;
                    case pBishop: piece_texture = client->b_B; break;
                    case pRook: piece_texture = client->b_R; break;
                    case pQueen: piece_texture = client->b_Q; break;
                    case pKing: piece_texture = client->b_K; break;
                    case pEmpty: piece_texture = client->b_K; break;
                }
                DrawTexturePro(piece_texture, src, dest, origin, 0.0f, bg_color);
            }
        }
    }
    EndDrawing();
}