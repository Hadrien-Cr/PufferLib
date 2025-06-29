#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "raylib.h"

#define BOARD_SIZE 5
#define PIECE_TYPES 6
#define N_ACTIONS 1225
#define MAX_LENGTH 512
#define MAX_MOVES_NO_CAPTURES_NO_PM 50

#define SIGN(i) (((i) == 0) ? 0 : ((i) > 0 ? 1 : -1))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABS(x) (((x) < 0) ? -(x) : (x))

typedef struct Offset {
    int x_offset;
    int y_offset;
} Offset;

typedef struct Square {
    int x;
    int y;
} Square;

const Offset kKnightOffsets[8] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
    { 2, -1}, { 2, 1}, { 1, -2}, { 1, 2}
};

const Offset kQueenOffsets[32] = {
    { 4, 0}, { 3, 0}, { 2, 0}, { 1, 0},
    {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0},

    { 0, 4}, { 0, 3}, { 0, 2}, { 0, 1},
    { 0,-4}, { 0,-3}, { 0,-2}, { 0,-1},

    { 4, 4}, { 3, 3}, { 2, 2}, { 1, 1},
    {-4,-4}, {-3,-3}, {-2,-2}, {-1,-1},

    {-4, 4}, {-3, 3}, {-2, 2}, {-1, 1},
    { 4,-4}, { 3,-3}, { 2,-2}, { 1,-1}
};

const int kRookOffsetsIdx[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
const int kBishopOffsetsIdx[16] = {16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
const int kKingOffsetsIdx[8] = {3, 7, 11, 15, 19, 23, 27, 31};
const int kPawnOffsetsIdx[6] = {11, 15, 19, 23, 27, 31};

typedef enum { kAcknowledgeEnemyPieces, kBreachEnemyPieces} PseudoLegalMoveSettings;
typedef enum  { cWhite = 0, cBlack = 1, cEmpty = 2 } PieceColor;
typedef enum {
    pKing = 0,
    pQueen = 1,
    pRook = 2,
    pBishop = 3,
    pKnight = 4,
    pPawn = 5,
    pEmpty = 6,
} PieceType;

typedef enum {
    BlackWin = 0,
    Draw = 1,
    WhiteWin = 2,
    NotEnded = 3,
} GameResult;

typedef struct Piece {
    PieceColor color;
    PieceType type;
} Piece;

const Piece StartContent[BOARD_SIZE*BOARD_SIZE] = {
    (Piece){cWhite,pRook},(Piece){cWhite,pKnight},(Piece){cWhite,pBishop},(Piece){cWhite,pQueen},(Piece){cWhite,pKing},
    (Piece){cWhite,pPawn},(Piece){cWhite,pPawn},(Piece){cWhite,pPawn},(Piece){cWhite,pPawn},(Piece){cWhite,pPawn},
    (Piece){cEmpty,pEmpty},(Piece){cEmpty,pEmpty},(Piece){cEmpty,pEmpty},(Piece){cEmpty,pEmpty},(Piece){cEmpty,pEmpty},
    (Piece){cBlack,pPawn},(Piece){cBlack,pPawn},(Piece){cBlack,pPawn},(Piece){cBlack,pPawn},(Piece){cBlack,pPawn},
    (Piece){cBlack,pRook},(Piece){cBlack,pKnight},(Piece){cBlack,pBishop},(Piece){cBlack,pQueen},(Piece){cBlack,pKing},
};

typedef struct Move {
    Square from;
    Square to;
    Piece piece;
    PieceType promotion_type;
    int is_castle;
} Move;

typedef struct {
    uint64_t* data;;
    size_t dim1, dim2, dim3;
} ZobristTable3D;

typedef struct ChessBoard {
    PieceColor current_player;
    int move_number;
    int move_no_capture_or_pm;
    Square WhiteKingSquare;
    Square BlackKingSquare;
    Piece* pieces;
    int* action_mask;
    uint64_t* prev_zobrist_hashes;
} ChessBoard;

static inline PieceColor color_to_play(ChessBoard* board){
    return (board->move_number %2 == 0) ? cWhite : cBlack;
}

static inline PieceColor color_opponent(ChessBoard* board){
    return (board->move_number %2 == 0) ? cBlack : cWhite;
}

ChessBoard* clone_chessboard(const ChessBoard* src) {
    ChessBoard* clone = malloc(sizeof(ChessBoard));
    clone->current_player = src->current_player;
    clone->move_number = src->move_number;
    clone->WhiteKingSquare = src->WhiteKingSquare;
    clone->BlackKingSquare = src->BlackKingSquare;
    clone->move_no_capture_or_pm = src->move_no_capture_or_pm;

    clone->pieces = (Piece*)calloc(BOARD_SIZE * BOARD_SIZE, sizeof(Piece));
    memcpy(clone->pieces, src->pieces, BOARD_SIZE * BOARD_SIZE * sizeof(Piece));
    clone->action_mask = (int*)calloc(N_ACTIONS, sizeof(int));
    memcpy(clone->action_mask, src->action_mask, N_ACTIONS * sizeof(int));
    clone->prev_zobrist_hashes = (uint64_t*)calloc(5, sizeof(uint64_t));
    memcpy(clone->prev_zobrist_hashes, src->prev_zobrist_hashes, 5 * sizeof(uint64_t));
    return clone;
}

Square IndexToSquare(int idx){ return (Square){idx % BOARD_SIZE, idx/BOARD_SIZE}; }
int SquareToIndex(Square sq){ return sq.y * BOARD_SIZE + sq.x; }

static inline Square sum(Square sq, Offset offset) {
    int x = sq.x + offset.x_offset;
    int y = sq.y + offset.y_offset;
    return (Square){x, y};
}

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline size_t zobrist_index(size_t x, size_t y, size_t z, size_t dim2, size_t dim3) {
    return x * dim2 * dim3 + y * dim3 + z;
}

static inline void zobrist_init(ZobristTable3D* zobrist_table, size_t dim1, size_t dim2, size_t dim3, uint64_t seed) {
    size_t total = dim1 * dim2 * dim3;
    zobrist_table->data = calloc(total, sizeof(uint64_t));
    zobrist_table->dim1 = dim1;
    zobrist_table->dim2 = dim2;
    zobrist_table->dim3 = dim3;
    uint64_t state = seed;
    for (size_t i = 0; i < total; ++i) {
        zobrist_table->data[i] = (uint64_t)xorshift64(&state);
    }
}

static inline uint64_t zobrist_get(const ZobristTable3D* table, size_t x, size_t y, size_t z) {
    size_t idx = zobrist_index(x, y, z, table->dim2, table->dim3);
    return table->data[idx];
}

static inline void set_square(ZobristTable3D* zobrist_table, ChessBoard* board, Square sq, Piece piece) {
    int position = SquareToIndex(sq);
    Piece current_piece = board->pieces[position];
    board->pieces[position] = piece;

    uint64_t cur_hash = board->prev_zobrist_hashes[4];
    uint64_t next_hash = cur_hash;
    next_hash ^= zobrist_get(zobrist_table, position, current_piece.color, current_piece.type);
    next_hash ^= zobrist_get(zobrist_table, position, piece.color, piece.type);
    board->prev_zobrist_hashes[4] = next_hash;
}

static inline void chessboard_allocate(ChessBoard* board){
    board->pieces = (Piece*)calloc(BOARD_SIZE*BOARD_SIZE, sizeof(Piece));
    board->prev_zobrist_hashes = (uint64_t*)calloc(5, sizeof(uint64_t)); 
    board->action_mask = (int*)calloc(N_ACTIONS, sizeof(int));
}

static inline void chessboard_init(ZobristTable3D* zobrist_table, ChessBoard* board){
    board->current_player = cWhite;
    board->move_number = 0;
    board->move_no_capture_or_pm = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        set_square(zobrist_table, board, IndexToSquare(i), StartContent[i]);
    }
    board->WhiteKingSquare = (Square) {4,0};
    board->BlackKingSquare = (Square) {4,4};
}


static inline void chessboard_destroy(ChessBoard* board){
    free(board->pieces);
    free(board->prev_zobrist_hashes);
    free(board->action_mask);
    free(board);
}

static inline void zobrist_destroy(ZobristTable3D* zobrist_table){
    free(zobrist_table->data);
    free(zobrist_table);
}

static inline bool is_in_range(Square sq) {
    return ((sq.x >= 0) && (sq.x < BOARD_SIZE) && (sq.y >= 0) && (sq.y < BOARD_SIZE));
}

static inline bool is_empty(ChessBoard* board, Square sq) {
    return (board->pieces[SquareToIndex(sq)].color == cEmpty);
}

static inline bool is_capturable(ChessBoard* board, Square sq, PieceColor color) {
    return (board->pieces[SquareToIndex(sq)].color != color);
}

static inline bool is_forward(Offset offset, PieceColor color) {
    return ((color == cWhite) && (offset.y_offset > 0)) ||((color == cBlack) && (offset.y_offset < 0));
}

static inline bool is_promotion(ChessBoard* board, Square sq, Offset offset, PieceColor color) {
    if (!(board->pieces[SquareToIndex(sq)].type == pPawn)){
        return false;
    }
    return ((color == cWhite) && ((sq.y + offset.y_offset) == (BOARD_SIZE - 1))) ||((color == cBlack) && (sq.y + offset.y_offset) == 0);
}

static inline bool is_diagonal(Offset offset) {
    return (offset.x_offset != 0);
}

static inline bool is_reachable(ChessBoard* board, Square sq, Offset offset, PieceColor color) {
    if (!is_capturable(board, sum(sq,offset), color)){
        return false;
    }    
    int k = MAX(ABS(offset.x_offset), ABS(offset.y_offset));
    int dx = SIGN(offset.x_offset);
    int dy = SIGN(offset.y_offset);

    for (int i = 1; i < k; i++){
        if (!is_empty(board, sum(sq, (Offset){i*dx, i*dy}))){
            return false;
        }
    }
    return true;
}

static inline bool has_sufficient_material(ChessBoard* board) {
    int white_bishops = 0, white_knights = 0, white_other = 0;
    int black_bishops = 0, black_knights = 0, black_other = 0;
    int white_bishop_square_colors[2] = {0, 0};
    int black_bishop_square_colors[2] = {0, 0};

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        Piece piece = board->pieces[i];
        if (piece.type == pEmpty || piece.type == pKing) continue;

        int color_index = ((i / BOARD_SIZE + i % BOARD_SIZE) % 2 == 0) ? 0 : 1; 

        if (piece.color == cWhite) {
            switch (piece.type) {
                case pBishop:
                    white_bishops++;
                    white_bishop_square_colors[color_index]++;
                    break;
                case pKnight:
                    white_knights++;
                    break;
                default:
                    white_other++;
                    break;
            }
        } else {
            switch (piece.type) {
                case pBishop:
                    black_bishops++;
                    black_bishop_square_colors[color_index]++;
                    break;
                case pKnight:
                    black_knights++;
                    break;
                default:
                    black_other++;
                    break;
            }
        }
    }

    bool white_has_only_single_color_bishops = (white_bishops > 0 && white_bishop_square_colors[0] == 0) ||
                                               (white_bishops > 0 && white_bishop_square_colors[1] == 0);

    bool black_has_only_single_color_bishops = (black_bishops > 0 && black_bishop_square_colors[0] == 0) ||
                                               (black_bishops > 0 && black_bishop_square_colors[1] == 0);

    bool white_insufficient = (white_other == 0) &&
        ((white_bishops == 0 && white_knights <= 1) ||
         (white_bishops > 0 && white_knights == 0 && white_has_only_single_color_bishops));

    bool black_insufficient = (black_other == 0) &&
        ((black_bishops == 0 && black_knights <= 1) ||
         (black_bishops > 0 && black_knights == 0 && black_has_only_single_color_bishops));

    return !(white_insufficient && black_insufficient);
}

static inline void apply_move(ZobristTable3D* zobrist_table, ChessBoard* board, Square sq, Offset offset) {
    PieceColor current_color = color_to_play(board);
    if (current_color==cWhite) {
        if (board->WhiteKingSquare.x == sq.x && board->WhiteKingSquare.y == sq.y){
            board->WhiteKingSquare = sum(sq, offset);
        }
    }   
    if (current_color==cBlack) {
        if (board->BlackKingSquare.x == sq.x && board->BlackKingSquare.y == sq.y){
            board->BlackKingSquare = sum(sq, offset);
        }
    } 
    for (int i = 3; i >= 0; i--) {
        board->prev_zobrist_hashes[i] = board->prev_zobrist_hashes[i + 1];
    }
    board->move_number++;
    board->move_no_capture_or_pm++;
    // TraceLog(LOG_INFO, "Act %d // %d %d // %d %d", is_in_range(sum(sq, offset)), sq.x, sq.y, offset.x_offset, offset.y_offset );
    if (!is_empty(board, sum(sq, offset)) ||  board->pieces[SquareToIndex(sq)].type == pPawn){
        board->move_no_capture_or_pm = 0;
    }
    set_square(zobrist_table, board, sum(sq, offset), board->pieces[SquareToIndex(sq)]);
    set_square(zobrist_table, board, sq, (Piece){cEmpty, pEmpty});
}

static inline void apply_action(ZobristTable3D* zobrist_table, ChessBoard* board, int action) {
    Square sq = IndexToSquare(action/49);
    int move = action%49;
    if (move < 8){
        return apply_move(zobrist_table, board, sq, kKnightOffsets[move]);
    }
    else if (move<40) {
        if (is_promotion(board, sq, kQueenOffsets[move-8], color_to_play(board))){
            set_square(zobrist_table, board, sq, (Piece){color_to_play(board), pQueen});
        }
        return apply_move(zobrist_table, board, sq, kQueenOffsets[move-8]);
    }
    else {
        int x_offset = ((move-40)/3) - 1;
        Offset o = {x_offset, (color_to_play(board) == cWhite) ? 1 : -1};
        set_square(zobrist_table, board, sq, (Piece){color_to_play(board), pRook + (move-40) % 3});
        return apply_move(zobrist_table, board, sq, o);
    }
}


static inline bool is_in_check(ChessBoard* board, PieceColor current_color){
    PieceColor opponent_color = (current_color == cWhite) ? cBlack : cWhite;
    Square kingSquare = (current_color == cWhite) ? board->WhiteKingSquare : board->BlackKingSquare;

    int i;
    
    for (int idx = 0; idx < BOARD_SIZE * BOARD_SIZE; idx++){
        Square sq = IndexToSquare(idx);
        Piece p = board->pieces[idx];
        
        if (p.color != opponent_color) continue;

        if (p.type == pKnight){
            for (int i = 0; i < 8; i++) {
                Offset o = kKnightOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y){
                    return true;
                }
            }
        }
        else if (p.type == pQueen){
            for (int i = 0; i < 32; i++) {
                Offset o = kQueenOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y && is_reachable(board, sq, o, opponent_color)){
                    return true;
                }
            }
        }
        else if (p.type == pRook){
            for (int u = 0; u < 16; u++) {
                i = kRookOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y && is_reachable(board, sq, o, opponent_color)){
                    return true;
                }
            }
        }
        else if (p.type == pBishop){
            for (int u = 0; u < 16; u++) {
                i = kBishopOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y && is_reachable(board, sq, o, opponent_color)){
                    return true;
                }
            }
        }
        else if (p.type == pPawn){
            for (int u = 0; u < 6; u++) {
                i = kPawnOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y && is_forward(o, opponent_color) && is_diagonal(o)){
                    return true;
                }
            }
        }
        else if (p.type == pKing){
            for (int u = 0; u < 8; u++) {
                i = kKingOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (sum(sq, o).x == kingSquare.x && sum(sq, o).y == kingSquare.y){
                    return true;
                }
            }
        }
    }
    return false;
}

bool is_legal_action(ChessBoard* board, int action, ZobristTable3D* zobrist_table) {
    ChessBoard* next_state = clone_chessboard(board);
    apply_action(zobrist_table, next_state, action);
    bool legal = !is_in_check(next_state, color_to_play(board));
    chessboard_destroy(next_state);
    return legal;
    return true;
}


static inline void compute_and_cache_legal_actions(ZobristTable3D* zobrist_table, ChessBoard* board){
    memset(board->action_mask, 0, N_ACTIONS * sizeof(int)); 
    int i;
    PieceColor current_color = color_to_play(board);

    for (int idx = 0; idx < BOARD_SIZE * BOARD_SIZE; idx++){
        Square sq = IndexToSquare(idx);
        Piece p = board->pieces[idx];
        if (p.color != color_to_play(board)){
            continue;
        }
        else if (p.type == pKnight){
            for (int i = 0; i < 8; i++) {
                Offset o = kKnightOffsets[i];
                if (is_in_range(sum(sq, o)) && is_capturable(board, sum(sq, o), current_color)  && is_legal_action(board, 49*idx + i, zobrist_table)){
                    board->action_mask[49*idx + i] = 1;
                }
            }
        }
        else if (p.type == pQueen){
            for (int i = 0; i < 32; i++) {
                Offset o = kQueenOffsets[i];
                if (is_in_range(sum(sq, o)) && is_reachable(board, sq, o, current_color) &&  is_legal_action(board, 49*idx + 8 + i, zobrist_table)){
                    board->action_mask[49*idx + 8 + i] = 1;
                }
            }
        }
        else if (p.type == pRook){
            for (int u = 0; u < 16; u++) {
                i = kRookOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (is_in_range(sum(sq, o)) && is_reachable(board, sq, o, current_color) &&  is_legal_action(board, 49*idx + 8 + i, zobrist_table)){
                    board->action_mask[49*idx + 8 + i] = 1;
                }
            }
        }
        else if (p.type == pBishop){
            for (int u = 0; u < 16; u++) {
                i = kBishopOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (is_in_range(sum(sq, o)) && is_reachable(board, sq, o, current_color) && is_legal_action(board, 49*idx + 8 + i, zobrist_table)){
                    board->action_mask[49*idx + 8 + i] = 1;
                }
            }
        }
        else if (p.type == pPawn){
            for (int u = 0; u < 6; u++) {
                i = kPawnOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (is_in_range(sum(sq, o)) && is_forward(o, current_color) && 
                    ((!is_diagonal(o) && is_empty(board, sum(sq, o))) || 
                     (is_diagonal(o) && !is_empty(board, sum(sq, o)) && is_capturable(board, sum(sq, o), current_color))) &&
                     is_legal_action(board, 49*idx + 8 + i, zobrist_table)
                ) {
                    board->action_mask[49*idx + 8 + i] = 1;
                    if (is_promotion(board, sq, o, current_color)) { 
                        board->action_mask[49*idx + 40 + 3 + 3 * o.x_offset    ] = 1;
                        board->action_mask[49*idx + 40 + 3 + 3 * o.x_offset + 1] = 1;
                        board->action_mask[49*idx + 40 + 3 + 3 * o.x_offset + 2] = 1;
                    }
                }
            }
        }
        else if (p.type == pKing){
            for (int u = 0; u < 8; u++) {
                i = kKingOffsetsIdx[u];
                Offset o = kQueenOffsets[i];
                if (is_in_range(sum(sq, o)) && is_reachable(board, sq, o, current_color) && is_legal_action(board, 49*idx + 8 + i, zobrist_table)){
                    board->action_mask[49*idx + 8 + i] = 1;
                }
            }
        }
    }
}

static inline bool draw_by_repetition(ChessBoard* board){
    if (board->move_number < 8) return false;
    uint64_t current_hash = board->prev_zobrist_hashes[4];
    int repetition_count = 1;
    for (int i = 0; i < 4; i++) {
        if (board->prev_zobrist_hashes[i] == current_hash) {
            repetition_count++;
        }
    }
    return repetition_count >= 3;
}

static inline bool draw_by_no_captures_or_pm(ChessBoard* board){
    return board->move_no_capture_or_pm >= MAX_MOVES_NO_CAPTURES_NO_PM;
}

static inline bool draw_by_truncation(ChessBoard* board){
    return board->move_number >= MAX_LENGTH;
}

static inline bool has_legal_moves(ChessBoard* board){
    for (size_t i = 0; i < N_ACTIONS; ++i) {
        if (board->action_mask[i] != 0) {
            return true;
        }
    }
    return false;
}

static inline GameResult chessboard_get_result(ChessBoard* board){
    if (draw_by_repetition(board) || draw_by_no_captures_or_pm(board) || draw_by_truncation(board)|| !has_sufficient_material(board)){
        // if (draw_by_repetition(board)){
        //     TraceLog(LOG_INFO, "Repetation");
        // }
        // if (draw_by_no_captures_or_pm(board)){
        //     TraceLog(LOG_INFO, "No captures");
        // }
        // if (draw_by_truncation(board)){
        //     TraceLog(LOG_INFO, "Truncation");
        // }
        // if (!has_sufficient_material(board)){
        //     TraceLog(LOG_INFO, "No Material");
        // }
        return Draw;
    }

    if (!has_legal_moves(board)) {
        PieceColor player = color_to_play(board);
        if (!is_in_check(board, player)) {
            // TraceLog(LOG_INFO, "Stalemate");
            return Draw;
        } 
        else {
            if (player == cWhite){
                return WhiteWin;
            }
            else {
                return BlackWin;
            };
        }
    }
    return NotEnded;
}

int sample_random_action(ChessBoard* board) {
    int action = -1;
    int count = 0;
    for (size_t i = 0; i < N_ACTIONS; ++i) {
        if (board->action_mask[i]) {
            count++;
            if (rand() % count == 0) {
                action = i;
            }
        }
    }
    return action;
}
