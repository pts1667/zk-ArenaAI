/* Recording sim games for cloning (tools/policy_clone_data.py), straight from a batch's arrays: libzkrecord.
 *
 * A recorder has slots, one per game of the batch.  A slot's run starts with zkr_begin and ends with zkr_finish, which writes one
 * file holding both sides of the game; every step in between, zkr_frames takes the listed games' frames from the batch's arrays
 * and zkr_orders the orders given at them.  Work is spread over a thread pool of the recorder's own, a game to a thread.
 *
 * The file (sim/py/zkrecord.py reads it):
 *
 *     "ZKREC001", u32 meta length, meta (the caller's JSON),
 *     f32 map_w, map_h, y_lo, y_hi, i32 frames, chunks, chunk_frames, sites,
 *     u32 length, zstd(the game's site records, as given at zkr_begin),
 *     then each chunk: u32 length, zstd(chunk).
 *
 * A chunk is up to chunk_frames frames of both sides: i32 frames, then for side 0 and then side 1: i32 unit counts, feature counts
 * and order counts by frame; the 18 globals; units; features; orders.  Every array is stored a column at a time, each column a byte
 * plane at a time (byte 0 of every value, then byte 1, ...).
 *
 * Units are narrowed to 40 bytes (zkr_unit below).  Positions are fractions of a range in 16 bits: x and z of the map, order
 * targets of [-0.25, 1.25] of it, y of the run's [y_lo, y_hi]; velocities, health, experience and a blip's top speed are half
 * floats; build progress is 16 bits of [0, 1]; an order's age is whole frames.  The rest are exact.  A unit or feature found in
 * the side's previous frame (by uid, feature id) is stored as each field minus its value there, wrapping, with how many records
 * back that was; else as itself with 0.  Globals are stored as their bits minus the previous frame's.  Features keep their metal
 * exactly; orders and sites are exact.
 */
#ifndef ZKRECORD_H
#define ZKRECORD_H

#include <stdint.h>

#include "zksim/capi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ZKR_ABI_VERSION 1
#define ZKR_GLOBALS 18
#define ZKR_CLAMPS 9 /* zkr_clamps: x, z, y, ord_x, ord_z, build_frac, ord_age, a half float out of range, a feature's x or z */

typedef struct {
    int n;                          /* listed games */
    const int32_t* games;           /* their slots, which are their positions in the batch's arrays too */
    const void* records;            /* zk::UnitRecord [positions][2][cap] */
    int cap;
    const int32_t* counts;          /* [positions][2] */
    const float* globs;             /* [positions][2][ZKR_GLOBALS] */
    const void* features;           /* zk::FeatureRecord [positions][2][feature_cap] */
    int feature_cap;
    const int32_t* feature_counts;  /* [positions][2] */
} zkr_frames_args;

int zkr_abi_version(void);
/* level: zstd's.  chunk_frames: frames of a side to a chunk, which bounds the memory a slot holds before compressing. */
void* zkr_new(int slots, int threads, int level, int chunk_frames);
void zkr_free(void* rec);
/* A run starts in the slot, dropping what it held.  y_lo, y_hi: the range y is kept in (clamped outside it).  sites: n_sites
   zk::SiteRecord, the same for every frame of the run. */
void zkr_begin(void* rec, int slot, float map_w, float map_h, float y_lo, float y_hi, const void* sites, int n_sites);
/* Both sides' present frames of the listed games. */
void zkr_frames(void* rec, const zkr_frames_args* args);
/* The orders given at the frames zkr_frames took last, for both sides of the listed games: seat s's (slot * 2 + side) are
   orders[offsets[s]:offsets[s + 1]], offsets over every seat of the recorder. */
void zkr_orders(void* rec, int n, const int32_t* games, const zk_order_rec* orders, const int32_t* offsets);
/* Ends the slot's run: its file at `path`, `meta` at the front, is queued for the recorder's writer thread, which writes it beside
   `path` and renames it.  0, or -1 if the slot has no run.  The slot holds nothing after, until zkr_begin. */
int zkr_finish(void* rec, int slot, const char* path, const char* meta, int meta_len);
/* Waits until every file finished is written: 0, or the errno of the first that failed since the last wait.  zkr_free waits too. */
int zkr_wait(void* rec);
/* Frames the slot's run holds so far. */
int zkr_frames_held(void* rec, int slot);
/* How many values were clamped to fit, over every run so far: ZKR_CLAMPS counts. */
void zkr_clamps(void* rec, int64_t* out);

/* zstd for the reader: the size a frame decompresses to (-1 if unknown), and decompressing it.  The decompressed size, or -1. */
int64_t zkr_frame_size(const void* src, int64_t len);
int64_t zkr_decompress(const void* src, int64_t len, void* dst, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
