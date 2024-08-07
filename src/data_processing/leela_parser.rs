// Description of lc0 v6 format from the github (8356 bytes total):
//                                 size         1st byte index
//     uint32_t version;                               0
//     uint32_t input_format;                          4
//     float probabilities[1858];  7432 bytes          8
//     uint64_t planes[104];        832 bytes       7440
//     uint8_t castling_us_ooo;                     8272
//     uint8_t castling_us_oo;                      8273
//     uint8_t castling_them_ooo;                   8274
//     uint8_t castling_them_oo;                    8275
//     uint8_t side_to_move_or_enpassant;           8276
//     uint8_t rule50_count;                        8277
//
//     invariance_info is bitfield with the following allocation:
//         bit 7: side to move (input type 3)
//         bit 6: position marked for deletion by the rescorer (never set by lc0)
//         bit 5: game adjudicated (v6)
//         bit 4: max game length exceeded (v6)
//         bit 3: best_q is for proven best move (v6)
//         bit 2: transpose transform (input type 3)
//         bit 1: mirror transform (input type 3)
//         bit 0: flip transform (input type 3)
//
//     uint8_t invariance_info;                     8278
//     uint8_t dep_result;                          8279
//     float root_q;                                8280
//     float best_q;                                8284
//     float root_d;                                8288
//     float best_d;                                8292
//     float root_m;      // In plies.              8296
//     float best_m;      // In plies.              8300
//     float plies_left;                            8304
//     float result_q;                              8308
//     float result_d;                              8312
//     float played_q;                              8316
//     float played_d;                              8320
//     float played_m;                              8324
// The folowing may be NaN if not found in cache.
//     float orig_q;      // For value repair.      8328
//     float orig_d;                                8332
//     float orig_m;                                8336
//     uint32_t visits;                             8340
// Indices in the probabilities array.
//     uint16_t played_idx;                         8344
//     uint16_t best_idx;                           8346
//     uint64_t reserved;                           8348


#[repr(packed)]
struct LeelaV6Data {
    version: u32,
    format: u32,
    probabilities: [f32; 1858],
    planes: [u64; 104],
    castling_q_us: u8,
    castling_k_us: u8,
    castling_q_them: u8,
    castling_k_them: u8,
    stm_or_ep: u8,
    halfmove_clock: u8,
    invariance_info: u8,
    dep_result: u8,
    root_q: f32,
    best_q: f32,
    root_d: f32,
    best_d: f32,
    root_m: f32,
    best_m: f32,
    plies_left: f32,
    result_q: f32,
    result_d: f32,
    played_q: f32,
    played_d: f32,
    played_m: f32,
    orig_q: f32,
    orig_d: f32,
    orig_m: f32,
    visits: u32,
    played_idx: u16,
    best_idx: u16,
    reserved: u64,
}

