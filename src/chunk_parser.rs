use std::fmt;

use ndarray::{Array3, ArrayViewMut2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use smallvec::SmallVec;

use super::policy_names::POLICY_NAMES;

use super::{dataloader::BatchItem, file_parser::CHUNK_CAPACITY};

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

pub struct ChunkParser {
    pub buffer: Vec<u8>,
    curr_idx: usize,
}

impl Default for ChunkParser {
    fn default() -> Self {
        Self {
            buffer: Vec::with_capacity(CHUNK_CAPACITY),
            curr_idx: Default::default(),
        }
    }
}

impl ChunkParser {
    pub fn next_item(&mut self, diff_focus_params: &mut DiffFocusCalculator) -> Option<BatchItem> {
        let data_array: &[LeelaV6Data] = unsafe { transmute_slice(&self.buffer) };

        if self.curr_idx == data_array.len() {
            return None;
        }

        let curr_data = &data_array[self.curr_idx];

        assert!(curr_data.version == 6 && curr_data.format == 1);

        self.curr_idx += 1;
        if curr_data.should_skip(diff_focus_params) {
            return self.next_item(diff_focus_params);
        }

        Some(curr_data.to_batchitem())
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.curr_idx = 0;
    }
}

/// # Safety
/// as this is basically a transmute, as long as the data is properly aligned we should be good
pub unsafe fn transmute_slice<T, U>(slice: &[T]) -> &[U] {
    let (a, to_return, c) = slice.align_to();
    assert!(a.is_empty() && c.is_empty());
    to_return
}

#[repr(packed)]
#[derive(Clone)]
pub struct LeelaV6Data {
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
    _invariance_info: u8,
    _dep_result: u8,
    root_q: f32,
    best_q: f32,
    _root_d: f32,
    _best_d: f32,
    _root_m: f32,
    _best_m: f32,
    plies_left: f32,
    result_q: f32,
    result_d: f32,
    _played_q: f32,
    _played_d: f32,
    _played_m: f32,
    orig_q: f32,
    _orig_d: f32,
    _orig_m: f32,
    _visits: u32,
    _played_idx: u16,
    _best_idx: u16,
    policy_kld: f32,
    short_q_target: f32,
}

// adjust the short-term q target to account for future positions
pub fn generate_q_target(chunk: &mut [LeelaV6Data], lambda: f32, weight_threshold: f32) {
    for i in 0..chunk.len() {
        let games = &chunk[i..];
        let mut value = 0.0;
        for (board_idx, board) in games.iter().enumerate() {
            let exponent = board_idx;
            let score_weight = lambda.powi(exponent as i32);

            if score_weight < weight_threshold {
                break;
            }

            let score = if board_idx % 2 == 0 {
                -board.root_q
            } else {
                board.root_q
            };

            value += score * score_weight
        }
        value *= 1.0 - lambda;
        chunk[i].short_q_target = value;
    }
}

impl LeelaV6Data {
    pub fn to_batchitem(&self) -> BatchItem {
        let probabilities = self.probabilities;
        let (planes, start) = self.to_planes().into_raw_vec_and_offset();
        assert!(start.is_some_and(|a| a == 0));
        BatchItem {
            input_planes: planes,
            policy_target: probabilities.to_vec(),
            wdl_target: self.wdl_target(),
            q_target: self.short_q_target,
            mlh_target: self.plies_left,
        }
    }

    fn to_planes(&self) -> Array3<f32> {
        let mut planes = Array3::zeros((112, 8, 8));
        assert!(planes.dim() == (112, 8, 8));

        fn unpack_bitboard(bitboard: u64, slice: &mut ArrayViewMut2<f32>) {
            if bitboard == u64::MAX {
                slice.fill(1.0);
                return;
            }
            // leela data is a little strange in that it doesn't exactly follow bitorder. So, we need to swap everything.
            let mut bitboard = bitboard.reverse_bits().swap_bytes();
            while bitboard != 0 {
                let square = bitboard.trailing_zeros() as usize;
                bitboard &= bitboard - 1;
                slice[[square / 8, 7 - (square % 8)]] = 1.0;
            }
        }

        planes.fill(0.0);
        let packed_planes = self.planes;
        for (&bitboard, mut plane) in packed_planes.iter().zip(planes.axis_iter_mut(Axis(0))) {
            unpack_bitboard(bitboard, &mut plane);
        }

        // plane 104: can we castle queenside
        // plane 105: can we castle kingside
        // plane 106: can they castle queenside
        // plane 107: can they castle kingside
        planes
            .index_axis_mut(Axis(0), 104)
            .fill((self.castling_q_us) as f32);
        planes
            .index_axis_mut(Axis(0), 105)
            .fill(((self.castling_k_us) & 1) as f32);
        planes
            .index_axis_mut(Axis(0), 106)
            .fill(((self.castling_q_them) & 1) as f32);
        planes
            .index_axis_mut(Axis(0), 107)
            .fill((self.castling_k_them) as f32);

        // plane 108: whether it is black to move
        planes
            .index_axis_mut(Axis(0), 108)
            .fill(self.stm_or_ep as f32);

        // plane 109: percentage of the way to a 50 move draw
        planes
            .index_axis_mut(Axis(0), 109)
            .fill(self.halfmove_clock as f32 / 100.0);

        // plane 110: all zeros
        // plane 111: all ones
        planes.index_axis_mut(Axis(0), 111).fill(1.0);
        // plane 112: all zeros

        planes
    }

    fn wdl_target(&self) -> [f32; 3] {
        let error = 1e-2;
        assert!(-1.0 - error <= self.result_q && 1.0 + error >= self.result_q);
        assert!(0.0 - error <= self.result_d && 1.0 + error >= self.result_d);
        let result_d = self.result_d.clamp(0.0, 1.0);
        let result_q = self.result_q.clamp(-1.0, 1.0);
        [
            0.5 * (1.0 - result_d + result_q),
            result_d,
            0.5 * (1.0 - result_d - result_q),
        ]
    }
}

// During training, randomly skip certain positions depending on how 'hard' they are, which is calculated based on the difference
// between the expected outcome of the neural net and the actual outcome of the search. (from lc0-training)
// q_weight and pol_scale are used to appropriately weight the relative difficulty of predicting the q
// and policy values respectively. diff_focus parameters control how likely it is that a position will be skipped
pub struct DiffFocusCalculator {
    q_weight: f32,
    pol_scale: f32,
    diff_focus_min: f32,
    diff_focus_slope: f32,
    rng: StdRng,
}

impl DiffFocusCalculator {
    pub fn new(diff_focus_min: f32, diff_focus_slope: f32) -> Self {
        DiffFocusCalculator {
            q_weight: 6.0,
            pol_scale: 3.5,
            diff_focus_min,
            diff_focus_slope,
            rng: StdRng::from_entropy(),
        }
    }
}

impl LeelaV6Data {
    pub fn should_skip(&self, diff_focus_params: &mut DiffFocusCalculator) -> bool {
        if self.orig_q.is_finite() {
            let DiffFocusCalculator {
                q_weight,
                pol_scale,
                diff_focus_min,
                diff_focus_slope,
                rng,
            } = diff_focus_params;
            let q_diff = (self.orig_q - self.best_q).abs();
            let total = (q_diff * *q_weight + self.policy_kld) / (*q_weight + *pol_scale);
            let threshold = *diff_focus_min + *diff_focus_slope * total;

            if threshold < 1.0 && rng.gen_range(0.0..1.0) > threshold {
                return true;
            }
        }

        false
    }
}

impl fmt::Display for LeelaV6Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(self.version == 6 && self.format == 1);
        const PIECE_NAMES: [char; 12] =
            ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];

        let mut pieces_array = ['.'; 64];
        let planes = self.planes;

        for (piecetype, bitboard) in planes[..12].iter().cloned().enumerate() {
            let mut bitboard = bitboard.reverse_bits().swap_bytes();
            while bitboard != 0 {
                let square = bitboard.trailing_zeros();
                bitboard &= bitboard - 1;
                pieces_array[square as usize] = PIECE_NAMES[piecetype];
            }
        }

        for rank in (0..8).rev() {
            write!(f, "{}  ", if self.stm_or_ep == 0 { rank + 1 } else { 8 - rank })?;
            let rank_chars = &pieces_array[(8 * rank)..(8 * rank + 8)];
            for item in rank_chars.iter() {
                write!(f, "{}  ", item)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "   a  b  c  d  e  f  g  h\n")?;
        let (root_q, best_q, result_q) = (self.root_q, self.best_q, self.result_q);
        writeln!(f, "Root Q: {}", root_q)?;
        writeln!(f, "Best Q: {}", best_q)?;
        writeln!(f, "Result Q: {}", result_q)?;
        writeln!(f, "Probabilities:")?;

        let probabilities = self.probabilities;

        let mut sorted_probabilities: SmallVec<[(usize, f32); 64]> = SmallVec::new();

        for (idx, &policy) in probabilities.iter().enumerate() {
            if policy > 0.0 {
                sorted_probabilities.push((idx, policy));
            }
        }

        sorted_probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (idx, policy) in sorted_probabilities {
            writeln!(f, "{}: {}", POLICY_NAMES[idx], policy)?;
        }
        Ok(())
    }
}
