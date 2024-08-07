use std::{mem, slice};

use numpy::ndarray::{Array1, Array3, Axis};

use super::{dataloader::BatchItem, file_parser::CHUNK_CAPACITY};

pub const BOARD_DATA_SIZE: usize = mem::size_of::<BoardData>();
pub const POLICY_SIZE: usize = mem::size_of::<Policy>();

#[repr(packed)]
#[derive(Clone)]
pub struct BoardData {
    pub boards: [PackedBoard; 8],
    pub repetition_mask: u8,
    pub castling_mask: u8,
    pub halfmove_clock: u8,
    pub is_flipped: bool,
    pub moves_left: u16,
    pub game_result: i8,
    pub q_target: f32, // uncertainty head target
    pub num_children: u8,
}

impl BoardData {
    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() == BOARD_DATA_SIZE);
        unsafe { &*(bytes.as_ptr() as *const BoardData) }.clone()
    }

    pub fn to_bytes(&self, buffer: &mut Vec<u8>) {
        let byte_slice = unsafe {
            std::slice::from_raw_parts(self as *const BoardData as *const u8, BOARD_DATA_SIZE)
        };

        buffer.extend_from_slice(byte_slice);
    }
}

#[repr(packed)]
#[derive(Clone, Copy)]
pub struct PackedBoard {
    pub occupancy: u64,
    pub pieces: [u8; 16],
}

#[repr(packed)]
#[derive(Clone)]
pub struct Policy {
    pub idx: u16,
    pub value: f32,
}

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
    pub fn next_item(&mut self) -> Option<BatchItem> {
        let mut batch_item = BatchItem::default();
        let mut bytes = &self.buffer[self.curr_idx..];
        if bytes.len() < BOARD_DATA_SIZE {
            assert!(bytes.is_empty());
            return None;
        }

        let (board_data_bytes, rem_bytes) = bytes.split_at(BOARD_DATA_SIZE);

        bytes = rem_bytes;

        let board_data = unsafe { BoardData::from_bytes(board_data_bytes) };

        let policy_size = POLICY_SIZE * (board_data.num_children as usize);

        let policy_bytes = &bytes[..policy_size];
        let policy_list: &[Policy] = unsafe { reinterpret_slice(policy_bytes).unwrap() };

        assert!(policy_list.len() == board_data.num_children as usize);

        let bytes_read = BOARD_DATA_SIZE + policy_size;

        board_data.fill_planes(&mut batch_item.input_planes);
        write_policy_target(policy_list, &mut batch_item.policy_target);

        batch_item.wdl_target = board_data.wdl_target();

        batch_item.mlh_target = board_data.moves_left as f32;

        batch_item.q_target = board_data.q_target;

        self.curr_idx += bytes_read;
        Some(batch_item)
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        self.curr_idx = 0;
    }
}

impl BoardData {
    fn fill_planes(&self, planes: &mut Array3<f32>) {
        // First 104 planes are for history
        let plane_chunk_iters = planes.axis_chunks_iter_mut(Axis(0), 13); // 8 total boards, each with 12 normal planes + 1 repetition plane
        for (board_idx, (board, mut history_planes)) in
            self.boards.iter().zip(plane_chunk_iters).enumerate()
        {
            let mut occupancy = board.occupancy;
            for i in 0..occupancy.count_ones() {
                let square = occupancy.trailing_zeros() as usize;
                occupancy &= occupancy - 1;

                let byte = board.pieces[i as usize / 2];
                let plane_idx = byte >> (4 * (i % 2));
                history_planes[[plane_idx as usize, square / 8, square % 8]] = 1.0;
            }
            let is_repetition = (self.repetition_mask >> board_idx) & 1;
            history_planes
                .index_axis_mut(Axis(0), 12)
                .fill(is_repetition as f32);
        }
        // plane 104: can we castle queenside
        // plane 105: can we castle kingside
        // plane 106: can they castle queenside
        // plane 107: can they castle kingside
        planes
            .index_axis_mut(Axis(0), 104)
            .fill((self.castling_mask & 1) as f32);
        planes
            .index_axis_mut(Axis(0), 105)
            .fill(((self.castling_mask >> 1) & 1) as f32);
        planes
            .index_axis_mut(Axis(0), 106)
            .fill(((self.castling_mask >> 2) & 1) as f32);
        planes
            .index_axis_mut(Axis(0), 107)
            .fill((self.castling_mask >> 3) as f32);

        // plane 108: whether it is black to move
        planes
            .index_axis_mut(Axis(0), 108)
            .fill(self.is_flipped as u8 as f32);

        // plane 109: percentage of the way to a 50 move draw
        planes
            .index_axis_mut(Axis(0), 109)
            .fill(self.halfmove_clock as f32 / 100.0);

        // plane 110: all zeros
        // plane 111: all ones
        planes.index_axis_mut(Axis(0), 111).fill(1.0);
        // plane 112: all zeros
    }

    fn wdl_target(&self) -> [f32; 3] {
        let mut wdl_array = [0.0; 3];
        wdl_array[self.game_result as usize + 1] = 1.0;
        wdl_array
    }
}

fn write_policy_target(policy_slice: &[Policy], policy_target: &mut Array1<f32>) {
    assert!(policy_target.dim() == 1858);
    policy_target.fill(0.0);
    for &Policy { idx, value } in policy_slice {
        policy_target[idx as usize] = value;
    }
}

unsafe fn reinterpret_slice<U: Sized, T: Sized>(source: &[U]) -> Option<&[T]> {
    let source_item_size = mem::size_of::<U>();
    let dest_item_size = mem::size_of::<T>();

    let source_slice_size = mem::size_of_val(source);
    
    if source_item_size == 0
        || dest_item_size == 0
        || source_slice_size % dest_item_size != 0
        || (source.as_ptr() as usize) % mem::align_of::<T>() != 0
    {
        return None;
    }

    let new_slice_size = source_slice_size / dest_item_size;
    let new_slice = slice::from_raw_parts(source.as_ptr() as *const T, new_slice_size);
    Some(new_slice)
}
