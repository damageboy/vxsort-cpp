use z3::ast::BV;

#[derive(Clone, Debug)]
pub enum Imm8 {
    Literal(u8),
    Expr(BV),
}

impl Imm8 {
    pub fn to_bv(&self) -> BV {
        match self {
            Self::Literal(value) => BV::from_u64(*value as u64, 8),
            Self::Expr(expr) => {
                assert_eq!(expr.get_size(), 8, "Imm8::Expr must be an 8-bit bit-vector");
                expr.clone()
            }
        }
    }
}

// Mimics the standard _MM_SHUFFLE2 intrinsic macro.
// Returns (x << 1) | y
pub fn mm_shuffle2(x: u8, y: u8) -> u8 {
    (x << 1) | y
}

pub fn mm_shuffle(z: u8, y: u8, x: u8, w: u8) -> u8 {
    (z << 6) | (y << 4) | (x << 2) | w
}

// Decodes an 8-bit shuffle mask into _MM_SHUFFLE parameters.
//
// Args:
//     imm8: 8-bit shuffle mask immediate value
//
// Returns:
//     Tuple (z, y, x, w) where:
//     - w = bits [1:0]
//     - x = bits [3:2]
//     - y = bits [5:4]
//     - z = bits [7:6]
//
// Example:
//     >>> decode_shuffle_mask(0x88)
//     (2, 0, 2, 0)
//     >>> decode_shuffle_mask(0xdd)
//     (3, 1, 3, 1)
pub fn decode_shuffle_mask(imm8: u8) -> (u8, u8, u8, u8) {
    let w = imm8 & 0b11;
    let x = (imm8 >> 2) & 0b11;
    let y = (imm8 >> 4) & 0b11;
    let z = (imm8 >> 6) & 0b11;
    (z, y, x, w)
}

// Returns a string representation of a shuffle mask in _MM_SHUFFLE format.
//
// Args:
//     imm8: 8-bit shuffle mask immediate value
//
// Returns:
//     String in format "_MM_SHUFFLE(z, y, x, w)"
//
// Example:
//     >>> mm_shuffle_str(0x88)
//     '_MM_SHUFFLE(2, 0, 2, 0)'
//     >>> mm_shuffle_str(0xdd)
//     '_MM_SHUFFLE(3, 1, 3, 1)'
pub fn mm_shuffle_str(imm8: u8) -> String {
    let (z, y, x, w) = decode_shuffle_mask(imm8);
    format!("_MM_SHUFFLE({z}, {y}, {x}, {w})")
}

// Decode an imm8 shuffle mask for 2-element operations (e.g., shuffle_pd).
//
// For shuffle_pd with 256-bit registers:
// - Bit 0: selects element from first 128-bit lane (0 or 1)
// - Bit 1: selects element from first 128-bit lane (0 or 1)
// - Bit 2: selects element from second 128-bit lane (0 or 1)
// - Bit 3: selects element from second 128-bit lane (0 or 1)
//
// Returns the high and low bits as a tuple (y, x) where:
// - x = bits [1:0] (first lane selection)
// - y = bits [3:2] (second lane selection)
//
// Args:
//     imm8: 8-bit immediate value (only lowest 4 bits used)
//
// Returns:
//     Tuple of (y, x) where each is a 2-bit value (0-3)
pub fn decode_shuffle2_mask(imm8: u8) -> (u8, u8) {
    let x = imm8 & 0b11;
    let y = (imm8 >> 2) & 0b11;
    (y, x)
}

// Returns a string representation of a shuffle mask in _MM_SHUFFLE2 format
// for 2-element operations like shuffle_pd.
//
// Args:
//     imm8: 8-bit shuffle mask immediate value (only lowest 4 bits used)
//
// Returns:
//     String in format "_MM_SHUFFLE2(y, x)"
//
// Example:
//     >>> mm_shuffle2_str(0x0)
//     '_MM_SHUFFLE2(0, 0)'
//     >>> mm_shuffle2_str(0x5)
//     '_MM_SHUFFLE2(1, 1)'
//     >>> mm_shuffle2_str(0xa)
//     '_MM_SHUFFLE2(2, 2)'
pub fn mm_shuffle2_str(imm8: u8) -> String {
    let (y, x) = decode_shuffle2_mask(imm8);
    format!("_MM_SHUFFLE2({y}, {x})")
}
