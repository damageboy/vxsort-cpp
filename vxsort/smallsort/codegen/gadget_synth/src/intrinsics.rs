use std::collections::BTreeMap;

use z3::Solver;
use z3::ast::BV;
use z3_avx::alignr::{
    mm256_alignr_epi8, mm512_alignr_epi32, mm512_alignr_epi64, mm512_mask_alignr_epi32,
    mm512_mask_alignr_epi64,
};
use z3_avx::blend::{mm256_blend_pd, mm256_blend_ps};
use z3_avx::control::Imm8;
use z3_avx::permute::{mm256_permute_ps, mm512_mask_permute_ps, mm512_permute_ps};
use z3_avx::permute_pd::{mm256_permute_pd, mm512_mask_permute_pd, mm512_permute_pd};
use z3_avx::permute2x128::mm256_permute2x128_si256;
use z3_avx::permute4x64::mm256_permute4x64_epi64;
use z3_avx::permutevar::{
    mm256_permutevar_pd, mm256_permutevar_ps, mm512_mask_permutevar_pd, mm512_mask_permutevar_ps,
    mm512_permutevar_pd, mm512_permutevar_ps,
};
use z3_avx::permutex2var::{
    mm512_mask_permutex2var_epi32, mm512_mask_permutex2var_epi64, mm512_permutex2var_epi32,
    mm512_permutex2var_epi64,
};
use z3_avx::permutexvar::{
    mm256_permutexvar_epi32, mm256_permutexvar_epi64, mm512_mask_permutexvar_epi32,
    mm512_mask_permutexvar_epi64, mm512_permutexvar_epi32, mm512_permutexvar_epi64,
};
use z3_avx::shuffle_i32x4::{mm512_mask_shuffle_i32x4, mm512_shuffle_i32x4};
use z3_avx::shuffle_pd::{mm256_shuffle_pd, mm512_mask_shuffle_pd, mm512_shuffle_pd};
use z3_avx::shuffle_ps::{mm256_shuffle_ps, mm512_mask_shuffle_ps, mm512_shuffle_ps};
use z3_avx::unpack::{
    mm256_unpackhi_epi32, mm256_unpackhi_epi64, mm256_unpacklo_epi32, mm256_unpacklo_epi64,
    mm512_mask_unpackhi_epi32, mm512_mask_unpackhi_epi64, mm512_mask_unpacklo_epi32,
    mm512_mask_unpacklo_epi64, mm512_unpackhi_epi32, mm512_unpackhi_epi64, mm512_unpacklo_epi32,
    mm512_unpacklo_epi64,
};

use crate::types::{Arch, DType, GadgetNode, InputRef, IntrinsicNode, Symbolic, SynthesisError};

pub fn single_input_nodes(
    arch: Arch,
    dtype: DType,
    input_name: &'static str,
) -> Vec<IntrinsicNode> {
    let input = GadgetNode::Input(InputRef { name: input_name });
    let tag = input_name;
    match (arch, dtype) {
        (Arch::Avx2, DType::I32) => vec![
            single_imm(
                "_mm256_permute_ps",
                input.clone(),
                format!("imm8_permute_ps_{tag}"),
            ),
            single_imm(
                "_mm256_permute4x64_epi64",
                input.clone(),
                format!("imm8_permute4x64_{tag}"),
            ),
            single_ctrl(
                "_mm256_permutexvar_epi32",
                input.clone(),
                "op_idx",
                format!("ctrl_permutexvar_{tag}"),
                256,
            ),
            single_ctrl(
                "_mm256_permutevar_ps",
                input,
                "b",
                format!("ctrl_permutevar_ps_{tag}"),
                256,
            ),
        ],
        (Arch::Avx2, DType::I64) => vec![
            single_imm(
                "_mm256_permute_pd",
                input.clone(),
                format!("imm8_permute_pd_{tag}"),
            ),
            single_imm(
                "_mm256_permute4x64_epi64",
                input.clone(),
                format!("imm8_permute4x64_{tag}"),
            ),
            single_ctrl(
                "_mm256_permutexvar_epi64",
                input.clone(),
                "op_idx",
                format!("ctrl_permutexvar_{tag}"),
                256,
            ),
            single_ctrl(
                "_mm256_permutevar_pd",
                input,
                "b",
                format!("ctrl_permutevar_pd_{tag}"),
                256,
            ),
        ],
        (Arch::Avx512, DType::I32) => vec![
            single_ctrl(
                "_mm512_permutexvar_epi32",
                input.clone(),
                "op_idx",
                format!("ctrl_permutexvar_{tag}"),
                512,
            ),
            single_imm(
                "_mm512_permute_ps",
                input.clone(),
                format!("imm8_permute_ps_{tag}"),
            ),
            single_ctrl(
                "_mm512_permutevar_ps",
                input.clone(),
                "b",
                format!("ctrl_permutevar_ps_{tag}"),
                512,
            ),
            masked_single_ctrl(
                "_mm512_mask_permutexvar_epi32",
                input.clone(),
                "op_idx",
                format!("ctrl_m_permutexvar_{tag}"),
                512,
                format!("k_mask_permutexvar_{tag}"),
                16,
            ),
            masked_single_imm(
                "_mm512_mask_permute_ps",
                input.clone(),
                format!("imm8_m_permute_ps_{tag}"),
                format!("k_mask_permute_ps_{tag}"),
                16,
            ),
            masked_single_ctrl(
                "_mm512_mask_permutevar_ps",
                input,
                "b",
                format!("ctrl_m_permutevar_ps_{tag}"),
                512,
                format!("k_mask_permutevar_ps_{tag}"),
                16,
            ),
        ],
        (Arch::Avx512, DType::I64) => vec![
            single_ctrl(
                "_mm512_permutexvar_epi64",
                input.clone(),
                "op_idx",
                format!("ctrl_permutexvar_{tag}"),
                512,
            ),
            single_imm(
                "_mm512_permute_pd",
                input.clone(),
                format!("imm8_permute_pd_{tag}"),
            ),
            single_ctrl(
                "_mm512_permutevar_pd",
                input.clone(),
                "b",
                format!("ctrl_permutevar_pd_{tag}"),
                512,
            ),
            masked_single_ctrl(
                "_mm512_mask_permutexvar_epi64",
                input.clone(),
                "op_idx",
                format!("ctrl_m_permutexvar_{tag}"),
                512,
                format!("k_mask_permutexvar_{tag}"),
                8,
            ),
            masked_single_imm(
                "_mm512_mask_permute_pd",
                input.clone(),
                format!("imm8_m_permute_pd_{tag}"),
                format!("k_mask_permute_pd_{tag}"),
                8,
            ),
            masked_single_ctrl(
                "_mm512_mask_permutevar_pd",
                input,
                "b",
                format!("ctrl_m_permutevar_pd_{tag}"),
                512,
                format!("k_mask_permutevar_pd_{tag}"),
                8,
            ),
        ],
    }
}

pub fn dual_input_nodes(
    arch: Arch,
    dtype: DType,
    ref1: &'static str,
    ref2: &'static str,
) -> Vec<IntrinsicNode> {
    let a = GadgetNode::Input(InputRef { name: ref1 });
    let b = GadgetNode::Input(InputRef { name: ref2 });
    let tag = format!("{ref1}_{ref2}");
    match (arch, dtype) {
        (Arch::Avx2, DType::I32) => vec![
            dual_imm(
                "_mm256_shuffle_ps",
                a.clone(),
                b.clone(),
                format!("imm8_shuffle_{tag}"),
                false,
            ),
            dual_plain("_mm256_unpacklo_epi32", a.clone(), b.clone(), false),
            dual_plain("_mm256_unpackhi_epi32", a.clone(), b.clone(), false),
            dual_imm(
                "_mm256_permute2x128_si256",
                a.clone(),
                b.clone(),
                format!("imm8_perm2x128_{tag}"),
                true,
            ),
            dual_imm(
                "_mm256_blend_ps",
                a.clone(),
                b.clone(),
                format!("imm8_blend_{tag}"),
                true,
            ),
            dual_imm(
                "_mm256_alignr_epi8",
                a,
                b,
                format!("imm8_alignr_{tag}"),
                false,
            ),
        ],
        (Arch::Avx2, DType::I64) => vec![
            dual_imm(
                "_mm256_shuffle_pd",
                a.clone(),
                b.clone(),
                format!("imm8_shuffle_{tag}"),
                false,
            ),
            dual_plain("_mm256_unpacklo_epi64", a.clone(), b.clone(), false),
            dual_plain("_mm256_unpackhi_epi64", a.clone(), b.clone(), false),
            dual_imm(
                "_mm256_permute2x128_si256",
                a.clone(),
                b.clone(),
                format!("imm8_perm2x128_{tag}"),
                true,
            ),
            dual_imm(
                "_mm256_blend_pd",
                a.clone(),
                b.clone(),
                format!("imm8_blend_{tag}"),
                true,
            ),
            dual_imm(
                "_mm256_alignr_epi8",
                a,
                b,
                format!("imm8_alignr_{tag}"),
                false,
            ),
        ],
        (Arch::Avx512, DType::I32) => vec![
            dual_ctrl(
                "_mm512_permutex2var_epi32",
                a.clone(),
                b.clone(),
                format!("ctrl_permutex2var_{tag}"),
                512,
                true,
            ),
            dual_imm(
                "_mm512_shuffle_ps",
                a.clone(),
                b.clone(),
                format!("imm8_shuffle_ps_{tag}"),
                false,
            ),
            dual_plain("_mm512_unpacklo_epi32", a.clone(), b.clone(), false),
            dual_plain("_mm512_unpackhi_epi32", a.clone(), b.clone(), false),
            dual_imm(
                "_mm512_shuffle_i32x4",
                a.clone(),
                b.clone(),
                format!("imm8_shuf_i32x4_{tag}"),
                false,
            ),
            dual_imm(
                "_mm512_alignr_epi32",
                a.clone(),
                b.clone(),
                format!("imm8_alignr_{tag}"),
                false,
            ),
            masked_dual_ctrl(
                "_mm512_mask_permutex2var_epi32",
                a.clone(),
                b.clone(),
                format!("ctrl_m_permutex2var_{tag}"),
                512,
                format!("k_mask_permutex2var_{tag}"),
                16,
                true,
            ),
            masked_dual_imm(
                "_mm512_mask_shuffle_ps",
                a.clone(),
                b.clone(),
                format!("imm8_m_shuffle_ps_{tag}"),
                format!("k_mask_shuffle_ps_{tag}"),
                16,
                false,
            ),
            masked_dual_plain(
                "_mm512_mask_unpacklo_epi32",
                a.clone(),
                b.clone(),
                format!("k_mask_unpacklo_{tag}"),
                16,
                false,
            ),
            masked_dual_plain(
                "_mm512_mask_unpackhi_epi32",
                a.clone(),
                b.clone(),
                format!("k_mask_unpackhi_{tag}"),
                16,
                false,
            ),
            masked_dual_imm(
                "_mm512_mask_shuffle_i32x4",
                a.clone(),
                b.clone(),
                format!("imm8_m_shuf_i32x4_{tag}"),
                format!("k_mask_shuf_i32x4_{tag}"),
                16,
                false,
            ),
            masked_dual_imm(
                "_mm512_mask_alignr_epi32",
                a,
                b,
                format!("imm8_m_alignr_{tag}"),
                format!("k_mask_alignr_{tag}"),
                16,
                false,
            ),
        ],
        (Arch::Avx512, DType::I64) => vec![
            dual_ctrl(
                "_mm512_permutex2var_epi64",
                a.clone(),
                b.clone(),
                format!("ctrl_permutex2var_{tag}"),
                512,
                true,
            ),
            dual_imm(
                "_mm512_shuffle_pd",
                a.clone(),
                b.clone(),
                format!("imm8_shuffle_pd_{tag}"),
                false,
            ),
            dual_plain("_mm512_unpacklo_epi64", a.clone(), b.clone(), false),
            dual_plain("_mm512_unpackhi_epi64", a.clone(), b.clone(), false),
            dual_imm(
                "_mm512_shuffle_i32x4",
                a.clone(),
                b.clone(),
                format!("imm8_shuf_i32x4_{tag}"),
                false,
            ),
            dual_imm(
                "_mm512_alignr_epi64",
                a.clone(),
                b.clone(),
                format!("imm8_alignr_{tag}"),
                false,
            ),
            masked_dual_ctrl(
                "_mm512_mask_permutex2var_epi64",
                a.clone(),
                b.clone(),
                format!("ctrl_m_permutex2var_{tag}"),
                512,
                format!("k_mask_permutex2var_{tag}"),
                8,
                true,
            ),
            masked_dual_imm(
                "_mm512_mask_shuffle_pd",
                a.clone(),
                b.clone(),
                format!("imm8_m_shuffle_pd_{tag}"),
                format!("k_mask_shuffle_pd_{tag}"),
                8,
                false,
            ),
            masked_dual_plain(
                "_mm512_mask_unpacklo_epi64",
                a.clone(),
                b.clone(),
                format!("k_mask_unpacklo_{tag}"),
                8,
                false,
            ),
            masked_dual_plain(
                "_mm512_mask_unpackhi_epi64",
                a.clone(),
                b.clone(),
                format!("k_mask_unpackhi_{tag}"),
                8,
                false,
            ),
            masked_dual_imm(
                "_mm512_mask_shuffle_i32x4",
                a.clone(),
                b.clone(),
                format!("imm8_m_shuf_i32x4_{tag}"),
                format!("k_mask_shuf_i32x4_{tag}"),
                16,
                false,
            ),
            masked_dual_imm(
                "_mm512_mask_alignr_epi64",
                a,
                b,
                format!("imm8_m_alignr_{tag}"),
                format!("k_mask_alignr_{tag}"),
                8,
                false,
            ),
        ],
    }
}

pub fn dispatch_intrinsic(
    name: &'static str,
    args: &BTreeMap<&'static str, BV>,
    solver: &Solver,
) -> Result<BV, SynthesisError> {
    match name {
        "_mm256_permute_ps" => {
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_permute_ps(a, Imm8::Expr(imm8.clone())))
        }
        "_mm256_permute_pd" => {
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_permute_pd(a, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm256_permute4x64_epi64" => {
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_permute4x64_epi64(a, Imm8::Expr(imm8.clone())))
        }
        "_mm256_permutexvar_epi32" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            Ok(mm256_permutexvar_epi32(a, op_idx, solver))
        }
        "_mm256_permutexvar_epi64" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            Ok(mm256_permutexvar_epi64(a, op_idx, solver))
        }
        "_mm256_permutevar_ps" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_permutevar_ps(a, b, solver))
        }
        "_mm256_permutevar_pd" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_permutevar_pd(a, b, solver))
        }
        "_mm256_shuffle_ps" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_shuffle_ps(a, b, Imm8::Expr(imm8.clone())))
        }
        "_mm256_shuffle_pd" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_shuffle_pd(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm256_unpacklo_epi32" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_unpacklo_epi32(a, b))
        }
        "_mm256_unpacklo_epi64" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_unpacklo_epi64(a, b))
        }
        "_mm256_unpackhi_epi32" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_unpackhi_epi32(a, b))
        }
        "_mm256_unpackhi_epi64" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm256_unpackhi_epi64(a, b))
        }
        "_mm256_permute2x128_si256" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_permute2x128_si256(
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm256_blend_ps" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_blend_ps(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm256_blend_pd" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_blend_pd(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm256_alignr_epi8" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm256_alignr_epi8(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm512_permutexvar_epi32" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            Ok(mm512_permutexvar_epi32(a, op_idx, solver))
        }
        "_mm512_permutexvar_epi64" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            Ok(mm512_permutexvar_epi64(a, op_idx, solver))
        }
        "_mm512_mask_permutexvar_epi32" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let a = required_arg(name, args, "a")?;
            Ok(mm512_mask_permutexvar_epi32(src, k, op_idx, a, solver))
        }
        "_mm512_mask_permutexvar_epi64" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let a = required_arg(name, args, "a")?;
            Ok(mm512_mask_permutexvar_epi64(src, k, op_idx, a, solver))
        }
        "_mm512_permute_ps" => {
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_permute_ps(a, Imm8::Expr(imm8.clone())))
        }
        "_mm512_permute_pd" => {
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_permute_pd(a, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm512_mask_permute_ps" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_permute_ps(
                src,
                k,
                a,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_mask_permute_pd" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_permute_pd(
                src,
                k,
                a,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_permutevar_ps" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_permutevar_ps(a, b, solver))
        }
        "_mm512_permutevar_pd" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_permutevar_pd(a, b, solver))
        }
        "_mm512_mask_permutevar_ps" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_permutevar_ps(src, k, a, b, solver))
        }
        "_mm512_mask_permutevar_pd" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_permutevar_pd(src, k, a, b, solver))
        }
        "_mm512_permutex2var_epi32" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_permutex2var_epi32(a, op_idx, b, solver))
        }
        "_mm512_permutex2var_epi64" => {
            let a = required_arg(name, args, "a")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_permutex2var_epi64(a, op_idx, b, solver))
        }
        "_mm512_mask_permutex2var_epi32" => {
            let a = required_arg(name, args, "a")?;
            let k = required_arg(name, args, "k")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_permutex2var_epi32(a, k, op_idx, b, solver))
        }
        "_mm512_mask_permutex2var_epi64" => {
            let a = required_arg(name, args, "a")?;
            let k = required_arg(name, args, "k")?;
            let op_idx = required_arg(name, args, "op_idx")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_permutex2var_epi64(a, k, op_idx, b, solver))
        }
        "_mm512_shuffle_ps" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_shuffle_ps(a, b, Imm8::Expr(imm8.clone())))
        }
        "_mm512_shuffle_pd" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_shuffle_pd(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm512_mask_shuffle_ps" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_shuffle_ps(
                src,
                k,
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_mask_shuffle_pd" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_shuffle_pd(
                src,
                k,
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_unpacklo_epi32" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_unpacklo_epi32(a, b))
        }
        "_mm512_unpackhi_epi32" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_unpackhi_epi32(a, b))
        }
        "_mm512_unpacklo_epi64" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_unpacklo_epi64(a, b))
        }
        "_mm512_unpackhi_epi64" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_unpackhi_epi64(a, b))
        }
        "_mm512_mask_unpacklo_epi32" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_unpacklo_epi32(src, k, a, b, solver))
        }
        "_mm512_mask_unpackhi_epi32" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_unpackhi_epi32(src, k, a, b, solver))
        }
        "_mm512_mask_unpacklo_epi64" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_unpacklo_epi64(src, k, a, b, solver))
        }
        "_mm512_mask_unpackhi_epi64" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            Ok(mm512_mask_unpackhi_epi64(src, k, a, b, solver))
        }
        "_mm512_shuffle_i32x4" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_shuffle_i32x4(a, b, Imm8::Expr(imm8.clone())))
        }
        "_mm512_mask_shuffle_i32x4" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_shuffle_i32x4(
                src,
                k,
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_alignr_epi32" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_alignr_epi32(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm512_alignr_epi64" => {
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_alignr_epi64(a, b, Imm8::Expr(imm8.clone()), solver))
        }
        "_mm512_mask_alignr_epi32" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_alignr_epi32(
                src,
                k,
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        "_mm512_mask_alignr_epi64" => {
            let src = required_arg(name, args, "src")?;
            let k = required_arg(name, args, "k")?;
            let a = required_arg(name, args, "a")?;
            let b = required_arg(name, args, "b")?;
            let imm8 = required_arg(name, args, "imm8")?;
            Ok(mm512_mask_alignr_epi64(
                src,
                k,
                a,
                b,
                Imm8::Expr(imm8.clone()),
                solver,
            ))
        }
        other => Err(SynthesisError::UnsupportedIntrinsic(other)),
    }
}

fn required_arg<'a>(
    intrinsic: &'static str,
    args: &'a BTreeMap<&'static str, BV>,
    operand: &'static str,
) -> Result<&'a BV, SynthesisError> {
    args.get(operand)
        .ok_or(SynthesisError::MissingOperand { intrinsic, operand })
}

fn symbolic(name: String, bit_width: u32) -> GadgetNode {
    GadgetNode::Symbolic(Symbolic { name, bit_width })
}

fn single_imm(name: &'static str, input: GadgetNode, imm_name: String) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![("a", input), ("imm8", symbolic(imm_name, 8))],
        isomorphic_order: true,
    }
}

fn single_ctrl(
    name: &'static str,
    input: GadgetNode,
    ctrl_key: &'static str,
    ctrl_name: String,
    bit_width: u32,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![("a", input), (ctrl_key, symbolic(ctrl_name, bit_width))],
        isomorphic_order: true,
    }
}

fn masked_single_imm(
    name: &'static str,
    input: GadgetNode,
    imm_name: String,
    mask_name: String,
    mask_bits: u32,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("src", input.clone()),
            ("k", symbolic(mask_name, mask_bits)),
            ("a", input),
            ("imm8", symbolic(imm_name, 8)),
        ],
        isomorphic_order: true,
    }
}

fn masked_single_ctrl(
    name: &'static str,
    input: GadgetNode,
    ctrl_key: &'static str,
    ctrl_name: String,
    ctrl_bits: u32,
    mask_name: String,
    mask_bits: u32,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("src", input.clone()),
            ("k", symbolic(mask_name, mask_bits)),
            (ctrl_key, symbolic(ctrl_name, ctrl_bits)),
            ("a", input),
        ],
        isomorphic_order: true,
    }
}

fn dual_plain(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![("a", a), ("b", b)],
        isomorphic_order,
    }
}

fn dual_imm(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    imm_name: String,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![("a", a), ("b", b), ("imm8", symbolic(imm_name, 8))],
        isomorphic_order,
    }
}

fn dual_ctrl(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    ctrl_name: String,
    ctrl_bits: u32,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("a", a),
            ("op_idx", symbolic(ctrl_name, ctrl_bits)),
            ("b", b),
        ],
        isomorphic_order,
    }
}

fn masked_dual_plain(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    mask_name: String,
    mask_bits: u32,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("src", a.clone()),
            ("k", symbolic(mask_name, mask_bits)),
            ("a", a),
            ("b", b),
        ],
        isomorphic_order,
    }
}

fn masked_dual_imm(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    imm_name: String,
    mask_name: String,
    mask_bits: u32,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("src", a.clone()),
            ("k", symbolic(mask_name, mask_bits)),
            ("a", a),
            ("b", b),
            ("imm8", symbolic(imm_name, 8)),
        ],
        isomorphic_order,
    }
}

fn masked_dual_ctrl(
    name: &'static str,
    a: GadgetNode,
    b: GadgetNode,
    ctrl_name: String,
    ctrl_bits: u32,
    mask_name: String,
    mask_bits: u32,
    isomorphic_order: bool,
) -> IntrinsicNode {
    IntrinsicNode {
        name,
        operands: vec![
            ("a", a),
            ("k", symbolic(mask_name, mask_bits)),
            ("op_idx", symbolic(ctrl_name, ctrl_bits)),
            ("b", b),
        ],
        isomorphic_order,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use z3::SatResult;
    use z3::ast::Ast;

    const MASKED_AVX512_INTRINSICS: &[&str] = &[
        "_mm512_mask_permutexvar_epi32",
        "_mm512_mask_permutexvar_epi64",
        "_mm512_mask_permute_ps",
        "_mm512_mask_permute_pd",
        "_mm512_mask_permutevar_ps",
        "_mm512_mask_permutevar_pd",
        "_mm512_mask_permutex2var_epi32",
        "_mm512_mask_permutex2var_epi64",
        "_mm512_mask_shuffle_ps",
        "_mm512_mask_shuffle_pd",
        "_mm512_mask_unpacklo_epi32",
        "_mm512_mask_unpackhi_epi32",
        "_mm512_mask_unpacklo_epi64",
        "_mm512_mask_unpackhi_epi64",
        "_mm512_mask_shuffle_i32x4",
        "_mm512_mask_alignr_epi32",
        "_mm512_mask_alignr_epi64",
    ];

    fn full_masked_args(tag: &str) -> BTreeMap<&'static str, BV> {
        BTreeMap::from([
            ("src", BV::new_const(format!("{tag}_src"), 512)),
            ("k", BV::new_const(format!("{tag}_k"), 16)),
            ("a", BV::new_const(format!("{tag}_a"), 512)),
            ("op_idx", BV::new_const(format!("{tag}_op_idx"), 512)),
            ("b", BV::new_const(format!("{tag}_b"), 512)),
            ("imm8", BV::new_const(format!("{tag}_imm8"), 8)),
        ])
    }

    #[test]
    fn dispatches_all_masked_avx512_intrinsics() {
        for intrinsic in MASKED_AVX512_INTRINSICS {
            let solver = Solver::new();
            let args = full_masked_args(intrinsic);
            let result = dispatch_intrinsic(intrinsic, &args, &solver)
                .unwrap_or_else(|error| panic!("{intrinsic} should dispatch: {error}"));

            assert_eq!(result.get_size(), 512, "{intrinsic} output width");
        }
    }

    #[test]
    fn masked_dispatch_preserves_z3_avx_dead_bit_canonicalization() {
        let solver = Solver::new();
        let args = full_masked_args("canonical_mask");
        let k = args.get("k").expect("test args include k").clone();
        let _ = dispatch_intrinsic("_mm512_mask_permute_pd", &args, &solver)
            .expect("masked permute_pd should dispatch");
        solver.assert(k.extract(15, 8).eq(BV::from_u64(1, 8)));
        assert_eq!(
            solver.check(),
            SatResult::Unsat,
            "dead mask bits are zeroed"
        );

        let solver = Solver::new();
        let args = full_masked_args("canonical_imm");
        let imm8 = args.get("imm8").expect("test args include imm8").clone();
        let _ = dispatch_intrinsic("_mm512_mask_alignr_epi64", &args, &solver)
            .expect("masked alignr_epi64 should dispatch");
        solver.assert(imm8.extract(7, 3).eq(BV::from_u64(1, 5)));
        assert_eq!(
            solver.check(),
            SatResult::Unsat,
            "dead imm8 bits are zeroed"
        );

        let solver = Solver::new();
        let args = full_masked_args("canonical_ctrl");
        let op_idx = args
            .get("op_idx")
            .expect("test args include op_idx")
            .clone();
        let _ = dispatch_intrinsic("_mm512_mask_permutexvar_epi64", &args, &solver)
            .expect("masked permutexvar_epi64 should dispatch");
        solver.assert(op_idx.extract(63, 3).eq(BV::from_u64(1, 61)));
        assert_eq!(
            solver.check(),
            SatResult::Unsat,
            "dead control-vector lane bits are zeroed"
        );
    }
}
