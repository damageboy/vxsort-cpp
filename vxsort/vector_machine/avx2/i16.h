template <>
class vxsort_machine_traits<i16, AVX2> {
public:
    typedef i16 T;
    typedef __m256i TV;
    typedef i32 TLOADSTOREMASK;
    typedef u32 TCMPMASK;
    typedef i16 TPACK;
    typedef typename std::make_unsigned<T>::type TU;

    static constexpr i32 N = sizeof(TV) / sizeof(T);
    static_assert(is_powerof2(N), "vector-size / element-size must be a power of 2");

    static constexpr bool supports_compress_writes() { return false; }
    static constexpr bool supports_packing() { return false; }

    template <i32 Shift>
    static bool can_pack(T) { return false; }

    static INLINE TLOADSTOREMASK generate_prefix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);

        return amount ? amount : N;
    }

    static INLINE TLOADSTOREMASK generate_suffix_mask(i32 amount) {
        assert(amount >= 0);
        assert(amount <= N);

        return amount ? -N + amount : -N;
    }

    static INLINE TV load_vec(TV* p) { return _mm256_lddqu_si256(p); }

    static INLINE void store_vec(TV* ptr, TV v) { _mm256_storeu_si256(ptr, v); }

    static void store_compress_vec(TV*, TV, TCMPMASK) { throw std::runtime_error("operation is unsupported"); }

    static INLINE TV load_partial_vec(TV *p, TV base, TLOADSTOREMASK mask) {
        // FML: There is only so much AVX2 stupidity one person can
        //      take in their entire lifetime, I'm personally over this crap
        std::array<T, N> base_vec;
        _mm256_storeu_si256((TV *)base_vec.data(), base);
        auto pt = (T *)p;
        auto psrc = mask > 0 ? pt : pt + N + mask;
        auto pdest = mask > 0 ? base_vec.begin() : base_vec.end() + mask;
        auto amount = abs(mask);
        std::copy_n(psrc, amount, pdest);
        return _mm256_lddqu_si256((TV *)base_vec.data());
    }

    static INLINE void store_masked_vec(TV *p, TV v, TLOADSTOREMASK mask) {
        memcpy(p, &v, sizeof(T) * mask);
    }

    static INLINE TV partition_vector(TV, i32) {
        // Should never be called, since we "hijack" 16b/avx2 partitioning with template
        // specializtion with partition_machine<T>
        throw std::runtime_error("operation is unsupported");
    }

    static INLINE TV broadcast(T pivot) { return _mm256_set1_epi16(pivot); }

    static INLINE TCMPMASK get_cmpgt_mask(TV a, TV b) {
        return _pext_u32(
                _mm256_movemask_epi8(_mm256_cmpgt_epi16(a, b)),
                0x55555555);
    }

    static INLINE TV shift_right(TV v, i32 i) { return _mm256_srli_epi16(v, i); }
    static INLINE TV shift_left(TV v, i32 i) { return _mm256_slli_epi16(v, i); }

    static INLINE TV add(TV a, TV b) { return _mm256_add_epi16(a, b); }
    static INLINE TV sub(TV a, TV b) { return _mm256_sub_epi16(a, b); };

    static INLINE TV pack_unordered(TV, TV) { throw std::runtime_error("operation is unsupported"); }
    static INLINE void unpack_ordered(TV, TV&, TV&) { }

    template <i32 Shift>
    static INLINE T shift_n_sub(T v, T sub) {
        if (Shift > 0)
            v >>= Shift;
        v -= sub;
        return v;
    }

    template <i32 Shift>
    static INLINE T unshift_and_add(TPACK from, T add) {
        add += from;
        if (Shift > 0)
            add = (T) (((TU) add) << Shift);
        return add;
    }
};
