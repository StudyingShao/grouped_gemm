/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "moe/cutlass_kernels/cutlass_heuristic.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#pragma GCC diagnostic pop

#include <cuda_runtime_api.h>
#include <vector>

namespace groupedgemmformoe {

struct TileShape {
    int m;
    int n;
};

TileShape get_cta_shape_for_config(CutlassTileConfig tile_config)
{
    switch (tile_config) {
        case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            return TileShape{32, 128};
        case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
            return TileShape{64, 128};
        case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
        case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
        case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
            return TileShape{128, 128};
        default:
            throw std::runtime_error("[FT Error][get_grid_shape_for_config] Invalid config");
    }
}

bool is_valid_split_k_factor(const int64_t   m,
                             const int64_t   n,
                             const int64_t   k,
                             const TileShape tile_shape,
                             const int       split_k_factor,
                             const size_t    workspace_bytes,
                             const bool      is_weight_only)
{

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only) {
        if ((k % k_tile) != 0) {
            return false;
        }

        if ((k % split_k_factor) != 0) {
            return false;
        }

        const int k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0) {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    const int ctas_in_m_dim     = (m + tile_shape.m - 1) / tile_shape.m;
    const int ctas_in_n_dim     = (n + tile_shape.n - 1) / tile_shape.n;
    const int required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > workspace_bytes) {
        return false;
    }

    return true;
}

std::vector<CutlassTileConfig> get_candidate_tiles(const bool is_weight_only, const bool simt_configs_only)
{

    std::vector<CutlassTileConfig> simt_configs{CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};

    std::vector<CutlassTileConfig> square_configs{CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                                                  CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64,
                                                  CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64};

    std::vector<CutlassTileConfig> quant_B_configs{CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                                                   CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                                                   CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64};

    const std::vector<CutlassTileConfig> allowed_configs = is_weight_only ? quant_B_configs : square_configs;
    return simt_configs_only ? simt_configs : allowed_configs;
}

std::vector<CutlassGemmConfig> get_candidate_configs(int sm, const bool is_weight_only, const bool simt_configs_only)
{
    std::vector<CutlassTileConfig> tiles = get_candidate_tiles(is_weight_only, simt_configs_only);

    std::vector<CutlassGemmConfig> candidate_configs;
    const int                      min_stages = 2;
    const int                      max_stages = sm >= 80 ? 4 : 2;

    for (const auto& tile_config : tiles) {
        for (int stages = min_stages; stages <= max_stages; ++stages) {
            CutlassGemmConfig config{tile_config, SplitKStyle::NO_SPLIT_K, 1, stages};
            candidate_configs.push_back(config);
        }
    }

    return candidate_configs;
}

CutlassGemmConfig estimate_best_config_from_occupancies(const std::vector<CutlassGemmConfig>& candidate_configs,
                                                        const std::vector<int>&               occupancies,
                                                        const int64_t                         m,
                                                        const int64_t                         n,
                                                        const int64_t                         k,
                                                        const int64_t                         num_experts,
                                                        const int                             split_k_limit,
                                                        const size_t                          workspace_bytes,
                                                        const int                             multi_processor_count,
                                                        const int                             is_weight_only)
{

    if (occupancies.size() != candidate_configs.size()) {
        throw std::runtime_error("[FT Error][estimate_best_config_from_occupancies] occpancies and "
                                 "candidate configs vectors must have equal length.");
    }

    CutlassGemmConfig best_config;
    // Score will be [0, 1]. The objective is to minimize this score.
    // It represents the fraction of SM resources unused in the last wave.
    float config_score   = 1.0f;
    int   config_waves   = INT_MAX;
    int   current_m_tile = 0;

    const int max_split_k = n >= multi_processor_count * 256 ? 1 : split_k_limit;
    for (int ii = 0; ii < candidate_configs.size(); ++ii) {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        TileShape         tile_shape       = get_cta_shape_for_config(candidate_config.tile_config);
        int               occupancy        = occupancies[ii];

        if (occupancy == 0) {
            continue;
        }

        // Keep small tile sizes when possible.
        if (best_config.tile_config != CutlassTileConfig::ChooseWithHeuristic && m < current_m_tile
            && current_m_tile < tile_shape.m) {
            continue;
        }

        const int ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
        const int ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;

        for (int split_k_factor = 1; split_k_factor <= max_split_k; ++split_k_factor) {
            if (is_valid_split_k_factor(m, n, k, tile_shape, split_k_factor, workspace_bytes, is_weight_only)) {
                const int ctas_per_wave    = occupancy * multi_processor_count;
                const int ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * split_k_factor;

                const int   num_waves_total      = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                const float num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                const float current_score        = float(num_waves_total) - num_waves_fractional;

                const float score_slack = 0.1f;
                if (current_score < config_score
                    || ((config_waves > num_waves_total) && (current_score < config_score + score_slack))) {
                    config_score = current_score;
                    config_waves = num_waves_total;
                    SplitKStyle split_style =
                        split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig{
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages};
                    current_m_tile = tile_shape.m;
                }
                else if (current_score == config_score
                         && (best_config.stages < candidate_config.stages || split_k_factor < best_config.split_k_factor
                             || current_m_tile < tile_shape.m)) {
                    // Prefer deeper pipeline or smaller split-k
                    SplitKStyle split_style =
                        split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig{
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages};
                    current_m_tile = tile_shape.m;
                    config_waves   = num_waves_total;
                }
            }
        }
    }

    if (best_config.tile_config == CutlassTileConfig::ChooseWithHeuristic) {
        throw std::runtime_error("[FT Error] Heurisitc failed to find a valid config.");
    }

    return best_config;
}

}  // namespace groupedgemmformoe