"""Verify parameter counts for all model configurations."""

from felix_lm.config import make_m0_config, make_m2_config
from felix_lm.model import FelixLM
from felix_lm.utils import count_parameters, count_parameters_by_component


def report(name, config):
    model = FelixLM(config)
    actual = count_parameters(model)
    estimated = config.count_params()

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"  Stages: {config.num_stages}, Total layers: {config.total_layers}")
    for k, s in enumerate(config.stages):
        print(f"  Stage {k}: {s.num_streams} streams, dim={s.dim}, "
              f"layers={s.num_layers}, attn={s.attention_type}")
    print(f"  Actual params:    {actual:>12,}")
    print(f"  Estimated params: {estimated:>12,}")
    print(f"  Difference:       {abs(actual - estimated):>12,} ({abs(actual-estimated)/actual*100:.1f}%)")
    print(f"\n  By component:")
    for comp, count in sorted(count_parameters_by_component(model).items()):
        print(f"    {comp:20s}: {count:>10,}")


if __name__ == "__main__":
    report("M0 UNIFORM (baseline)", make_m0_config())
    report("M2 MSPM-HETERO (proof-of-concept)", make_m2_config())
