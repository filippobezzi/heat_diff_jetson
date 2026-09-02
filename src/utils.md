# 3D Heat Simulation Utilities & Hardware Profile

## Jetson Nano Specifications
- **CPU**: Quad-core ARM Cortex-A57 @ 1.43 GHz
- **GPU**: 128-core NVIDIA Maxwell GPU (`sm_53`) @ 921 MHz
- **Memory**: 4 GB 64-bit LPDDR4 @ 1600 MHz (Peak Bandwidth: 25.6 GB/s)
- **Peak Compute**: 236 GFLOPS (FP32), 472 GFLOPS (FP16)

## Stencil Operational Intensity (CGMA)
- **Floating-point operations per cell**: 13 FLOPs
- **Memory traffic per cell (naive)**: 8 floats = 32 Bytes
- **CGMA Ratio**: $13 / 32 \approx 0.406$ FLOP/Byte
- **Machine Ridge Point**: $236 / 25.6 \approx 9.22$ FLOP/Byte
- **Regime**: Highly Memory Bandwidth-Bound
